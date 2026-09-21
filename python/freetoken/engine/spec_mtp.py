"""MTP speculative decoding — verify-chain logic (wave 1, eager, greedy).

Depth-1 speculative flow per scheduler iteration (docs/mtp-plan.md 1.1-1.5):

  1. normal trunk decode step: input t_k -> logits -> t_{k+1}; hidden H_k
  2. MTP draft: carry H_k, token t_{k+1} -> d   (MTP-layer KV row written)
  3. verify: one eager trunk forward over TWO rows per request —
       row A: t_{k+1}  (the normal next decode input)
       row B: d        (the speculative guess for k+2)
     row A's argmax a is the trunk's true k+2 token. Accept d iff a == d:
       accepted -> emit [t_{k+1}, d], next input = row B's argmax (k+3)
       rejected -> emit [t_{k+1}], next input = a, roll back row B's KV
       slots and device_len.

Losslessness: greedy verification compares trunk argmaxes only — the
emitted token sequence is byte-identical to non-spec greedy by
construction (a rejected draft is replaced by the trunk's own argmax
computed at the same position).

The verify forward runs eagerly (no CUDA graphs): the MTP path forces
graph-capture exclusion, and the verify batch is built by the scheduler
hook cloning the decode batch's bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

@dataclass
class SpecStats:
    drafted: int = 0
    accepted: int = 0
    # per-position accept counts (stage 4 telemetry): accepted_at[j] =
    # how many drafts were accepted at chain position j in the window.
    accepted_at: list = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


def verify_chain(target_argmaxes: torch.Tensor,
                 draft_tokens: torch.Tensor) -> torch.Tensor:
    """Greedy acceptance chain (pure logic, unit-testable).

    target_argmaxes: [B, n+1] — the verify forward's per-row argmaxes;
        row i predicts the token at position i+1.
    draft_tokens: [B, n] — the drafted tokens for positions 1..n.

    Returns an integer mask [B, n]: mask[b, i] = 1 iff draft token i is
    accepted (all earlier drafts accepted and the trunk's argmax at the
    row that predicts position i+1 equals draft_tokens[b, i]).
    """
    accepted = target_argmaxes[:, : draft_tokens.shape[1]] == draft_tokens
    # a later draft is only valid if every earlier one was accepted
    return accepted.cumprod(dim=1, dtype=torch.int32)

@dataclass
class PerReqStep:
    """The per-request resolve outcome for one spec iteration (any depth).

    Mirrors tests/engine/test_mtp_loop.py's IterationResult / the semantics
    doc's pseudocode exactly. Depth-1 is the k=0/k=1 special case.
      k accepts (rows 0..k-1 argmax == their drafts): emit
      [d_0, ..., d_{k-1}, bonus]; next input = bonus; carry = row k's
      hidden; no rollback.
      reject at draft k: emit [a_0, ..., a_k] (the trunk argmaxes up to
      and including the mismatching row); next input = a_k; carry = row
      k's hidden; roll back rows k+1..n (free their page slots, rewind
      device_len, restore the mid-verify GDN snapshot).
    """
    accepted: int             # count of accepted drafts k (0..depth); truthy iff k > 0
    draft_token: int          # depth-1 alias: drafts[0] (always present)
    drafts: tuple[int, ...]   # the offered draft chain (d_0..d_{n-1})
    emitted: tuple[int, ...]  # (d_0..d_{k-1}, a_k): the certain emissions
    next_input: int           # the trunk argmax the next iteration consumes
    carry_row: int            # k: carry = the trunk hidden of verify row k


def resolve_chain(row_argmaxes: Sequence[int], drafts: Sequence[int]) -> PerReqStep:
    """Greedy chain resolve (pure logic, CPU-testable): accept the longest
    prefix of drafts whose verify-row argmax matches; the mismatching
    (or final) row's argmax is the certain token.

    row_argmaxes: (n+1,) — verify row j's argmax predicts position j+1,
        i.e. the trunk's opinion of draft j (and of the final position).
    drafts: (n,) — the drafted tokens for positions 1..n.
    """
    drafts = tuple(drafts)
    k = 0
    while k < len(drafts) and row_argmaxes[k] == drafts[k]:
        k += 1
    return PerReqStep(
        accepted=k, draft_token=drafts[0], drafts=drafts,
        emitted=drafts[:k] + (row_argmaxes[k],),
        next_input=row_argmaxes[k], carry_row=k)


def resolve_step(row_a_argmax: int, row_b_argmax: int,
                 draft_token: int) -> PerReqStep:
    """Depth-1 resolve: accept iff the verify row-A argmax equals the
    draft. Delegates to the general chain resolve."""
    return resolve_chain((row_a_argmax, row_b_argmax), (draft_token,))


def batch_resolve_chain(row_argmaxes: torch.Tensor,
                        drafts: torch.Tensor) -> list[PerReqStep]:
    """Batched chain resolve. row_argmaxes [B, n+1], drafts [B, n] int
    tensors; returns one PerReqStep per request. Row n's argmax is
    computed for every request (the verify forward yields it regardless);
    it is only *meaningful* where k == n."""
    rows = row_argmaxes.tolist()
    ds = drafts.tolist()
    return [resolve_chain(rows[i], ds[i]) for i in range(len(ds))]


def batch_resolve(row_a_argmaxes: torch.Tensor, row_b_argmaxes: torch.Tensor,
                  drafts: torch.Tensor) -> list[PerReqStep]:
    """Batched depth-1 resolve. [B] int tensors each; delegates to the
    general chain resolve with n=1."""
    return batch_resolve_chain(
        torch.stack([row_a_argmaxes, row_b_argmaxes], dim=1),
        drafts.unsqueeze(1))

def per_position_accepts(steps: Sequence[PerReqStep], depth: int) -> list[int]:
    """Per-position accept counts over one spec iteration: result[j] =
    how many requests accepted draft j (0-indexed), for the depth
    offered drafts. The plan's per-position acceptance signal."""
    return [sum(1 for s in steps if s.accepted > j) for j in range(depth)]


@dataclass
class SpecResult:
    """One spec iteration's batch outcome (the engine-side SpecStep result).

    next_input: [B] int32 — the token the NEXT decode iteration consumes
    (the certain input at position q+2 after accept, q+1 after reject).
    carry: [B, H] — the trunk hidden the next draft consumes (row B's on
    accept, row A's on reject — the semantics doc's Q2).
    emitted_tokens: [B, 2] padded with -1: (draft, bonus) on accept,
    (a, -1) on reject. emitted_lens: [B] — 2 or 1.
    rollback: [B] bool — reject requests needing the row-B rollback.
    """
    next_input: torch.Tensor
    carry: torch.Tensor
    emitted_tokens: torch.Tensor
    emitted_lens: torch.Tensor
    accepted: torch.Tensor
    rollback: torch.Tensor

    @property
    def drafted(self) -> int:
        return int(self.emitted_tokens.shape[0])

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum().item())

    @property
    def rate(self) -> float:
        d = self.drafted
        return self.n_accepted / d if d else 0.0


def commit_reqs(reqs, result: SpecResult) -> None:
    """Advance the reqs to the next iteration's input state from a
    SpecResult (pure bookkeeping; mirrors the loop model's device_len/q
    arithmetic). Complete pre-condition: the verify forward already
    advanced each req by 2 device positions (complete_one-style, called
    twice — see forward_batch's spec arm); this fn consumes that state.

    Per req i (all on host tensors, mirroring complete_one + append_host):
      accept: emit [d, b]; append both; next input = b; the req now sits
              at device_len = q+2 with cached_len advanced past row B —
              the bonus b is the input at q+2 (one past the written
              region; next iteration's 2-slot advance allocates it).
      reject: emit [a]; append it; next input = a; rewind device_len by 1
              (row B's position un-commits — the caller frees the page
              slot / restores GDN state via the returned rollback mask).
    """
    emitted = result.emitted_tokens.tolist()
    lens = result.emitted_lens.tolist()
    accepted = result.accepted.tolist()
    rollback = result.rollback.tolist()
    next_input = result.next_input.tolist()
    for i, req in enumerate(reqs):
        req.append_host(
            torch.tensor(emitted[i][: lens[i]], dtype=req.input_ids.dtype))
        if rollback[i]:
            # Row B's KV page slot is freed by the caller (needs the page
            # table); here the device_len rewind: the req's frontier moves
            # back by 1 so the next iteration re-processes the corrected
            # position as its verify row A.
            req.device_len -= 1
            req.cached_len -= 0  # cached_len was already advanced to q
        # The next input token rides on the req: it is written by the
        # scheduler drain's append_host path? No — the drain appends the
        # EMITTED tokens. The NEXT INPUT token is the last emitted token
        # by construction (accept: next == bonus == last emitted; reject:
        # next == a == the only emitted), so the drain's appends produce
        # exactly the state the next iteration consumes. assert it.
        assert req.input_ids[-1].item() == next_input[i], (
            f"spec loop invariant violated: last emitted token "
            f"{req.input_ids[-1].item()} != next input {next_input[i]}")

class MTPDrafter:
    """The MTP draft step's engine-side owner: stats + the batched eager
    draft call. Built once at engine init under --spec-mtp.

    History: this class was defined twice back-to-back (the wave-1
    integration left both a SpecStep-shaped and a MTPDrafter-shaped copy
    behind); consolidated 2026-09-14. The two ``draft`` methods were
    byte-equivalent modulo parameter names -- (carry, tokens) vs
    (hidden, next_tokens) -- and NEITHER has a production call site: the
    scheduler's spec loop calls ``mtp.draft_step`` directly
    (scheduler.py _prepare_spec_batch), the engine's resolve uses the
    drafter for stats only, and stage 2's graph replay bypasses the eager
    path entirely. ``draft`` stays as the eager reference the stage-2
    bit-equality gate exercises.

    The verify forward + req bookkeeping live in the scheduler hook (the
    verify batch clones the decode batch's page allocation); the loop's
    pure resolve logic is the module functions above.
    """

    def __init__(self, engine):
        self.engine = engine
        self.mtp = engine.model.model.mtp
        assert self.mtp is not None, "--spec-mtp requires an MTP-capable checkpoint"
        self.stats = SpecStats()

    def draft(self, carry: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """One batched draft step: [B, H] carries (the trunk's post-norm
        hidden at the just-sampled position) + [B] sampled tokens -> [B]
        drafted tokens. The draft layer's KV row is written through the
        trunk attention (layer_id = num_layers)."""
        _, logits = self.mtp.draft_step(carry, tokens)
        return logits.argmax(dim=-1).to(torch.int32)
