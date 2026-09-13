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

from dataclasses import dataclass

import torch


@dataclass
class SpecStats:
    drafted: int = 0
    accepted: int = 0

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

class MTPDrafter:
    """Runs the MTP draft step after the target decode step. The verify
    forward + req bookkeeping live in the scheduler hook (the verify batch
    clones the decode batch's page allocation)."""

    def __init__(self, engine):
        self.engine = engine
        self.mtp = engine.model.model.mtp
        assert self.mtp is not None, "--spec-mtp requires an MTP-capable checkpoint"
        self.stats = SpecStats()

    def draft(self, hidden: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
        """One draft step for the whole batch: carry = the trunk's post-norm
        hidden at the just-sampled position, token = the sampled next token.
        Returns the drafted token per request [B]. The MTP layer's KV row
        is written through the trunk attention (layer_id = num_layers)."""
        carry, logits = self.mtp.draft_step(hidden, next_tokens)
        return logits.argmax(dim=-1).to(torch.int32)
