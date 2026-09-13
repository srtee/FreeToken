# FreeToken MTP Speculative Decoding — Implementation Plan

Port MTP-based speculative decoding into FreeToken so Qwen3.6-35B-A3B (and later
Qwen3.8-Flash-Next) decode with their native MTP head, staged for a local coding
agent. Verified against both trees:

- **FreeToken**: the checkpoint carries 19 MTP tensors (`mtp.fc`,
  `mtp.norm`, `mtp.pre_fc_norm_{embedding,hidden}`, `mtp.layers.0.*` — a full
  transformer layer with its own routed + shared experts). `text_config`
  declares `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`.
  Today `weight.py` *drops* these tensors (`raw_name.startswith("mtp.")`).
- **buun-llama-cpp reference**: `src/models/qwen35moe.cpp::graph_mtp` (the draft
  graph: enorm/hnorm RMSNorms → concat → eh_proj → transformer layer over shared
  KV) and `common/speculative.cpp` (~7K lines; the MTP-specific parts are
  `common_speculative_mtp_context_params_resolve`, the carry lifecycle
  `common_speculative_mtp_carry_lifecycle`, and the target-verification loop).

## Key FreeToken facts the agent needs (verified)

| Fact | Where |
|---|---|
| Decode loop: `Scheduler._forward` → `engine.forward_batch` → `graph_runner.replay` → `Sampler.sample` | `python/freetoken/scheduler/scheduler.py:866`, `python/freetoken/engine/engine.py` |
| Graphs capture per-BS decode with `dummy_req` padding; `GraphCaptureBuffer` owns I/O tensors | `python/freetoken/engine/graph.py:94` |
| Hidden state before lm_head is available inside the model but **not returned** — `Qwen3_5MoEForCausalLM.forward` returns only logits | `python/freetoken/models/qwen3_5_moe/model.py:111` |
| KV write goes through `attn_backend.forward` → `kvcache.store_kv(k, v, batch.out_loc, layer_id)`; the MTP layer can reuse the **same pool** at `layer_id = num_layers` (one extra slot row per request) | `python/freetoken/attention/fa.py:69` |
| Per-request state lives on `Req` (`core.py:34`); overlap scheduling means abort during in-flight forward is handled by `aborted` flag + drain | `python/freetoken/core.py` |
| Sampler: greedy = argmax; sampled path `sample_impl` | `python/freetoken/engine/sample.py:76` |
| `max_running_requests` default 4; MTP targets low concurrency | `ServerArgs` |
| Checkpoint MTP tensors are NVFP4-quantized for experts, bf16 elsewhere (same layout as trunk layers) | checkpoint index |

## Architecture: draft loop inside the engine

```
decode step (per batch, bs ≤ max_running_requests):
  1. target forward (existing graph) → logits T0, hidden H0
  2. sample token t1 from T0                        (target-verified by construction)
  3. MTP draft forward, depth n ∈ [1..n_max]:
       input = [t_i embedding (enorm-normed) ‖ H_{i-1} (hnorm-normed)] → fc → MTP layer
       → hidden H_i → (shared lm_head) logits D_i
       sample t_{i+1} ~ D_i  (greedy or target-temp)
  4. target verify forward on [t1..t_{n+1}] (one decode-shaped batch, n+1 rows/req)
       → argmax chain; accept longest matching prefix
  5. roll back KV for rejected suffix; set next input to first rejected position's token
```

The MTP layer shares the target's KV cache pool and page table (buun's draft
context reuses target KV — its `cparams_mtp` mirrors target geometry), so no
second KV pool. The MTP layer's KV row lives at a reserved extra layer index.
`mtp_use_dedicated_embeddings: false` means the draft reuses the target embedding
table and lm_head — no duplicated vocab tensors (matches buun's "shared" sidecar
mode, and FreeToken's checkpoint has no separate MTP embedding).

Losslessness: target always verifies over the full vocabulary; a draft can only
change speed, never output (greedy identical, sampled distribution preserved
when drafting at target temperature).

---

# Wave 0 — Weight loading + MTP layer as a plain module

**Goal**: load the MTP tensors, build the module, verify one draft step
numerically against a reference. No scheduler changes.

## Tasks

0.1 **Stop dropping MTP weights** — `python/freetoken/models/qwen3_5_moe/weight.py`:
- Add `mtp.*` patterns to the load list (keep the `model.visual.*` drop).
- Experts: `mtp.layers.0.mlp.experts.{down_proj,gate_up_proj}` are NVFP4 fused
  tensors routed through `nvfp4_banks.py` — give the MTP layer its own bank or a
  dense (dequantized-on-load) fallback first. Bank-per-layer is cleaner: the
  trunk bank index is `layer`; MTP needs a synthetic index (e.g. `num_layers`).
- `mtp.fc`, `mtp.norm`, `mtp.pre_fc_norm_*`, `mtp.layers.0.self_attn.*`,
  `mtp.layers.0.{input,post_attention}_layernorm.weight`, shared-expert tensors:
  plain bf16, straight copy.

0.2 **Config plumbing** — `config.py`: expose
`mtp_num_hidden_layers` (assert == 1 when present), `mtp_use_dedicated_embeddings`.

0.3 **Module** — `python/freetoken/models/qwen3_5_moe/mtp.py`:
```python
class MTPHead(torch.nn.Module):
    """Qwen3.5/3.6 single-block MTP draft head (DeepSeek-V3 style)."""
    # weights: fc [hidden*2, hidden], pre_fc_norm_{embedding,hidden}, norm (final RMS),
    #          one decoder layer (reuse Qwen3_5DecoderLayer with layer_id=num_layers),
    #          shared embedding + lm_head by reference.
    def forward(self, last_hidden_normed, input_ids, positions, attn_metadata, layer_kv):
        e = self.enorm(self.embed_tokens(input_ids))     # pre_fc_norm_embedding
        h = self.hnorm(last_hidden_normed)               # pre_fc_norm_hidden
        x = self.fc(torch.cat([e, h], dim=-1))
        x, _ = self.layer.forward(x, None)               # writes KV to reserved layer row
        return self.norm(x)                              # post-norm hidden for next step / lm_head
```
Port semantics from buun `graph_mtp` (order: enorm/hnorm → concat → fc →
attn_norm → QKV attention → MoE → post_attention_layernorm). Note buun feeds the
MTP block's own `attn_norm`; confirm against the HF reference implementation
(`Qwen3_5MoeForCausalLM` MTP / DeepSeek-V3 nextn) — the checkpoint's
`mtp.norm.weight` is the block-final norm whose output goes to lm_head.

0.4 **Numerical oracle** — `scripts/mtp_oracle.py`:
- Load trunk + MTP via transformers on CPU (or GPU, bf16) from the snapshot;
  run 5 tokens of decode; capture hidden rows.
- Run FreeToken trunk + MTPHead; compare hidden rows and draft logits.
- Gate: draft logits cosine sim > 0.999 vs HF reference, argmax identical on
  greedy for 5 consecutive steps.

## Verification
- `pytest tests/models/qwen3_5_moe/test_mtp_load.py` — all 19 tensors present with
  right shapes/dtypes; expert bank count = layers + 1.
- Oracle script passes the cosine/argmax gates.
- Deliverable: numbers in `docs/mtp-baseline-numbers.md`.

---

# Wave 1 — Draft step integration (eager path, no graphs)

**Goal**: end-to-end speculative decode in eager mode with greedy sampling.
Correctness over speed; CUDA graphs in Wave 2.

## Tasks

1.1 **Engine plumbing** — `python/freetoken/engine/engine.py`:
- Construct `MTPHead` when `config.mtp_num_hidden_layers > 0` and
  `--spec-mtp` (new flag, default off). KV pool gets one extra storage layer
  (`num_layers + 1`); document the small VRAM cost (~0.3% of pool).
- Expose trunk hidden state: change model `forward()` to optionally return the
  pre-lm_head hidden (`self.model.forward(...)` output) alongside logits.
  Cheapest route: a flag on `get_global_ctx()` or a second forward entry point
  used only when spec is on, so graph-only paths stay untouched.

1.2 **Draft step** — new `python/freetoken/engine/spec_mtp.py`:
```python
class MTPDrafter:
    def draft(self, batch, hidden0, tokens0, n_draft: int) -> DraftResult:
        # iteratively: embed → concat → fc → layer.forward → norm → lm_head → sample
        # writes each draft token's KV row into pool layer `num_layers`
        # returns draft_tokens [bs, n_draft], draft logits (for p_min gating)
```
Runs on the engine stream after the target decode step. Attention metadata for
the MTP layer rows: reuse the decode metadata machinery with `layer_id =
num_layers` — verify `fa.py`/`fi.py` build per-layer metadata (they do:
metadata is per-batch, cache is per-layer).

1.3 **Verify step** — same module:
- Build a decode-shaped `Batch` whose per-request token list is
  `[t_next, d1..dn]` (n+1 rows). Feed through the normal `engine.forward_batch`
  path. Compare argmax chain: accepted = longest prefix where target argmax ==
  drafted token (greedy). Sampled case in Wave 3 (p/q matching needs draft
  probs + target probs; store draft logits row-wise).
- On accept count `k`: accepted tokens = `d1..dk` + the next target token from
  verify position k. Roll back KV rows for positions `> k+1` in BOTH the trunk
  layers and the MTP layer (free page slots via `out_loc` tombstone or page-table
  pop — check `store_kv`'s allocator: `Req` device_len rewinds and page table
  truncates; follow `abort`/rollback precedent in `scheduler.py` cache manager).
- Rewind `Req.cached_len/device_len` and re-point `input_ids` view; the existing
  `_alloc_ids_buf` pattern supports truncation (keep the full max buffer).

1.4 **Scheduler wiring** — `scheduler.py::_forward`: when spec enabled and
`batch.is_decode` and all reqs greedy: target step → draft → verify → merge
accepted tokens into `forward_output.next_tokens_gpu` bookkeeping (output_len
accounts multiple tokens per step; `DecodeManager.inflight_tokens` already uses
`remain_len`, so it stays correct).

1.5 **Stats** — extend the decode log line with
`#drafted: N, #accepted: k (rate r)`, mirroring buun's counters
(`n_acc_drafts`, `n_acc_tokens`, per-position acceptance).

## Verification
- Unit: verify-chain logic on synthetic logits (known accept counts) —
  `tests/engine/test_mtp_verify.py`.
- E2E eager: `ft serve --spec-mtp` on Qwen3.6, greedy, 512-token generation:
  - Output **byte-identical** to non-spec greedy run (the core losslessness gate).
  - Log shows acceptance rate ≥ 55% on prose (buun saw 56% on Qwen3.5-27B at
    depth 5; depth 1–2 on 35B-A3B should land ~60–75% for depth 2).
- Perf: eager spec must not be slower than eager non-spec (draft + verify costs
  ~2 extra small forwards; eager overhead dominates, so gate is merely "not a
  regression" — the real numbers come with graphs).

---

# Wave 2 — CUDA graphs for draft + verify

**Status: SUPERSEDED by docs/mtp-wave2-plan.md (2026-09-13).** Wave-1's
parked verify design (below in 1.3) ran one structurally wasted forward
per iteration; the corrected loop, stage breakdown, and gates live in
the wave-2 plan. The economics correction: the verify forward IS the
next-token producer (one 2-row forward per iteration, not
fwd1 + a separate 2-row verify); eager ceiling ≈ (1+r)/(c2+cd), not
"≤ 1.0". This section is kept for history.

## Tasks

2.1 **Draft graph** — extend `GraphRunner` (or a sibling `MTPGraphRunner`) to
capture the draft step per BS in the same bs list: inputs = hidden0 row, token
ids; outputs = draft hidden + logits. Static shapes only (bs padded with
`dummy_req`; n_draft fixed per capture at `--spec-draft-n-max`).

2.2 **Verify graph**: capture decode graphs at `n+1` tokens per request instead
of 1 when spec is on. Two options:
- (a) capture a second graph family keyed `(bs, seq_len=n+1)` — simplest, but
  doubles graph memory;
- (b) make the existing decode graph length-parameterized (token count per row
  as an input tensor). Check `_make_positions`/`_make_input_tuple`: they already
  handle extend batches with arbitrary per-req lengths on the eager path; the
  captured decode graph fixes `extend_len=1`. Option (b) is the right long-term
  shape but touches the capture invariants; start with (a), measure graph VRAM
  (n+1=3 rows vs 1 → ~3× KV metadata per graph, acceptable), and only attempt
  (b) if memory hurts.
- MTP-layer KV rows in the verify graph: the MTP layer must read/write KV for
  the *drafted* positions — i.e. the verify batch itself runs the trunk only;
  MTP hidden carry comes from the last accepted draft state (see 2.3).

2.3 **Hidden carry** — the draft's next input is the target hidden at the last
accepted position, which the verify forward produces for free (its last row).
Keep a per-req `[max_bs, hidden]` scratch row (`mtp_carry`) written by the verify
graph's hidden output; buun's `common_speculative_mtp_carry_lifecycle` is the
reference for the abort/rollback edge cases (pending hidden row is process-local
state; sequence images must not carry it).

2.4 **Rollback under graphs**: page-table and cached_len rewinds happen outside
captured regions (scheduler-side) — verify this holds; the KV slot freeing must
not be captured.

## Verification
- Byte-identical greedy output vs eager spec run (same seeds/prompts).
- Perf gate on Qwen3.6 35B-A3B, bs=1, greedy, 8K ctx: **≥ 1.6× decode t/s** vs
  the 125 t/s baseline (expect ~190–220 t/s at depth 2 with ~70% acceptance;
  buun's fused MTP numbers on bigger models support this range).
- Graph VRAM check: total VRAM within the tuned budget (the `--num-tokens
  49152 --moe-cache-rate 0.25` config from the omp session); if graphs push over,
  reduce `cuda_graph_max_bs` before touching KV.
- Abort mid-verify test: abort a request between draft and verify; assert no
  dangling KV rows (follow `test_abort_inflight_prefill.py` conventions).

---

# Wave 3 — Sampled decoding, depth tuning, production hardening

3.1 **Sampled verification (p/q)** — extend verify to sample-accept:
target probs `q` and draft probs `p` (stored in 1.3); accept token with prob
`min(1, q/p)`; on rejection resample from the renormalized residual
`(q - p)+`. Gate behind non-greedy paths; greedy stays the fast path.
Reference: standard speculative sampling; buun implements the same in
`common_speculative.cpp` (its `p_min` pre-gate is an optimization: skip drafting
tokens whose draft prob is below threshold).

3.2 **Depth auto-tune** — measure per-position acceptance online
(`n_acc_tokens_per_pos`); expose `--spec-draft-n-max` and pick the best depth
from the log after warmup. Buun's finding: depth 2 beat 3–5 on their CPU-heavy
setup; on this box (fast GPU, offloaded experts) the verify cost is dominated by
expert fetch misses, so deeper drafts amortize better — measure, don't assume.

3.3 **Concurrency guardrails**: spec decoding with bs > 1 multiplies verify
batch rows; enforce `--spec-mtp` implies `--max-running-requests` clamp (warn
when unset), matching buun's "intended primarily for low-concurrency serving."

3.4 **Interaction with MoE offload**: draft + verify steps issue more expert
fetches per generated token; the LRU expert cache hit rate should *improve*
(deeper verification = more experts per fetch = better prefetch amortization).
Watch `ft ctl stats` cache hit metrics during the soak; if hit rate collapses,
auto-lower depth.

3.5 **FTW/checkpoint**: confirm `ft checkpoint` conversion keeps MTP tensors
(weight map passthrough) so FTW-served models can use spec too.

3.6 **Docs + CLI**: `--spec-mtp`, `--spec-draft-n-max`, `--spec-draft-p-min`,
`--spec-mtp-vocab-size` (reserved; the vocab-trim repack is a Wave 3+ optional —
see non-goals), server docs, models.md notes.

## Verification
- Sampled-vs-nonspec distribution test: same prompt, fixed seed, sampled outputs
  from spec and non-spec runs are both valid samples (statistical test on 200
  generations, KS test on token frequencies — no systematic shift).
- 30-min omp agent soak at `--spec-mtp`: acceptance telemetry healthy, no
  drift/loops, `ft ctl stats` clean.
- Final table in `docs/mtp-baseline-numbers.md`: t/s and acceptance at
  depth 1/2/3/4, bs 1/2/4, greedy + sampled, vs the 125 t/s f16-KV baseline and
  the TCQ-plan baselines.

---

# Non-goals (this plan)

- **Vocab-trim repack** (`mtp-vocab-trim.cpp`, d2t 32K map): only needed for
  standalone MTP sidecars of Qwen-27B-family checkpoints to cut lm_head cost.
  FreeToken's MTP head shares the target lm_head and runs a single layer; the
  full-vocab draft logits cost one GEMV per draft step — measure before adding
  the repack. If the draft logits GEMV shows up in profiles (>8% of step),
  port the d2t map as a follow-up.
- **DFlash/DFlash2/DSpark sidecars** (block-diffusion drafters): separate plan;
  needs a drafter-loading surface (external GGUF sidecar) first.
- **EAGLE-3**: different training requirement (needs trained draft weights).
- Multi-GPU TP: draft head is TP-sharded like a trunk layer (experts banked per
  rank); defer testing until a second GPU exists, assert TP=1 correctness.

# Agent execution notes

- One wave per agent session; each wave's Verification gates must pass before
  the next wave starts. Wave 0 and Wave 1 are each ~one session; Wave 2 is the
  big one (graph machinery) — split into 2.1+2.3 (draft graph + carry) and 2.2
  (verify graphs) if the session is tight.
- The MTP layer reuses `Qwen3_5DecoderLayer` — do not fork the layer class; pass
  `layer_id=num_layers` and let the existing prefix/weight-name machinery load
  `mtp.layers.0.*` into it (a thin name-mapping shim in `weight.py` is cleaner
  than renaming weights on disk).
- The 19 checkpoint tensors are the contract. Any extra tensor the HF
  implementation expects is a bug in the oracle, not the checkpoint.
- Losslessness (greedy byte-identical) is the primary invariant — every wave's
  E2E gate re-asserts it. A spec implementation that changes greedy output is
  broken no matter how fast.
- Local coder model: `omp --model freetoken/qwen3.6-35b-a3b` (FreeToken on
  1919). Baseline numbers to beat: 125 t/s decode, ~0.9 GiB KV at 49K tokens.
- buun reference paths: `src/models/qwen35moe.cpp::graph_mtp`,
  `common/speculative.cpp` (carry lifecycle + verify loop),
  `common/mtp-vocab-trim.cpp` (future vocab repack), `docs/speculative.md`.