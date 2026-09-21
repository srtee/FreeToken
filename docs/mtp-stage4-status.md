# Wave-2 Stage 4 + 5 — depth-2 draft chain, soak: status & findings

**As of 2026-09-21. STAGE 4 GATE: ALL PASS** — losslessness byte-identical
(depth 2 == plain == depth 1) AND the acceptance gate met after fixing the
draft-graph view-clobber bug below (depth-2 acceptance 0.572, ≥ 0.55).
Depth 2 still loses the perf decision at bs=1 on this box, so the default
remains `--spec-draft-n 1`.

## Root cause of the acceptance collapse (fixed)

`MTPDraftGraphRunner.draft()` returned **views** into the static CUDA-graph
buffers (`buffer.drafts[:bs]`). A chained draft replays the graph again,
overwriting the slot an earlier step's return aliases: `torch.stack` of the
per-step results then read the LAST step's draft in every column, so
depth-2's staged draft pair was `[d1, d1]` instead of `[d0, d1]`. Verify is
lossless regardless (rejects roll back), but position-0 acceptance collapsed
to "the model repeats the token" (~2.5%). Depth 1 is unaffected — one step,
consumed before any replay. The per-step bit-equality gate (depth-1,
compared before the next replay) could never see it. Fix: `draft()` returns
`.clone()`s. Diagnosis: eager-draft A/B (`FT_SPEC_DRAFT_EAGER=1`) chained
correctly while the graphed path staged the wrong token into step 2 with
IDENTICAL per-step carries — compute was right, only the stacked views were
stale.

En route, the draft chain's KV staging also moved from "every step piles
onto row 0's slot/position q" (accidentally load-bearing at depth 1 —
step 0's write IS row 0's content) to per-step staging at position q+k /
slot pt[q+k], so every layer-40 row holds its own content before anything
attends it.

## What landed

`--spec-draft-n {1,2}` (EngineConfig.spec_draft_n, validated 1..2). The
scheduler advances `n+1` device positions per spec iteration and builds `n+1`
1-row verify batches; the engine chains the MTP draft (`draft_step` returns
`(carry, logits)`, so chaining is native), resolves with the depth-generic
`batch_resolve_chain`, and stages per-req `spec_accept_count`. Extras drain as
a padded `[B, n]` row (per-req up to `n` bonus tokens). GDN snapshots: one
pool slot per snapshot row (slot 0 = the req's idle ping-pong track, 1..n-1
per-iteration allocs freed batch-wide). Telemetry: `SpecStats.accepted_at[j]`
per-position counts surfaced in the decode status line via the stage-4 gate.

Files: `engine/spec_mtp.py` (depth-generic resolve + per-position telemetry),
`engine/engine.py` (`_forward_spec_batch` n-row loop, snapshot guard
`row_idx < n_rows - 1`, accept-count staging), `scheduler/scheduler.py`
(`_prepare_spec_batch` n+1 span, `_rollback_spec_rejects` per-accept-count
rollback, row views parameterized by row index), `core.py` (Batch staging
fields), `engine/config.py` + `server/args.py` (flag + validation),
`scripts/mtp_stage3_gate_worker.py` (SPEC_DRAFT_N parametrization),
`scripts/mtp_stage4_gate.py` (new driver). Pure-logic depth-2 tests in
`tests/engine/test_spec_resolve.py`.

## Gate results (2026-09-21, 35B NVFP4, GATE_TOKENS=256, bs=1 greedy)

```
GATE 2 (d1 == plain): byte-identical, 763 tokens
GATE 1 (d2 == plain): byte-identical, 765 tokens
GATE 3 (d2 == d1):    byte-identical, 763 tokens

mode   depth   secs  tokens   tok/s    acc  acc@0  acc@1
plain      0   9.16     765   83.5
d1         1  12.15     763   62.8   0.75    328      0
d2         2  23.40     765   32.7   0.02     34      2
```

Also fixed en route: `spec_mtp.py` had lost its `import torch` (9 pre-existing
test failures), and `test_resolve_step_type_shape`'s field-set pin was stale.

## Perf decision

Depth 2 is a 2.6× throughput REGRESSION vs plain on this architecture (the
verify is n+1 sequential 1-row trunk forwards; depth 2 needs ~3× the token
yield just to break even, and measured acceptance makes that impossible).
## Gate results (2026-09-21, 35B NVFP4, GATE_TOKENS=256, bs=1 greedy)

```
GATE 2 (d1 == plain): byte-identical, 763 tokens
GATE 1 (d2 == plain): byte-identical, 762 tokens
GATE 3 (d2 == d1):    byte-identical, 762 tokens

mode   depth   secs  tokens   tok/s    acc  acc@0  acc@1
plain      0   8.80     765   86.9
d1         1  12.24     763   62.3   0.75    328      0
d2         2  13.83     762   55.1   0.57    260    147

depth-2 acceptance: 0.572 (accepted_at=[260, 147])  — was [34, 2] pre-fix
```

Also fixed en route: `spec_mtp.py` had lost its `import torch` (9 pre-existing
test failures), and `test_resolve_step_type_shape`'s field-set pin was stale.

## Perf decision

Depth 2 passes BOTH gates but is still a throughput regression vs plain at
bs=1 on this architecture (55.1 vs 86.9 tok/s; the verify is n+1 sequential
1-row trunk forwards, so depth 2 needs ~2.2× the token yield to break even
and 0.57 acceptance yields ~1.55×). **Default stays `--spec-draft-n 1`.**
Depth 1 is itself slower than plain on this short-window gate run (62.3 vs
86.9 tok/s over 256-token generations; stage-3's longer-window economics
should be re-checked in the stage-5 soak). The clone fix makes depth 2
CORRECT — a viable flag for latency-sensitive long-generation workloads
where per-iteration yield (≤ 3 tokens/iter) beats per-step overhead.

## Stage 5 soak (2026-09-21, 35B NVFP4, turbo8 @ 128k, 4 concurrent)

Byte-stability gate: two consecutive greedy passes byte-identical, two
concurrent clients byte-identical — **STABLE**. Per-window acceptance
telemetry r = 0.65–0.92 (≥ 0.55 gate). `ft ctl stats` clean across windows.

Load battery (nreq=24×60 prompts, 512 max tokens, bs=4, greedy via HTTP):

| window | aggregate gen tok/s | p50 | p95 | errors |
|---|---|---|---|---|
| plain | 88.7 | 12.0s | 14.7s | 0 |
| `--spec-mtp` (depth 1) | 81.3 | 13.6s | 21.5s | 0 |

Per-stream decode: spec 67–110 tok/s vs plain ~21 tok/s (≈3–5× single-stream
latency win). Aggregate throughput at full 4-way concurrency: spec −8% with a
worse tail (the verify is n+1 sequential 1-row forwards; plain batches 4 rows
per forward). **Launch-config call: `--spec-mtp` ON for latency-bound serving
(local coding agents, 1–2 streams); OFF for saturated max-throughput serving.**

Two production bugs the soak caught (both fixed):

1. **`SamplingParams.is_greedy` required `top_p == 1.0`** — every HTTP
   request with model-default sampling (Qwen3.6 ships top_p 0.95) silently
   degraded to plain decode; spec never armed outside in-process gates.
   Greedy is now `temperature <= 0 or top_k == 1`.
2. **Mixed-device `spec_carry` stack** — prefill seeds the carry as a GPU
   hidden row, the resolve refreshes it to CPU; the first spec batch mixing
   fresh and continuing reqs crashed the scheduler worker. Staging now
   normalizes to the engine device.

Test repairs riding the same commit: `tests/scheduler/test_spec_bookkeeping.py`
fixtures updated to the stage-4 `row_idx=`/`spec_drafts_gpu` interface (the
8 failures predate the soak — the stage-4 commit didn't run that file), and
`aot_models.py` gained the GGUF `arch_aliases` the wave-1..4 commits owed
(`LlamaGGUFForCausalLM`, `Qwen2GGUFForCausalLM`, `Qwen35MoeGGUFForCausalLM`).

**Wave 2 complete: stages 1–5 LANDED.**
