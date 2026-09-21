# Wave-2 Stage 4 — depth-2 draft chain: status & findings

**As of 2026-09-21. STAGE 4 GATE: byte-identity ALL PASS; acceptance gate NOT
met — depth 2 stays config-gated, default remains 1.** Losslessness (the hard
gate): depth-2 output is byte-identical to plain AND to depth-1 on all three
gate prompts. The acceptance-rate signal (plan §Verification: "lossless-but-
quality-destroying" class) collapsed at depth 2 — open investigation below.

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
**Default stays `--spec-draft-n 1`.** Depth 1 is itself slower than plain on
this short-window gate run (62.8 vs 83.5 tok/s over 256-token generations;
3 trunk forwards per iteration need >2× yield; stage-3's longer-window numbers
should be re-checked in the stage-5 soak).

## Open item: depth-2 acceptance collapse (blocks any depth-2 default)

`accepted_at=[34, 2]` over ~750 iterations: position-0 acceptance ~4.5% at
depth 2 vs ~43% at depth 1 — the SAME first-draft quantity (argmax of
`draft(H@q+1, c@q)` verified by the trunk row predicting q+2), so the
distributions should match. Output correctness is unaffected (all verify rows
are trunk forwards; rejects roll back fully — hence the byte-identity PASS),
but draft quality dies. Leading hypothesis: the MTP draft-layer KV rows of
the chained drafts (positions q+1, q+2 at layer_id=num_layers) collide or go
stale across rejects — after the first ~40 iterations (where the 34 accepts
concentrate) every subsequent draft attends a polluted layer-40 history. This
is exactly the plan's "layer-40 KV gap class": lossless but quality-
destroying; the r ≥ 0.55 acceptance gate fired as designed.

Next probe: `FT_SPEC_TRACE=1` iteration dump comparing the staged draft row
positions/page slots per chain step against the mapped table; then a
deterministic 2-iteration replaydiff (the stage-3 `mtp_stage3_replaydiff.py`
pattern) with one forced reject between iterations.
