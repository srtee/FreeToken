# MTP Wave 2 — Speculative decode in production (draft + verify, graphs)

Status: **stages 1–4 LANDED** (stage-3 gate ALL PASS 2026-09-17, commits
`22a83b8` + `f08f033`; stage-4 byte-identity ALL PASS but acceptance gate
NOT met — depth 2 config-gated, default 1; see `mtp-stage4-status.md`).
Supersedes the Wave-2 section of `mtp-plan.md` (kept for history). Stage 5
(soak + final docs) is the remaining open stage, plus the depth-2 draft-KV
investigation. Prerequisite state at planning time: waves 0 + 1 landed
(`8da83f9`, `8203e81`) — MTP weights load, `MTPHead.draft_step` runs on the
trunk attention (layer-40 KV), `--spec-mtp` plumbs through, `verify_chain`
unit-tested.
state at planning time: waves 0 + 1 landed (`8da83f9`, `8203e81`) — MTP
weights load, `MTPHead.draft_step` runs on the trunk attention (layer-40
KV), `--spec-mtp` plumbs through, `verify_chain` unit-tested.

## The corrected loop (read first — wave-1's parked design was wrong)

The parked wave-1 design ran fwd1 (1-row) THEN a 2-row verify whose row A
re-processed fwd1's output — three trunk rows per iteration for two
emitted tokens: structurally one wasted forward. The correct loop (the
standard Leviathan/DeepSeek formulation — the verify forward IS the next
-token producer):

Per decode iteration, with the certain input token `c@q` and the trunk
hidden carry `H@q-1`:

1. **Draft** (MTP head, 1 layer, 1 token): `draft(H@q-1, c@q) → d` —
   the speculative guess for position q+1. Writes the layer-40 KV row
   for position q.
2. **Verify** (trunk, ONE 2-row forward): rows `[c@q, d@q+1]`.
   - row A (processing c@q) → argmax `a` = the trunk's true prediction
     for q+1 — the *verification* of d.
   - row B (processing d@q+1) → argmax `b` = the prediction for q+2
     (the *bonus* token, only meaningful if d is accepted).
3. **Resolve** (per req):
   - accept (`a == d`): emit `[d, b]`; next input = `b@q+2`;
     carry = row B's hidden.
   - reject: emit `[a]`; next input = `a@q+1`; carry = row A's hidden;
     roll back row B (free its KV page slot, rewind `device_len` by 1,
     restore the GDN state to the post-row-A point).
4. Layer-40 KV: the draft wrote position q's row. Position q+1's row is
   written by the **catch-up draft step** on the next iteration's input
   (the draft always processes the *input* token of its iteration, so
   layer-40 rows exist for every position the draft layer attends over)
   — EXCEPT the accepted-draft position, whose row nobody writes. This
   **gap question** (does the draft layer need a row at accepted-draft
   positions, and if so who writes it — an extra catch-up MTP step on
   accept, or an attention-side skip) is settled in Stage 0 against
   buun's `common_speculative_mtp_carry_lifecycle`. Not hand-waved.

Token accounting per iteration: 1 draft + 1 two-row trunk forward;
emits `1 + accepted`. Eager ceiling ≈ `(1+r) / (c2 + cd)` where c2 =
cost of a 2-row forward in 1-row-decode units, cd = draft cost
(≈0.2–0.3). At r=0.7, c2=1.2, cd=0.25 → **≈1.17× eager**; graphed
(launch overhead gone, c2→~1.05) → **≈1.3×**. Depth 2 (Stage 4, config
-gated) → 3-row forward, `(2+r2)/(c3+2cd)` ≈ **1.6×+** — the plan's
original 1.6× target is a depth-2 number; wave 2's depth-1 gate is set
honestly at ≥1.15× graphed, and the Stage-0 bench anchors the real
ceiling.

Losslessness is by construction (greedy argmax comparison at the same
position; a rejected draft is replaced by the trunk's own argmax) and
is re-asserted by an E2E byte-identical gate at every stage.

## Code contracts (scoped now; details per stage)

- `engine/spec_mtp.py` owns the loop logic:
  - `SpecStep.run(drafter, batch, carry) -> SpecResult` — draft + verify
    orchestration, on the engine stream, inside `_forward`.
  - `SpecResult`: `emitted_tokens` [bs, ≤2] + per-req lens, `next_input`
    [bs], `carry'` [bs, H], `accept_mask` [bs], `rollback` [bs].
  - `verify_chain` stays (extended: row-B argmax selection per accept).
- `engine.forward_batch`: when spec armed and `batch.is_decode` — build
  the 2-row verify batch (2 fresh page slots/req, positions [q, q+1]),
  run trunk forward once, resolve, roll back rejects, return a
  ForwardOutput whose `next_tokens_gpu` is the next input AND whose
  emitted tokens ride alongside for the drain.
- `scheduler._forward` drain: per-req multi-token emission with
  per-token EOS/stop/length checks (append_host per token).
- `cache_manager`: `snapshot_gdn(batch)` / `restore_gdn(snapshot)` for
  the reject path (GDN recurrent state is in-place — a rejected row B
  pollutes it; snapshot after row A is impossible in one 2-row varlen
  pass, so the mechanism is: snapshot BEFORE the verify (state@q) and on
  reject restore + replay... — Stage 1 pins the cheapest correct
  mechanism from: (a) snapshot/restore + one 1-row GDN-only replay,
  (b) two sequential 1-row GDN steps with a mid-snapshot,
  (c) accept-path-only state refresh. Decided by measurement, not taste.)
- `engine/graph.py`: `draft_graph` family (bs, 1 row) + `verify_graph`
  family (bs, 2 rows) — Stage 2/3. `GraphCaptureBuffer` generalized to
  `rows_per_req` (default 1; existing callers unchanged).

## Stages (each: implement → gate → my review → next dispatch)

### Stage 0 — Semantics freeze + bench (no engine changes)

0.1 Read buun (`/home/sherntee/20llms/buun-llama-cpp/common/speculative.cpp`,
`common_speculative_mtp_carry_lifecycle`, `src/models/qwen35moe.cpp::graph_mtp`)
and write `docs/mtp-spec-semantics.md`: exact answers, each checkable:
  - layer-40 KV row ownership per position under accept/reject (the gap
    question);
  - carry source per resolve case;
  - bonus-token emission;
  - what happens on the first decode after prefill.
0.2 Pure-logic unit test `tests/engine/test_mtp_loop.py`: the iteration
state machine on synthetic logits — given an accept/reject sequence,
assert emitted sequence, next inputs, carry selection, and slot
bookkeeping match hand-computed values. No GPU. This catches loop bugs
before any engine code exists.
0.3 `scripts/mtp_bench_rows.py`: on the 35B (GPU free), measure a 1-row
decode step vs a 2-row extend step vs 2×1-row, greedy, 8K ctx — c2 and
cd. Output: the numbers + honest eager/graphed ceiling estimate into
`docs/mtp-spec-semantics.md`.

**Gate**: semantics doc answers all four questions with buun line
references; logic tests green; bench recorded. I review before Stage 1.

### Stage 1 — Eager spec loop (the surgery, correct loop)

Implement the contracts above, eager (graphs stay off for spec batches).
The verify forward goes through the existing extend machinery
(`extend_len=2` batches — triton attention and GDN varlen already
handle multi-row extends on the eager path). Rollback: the mechanism
Stage 1.0 picked. Keep invariant asserts in the hot path
(`device_len == cached_len + committed rows`, slot ownership) — they
fire loudly on the first bookkeeping bug.

**Gate** (all must pass on the 35B):
- `--spec-mtp` greedy 512-token generations on the 3 ground-truth
  prompts are **byte-identical** to non-spec greedy (losslessness).
- Multi-request A/B: 2 concurrent reqs, spec on vs off — identical
  outputs per req (the soak lesson: single-request tests cannot catch
  reuse-path bugs).
- Acceptance-rate telemetry in the decode log line
  (`#drafted: N, #accepted: k (rate r)`), r ≥ 0.55 on prose.
- Full existing suite green (116 MTP/engine tests).
- Abort mid-iteration test: abort between draft and verify; no dangling
  KV slots or GDN state (follow `test_abort_inflight_prefill.py`).

### Stage 2 — Draft graph

Capture the draft step per-bs (static shapes: token [bs], carry [bs, H],
out_loc, positions → draft token, carry'). Reuse GraphRunner patterns;
own buffer family. `--spec-mtp` keeps verify eager this stage.

**Gate**: graphed draft tokens == eager draft tokens on 200 consecutive
steps (bit-identical argmaxes); losslessness gate re-run; serve smoke.

### Stage 3 — Verify graph (2-row family)

Generalize `GraphCaptureBuffer` to `rows_per_req`; capture the 2-row
verify per-bs (FLA cu_seqlens `[0,2,4,...]`, triton extend metadata at
fixed shapes). Replay path writes positions/out_loc from the buffer as
today. Rollback stays scheduler-side (outside captured regions — assert
this: no page-free or state-restore op inside a captured graph).

**Gate**: graphed verify argmaxes == eager verify argmaxes bit-identical
on fixed shapes; losslessness gate re-run; multi-request A/B re-run;
perf: ≥1.15× decode t/s vs non-spec greedy at bs=1 (bench-anchored);
graph VRAM within budget (else drop cuda_graph_max_bs before KV).

### Stage 4 — Depth-2 parameterization (config-gated, after depth-1 green)

`--spec-draft-n 2`: rows=3 capture, emission ≤3, per-position
acceptance telemetry. Only depth-1 is captured by default.

**Gate**: byte-identical at depth 2; perf table depth 1 vs 2; the 1.6×
depth-2 perf decision from measured numbers.

### Stage 5 — Hardening + docs

30-min soak (spec on, 20 turns × 3 prompts, byte-stability per turn);
`ft ctl stats` clean; docs: `cli.md` (`--spec-mtp`), `models.md` (MTP
section), this plan's status table; final numbers into
`docs/mtp-baseline-numbers.md`.

## Bug-catching doctrine (applies to every stage)

1. Pure-logic tests precede engine code (Stage 0.2 pattern: encode the
   invariant, watch it fail on a wrong loop).
2. Eager is the reference implementation; every graph stage's gate is
   bit-equality against it.
3. Multi-request tests are mandatory at every stage (the turbo
   materializer page-id bug was invisible to single-request batteries).
4. Losslessness re-asserted per stage — a spec implementation that
   changes greedy output is broken no matter how fast.
5. Invariant asserts stay in the hot path (debug-cheap, loud).
6. Acceptance-rate telemetry is a correctness signal too: the layer-40
   KV gap class of bugs is lossless-but-quality-destroying; the r ≥ 0.55
   gate catches what byte-identity cannot.
