# Wave-2 Stage 3 — verify CUDA-graph family: status & findings

**As of 2026-09-17 09:0x UTC. STAGE 3 GATE: ALL PASS (2026-09-17).** Gate 1 bit-equality (200 spec
iterations, eager==graphed argmax/hidden/GDN snapshot, bs 1..4) + Gate 2 losslessness (3 prompts,
763 tokens byte-identical at bs=1, cap-short spec lengths allowed). Log: `/tmp/stage3_gate6.log`.

## State of the tree (uncommitted, on `gguf-serve` @ `7bab4f8`)

| File | Change |
|---|---|
| `python/freetoken/engine/engine.py` | **ES fix v2 (never enable)**: `_ensure_expandable_segments` is **skipped entirely** when `spec_mtp && cuda_graph_max_bs > 0` (engine.py:306-307). The earlier capture-time flip (v1) was proven insufficient (M1/M2 probes still IMA at bs=2/4) — ES must be off from the FIRST allocation (KV pool, FI workspace, all pre-capture tensors) or mixed segment types corrupt replay. Restore hooks in rebuild/shutdown removed (nothing to restore). |
| `python/freetoken/attention/fi.py` | **Wrapper reuse on re-capture**: `prepare_for_capture` no longer asserts `bs not in graph_wrappers`; a second graph family at the same bs (the verify family after the draft family) REUSES the existing per-bs FI wrapper (static scratch buffers, re-planned in place). Required for two families to coexist per bs; audited capture/replay clean. |
| `scripts/mtp_stage3_probe.py` | Seqlen-sweep bit-equality probe. **2026-09-16 harness fix (final form `+bs`)**: graphed-arm `linear_table_idx` must be (a) DISJOINT from the eager arm's slots (else the graphed row inherits eager's advanced GDN state as its initial state — the entire bs=4 "failure"; `+3` overlapped at slot 4 iff bs=4) and (b) IN-BOUNDS of the `4·mr+1`-slot pool (`slots + out_off=100` IMA'd every leg instantly). `slots + (bs if out_off else 0)` satisfies both at every bs. |
| `scripts/mtp_tail_probe.py` | Tail-op capture-vs-eager bisect (all ops exonerated). |
| `scripts/mtp_stage3_replaydiff.py` | New (2026-09-16): checksum-diff of every capture-read buffer across `replay → eager → replay` at fixed bs; includes the zero-FI-workspace causal test. Proved replay determinism + exonerated the FI float workspace. |
| `scripts/mtp_stage3_mini.py` | New mini-gate: fixed bs, 2 seqlens per row option, N steps — **run it next** (baseline run was killed at launch). |
| `scripts/mtp_stage3_gate_worker.py` | GDN band staging fixed `slots+11` → `slots+5` (in-bounds). |

## Established facts (evidence-backed)

1. **bs=1 verify replay is numerically CORRECT.** Seqlen sweep q∈{0,1,8,32,100} at bs=1: all `maxdiff=0`, bit-equal to eager. The earlier "maxdiff 17, mean 1.07" result was an artifact of the old probe's mixed staging, NOT an engine bug.
2. ~~bs=2/4 pass with ES off globally~~ **SUPERSEDED — the "pass" evidence was invalid**: those runs left `PROBE_BS` unset (probe default = 1), so every "bs=2/4 pass" was actually a bs=1 run. Corrected fix-v2 matrix (ES never enabled):
   - bs=1: bit-equal at all seqlens ✓
   - bs=2: no IMA, but logits **diverge** (maxdiff 3.6–4.1, mean 0.27–0.50; argmax still equal)
   - bs=4: diverges worse (maxdiff 5–10.8, **argmax differs at 4/5 seqlens**)
   So fix v2 converted the bs≥2 IMA into **wrong values**. Eager-vs-eager and graph-vs-graph reruns diverge only because the statified GDN layers carry over state between runs without reset (probe artifact); at q=0 (both slots zeroed, no carry) bs=2 eager-vs-graph maxdiff=3.59 while bs=1 is 0.0 — the graphed arm reads/computes wrong data at bs≥2.
3. **Layer bisect at bs=2 DONE** (`scripts/mtp_stage3_layerbisect.py`, log `/tmp/bisect2.log`): divergence starts at **layer 0, a GDN (linear_attention) layer** — layer-0 output differs eager-vs-graph (0.042, deterministic across runs; `fresh_state()` zeroing ALL GDN slots + MoE cache before each run does NOT change it — bit-identical 0.0419922). Divergence then amplifies through the stack (MoE layers 30/33/35/38/39 hit 0.2–1.25, final norm 5.6–6.9, logits 3.4–3.5). Full-attention layers and the draft head (stage-2 gate) are clean — draft is full-attention only, so this isolates the **captured trunk's GDN path at bs≥2**.
   - **Cross-check (XCHK): eager bs=1 row-0 vs eager bs=2 row-0 (same token/pos/slot, fresh state): logits maxdiff=0.031 (bf16 noise), first per-layer divergence at layer 5 = 1.5e-5** — eager arm is fine; the error is in the GRAPH arm.
   - **GDN layer-0 sub-op bisect** (class-level patched forward, snapshots recorded INTO the graph): `conv_in=0, a=0, b=0, mixed (conv decode out)=0, core (fla decode out)=0.188, normed=1.84`. So **staged inputs (conv_in/a/b) are bit-identical; the divergence is born inside `gdn_decode_fla`** (the vendored FLA triton decode kernel) when captured+replayed at bs=2. The gated-RMS norm amplifies it (0.19→1.8).
4. ~~Why does `gdn_decode_fla` diverge under capture at bs≥2?~~ **RESOLVED 2026-09-16 — the kernel read a FREED `cu_seqlens`.** The verify runner's capture loop built `FLAMetadata(cu_seqlens=torch.arange(bs+1), …)` from a loop-local `batch`; when the loop rebound `batch` for the next family size, the arange's storage was freed and reused by eager temps. Captured GDN kernels load `bos`/`eos`/`T` from that memory AT REPLAY → garbage `T` skips/corrupts the recurrent update. Explains every prior observation: the deterministic 0.042 (fixed reuse pattern), fresh-state no-op, bs=1 bit-equality (allocator kept the block by luck), bs≥2 wrong values, and bs=2 IMAs. The kernel is autotune-free (`num_warps=1`, fixed constexprs) — the autotune suspect was dead. The main trunk never had this bug: `build_fla_metadata`'s docstring (linear.py:54-56) documents the contract that under CUDA graph, metadata must be built against persistent buffers via `GraphCaptureBuffer.set_batch`; the verify runner's bespoke capture loop was the one path that violated it. **Fix**: persistent `self.fla_cu_seqlens` in `MTPVerifyGraphRunner` (see tree table).
5. **Post-fix verification (2026-09-16, FINAL MATRIX GREEN):** with the GDN `cu_seqlens` fix + the disjoint-slot harness fix (`+bs`), the seqlen-sweep probe PASSES at **bs=1, 2 and 4** — every q∈{0,1,8,32,100}: `eager-vs-graph maxdiff=0, mean=0, argmax_equal=True`, KV layers 3/39 bit-equal. `graph1-vs-graph2` exactly equals `eager-vs-eager2` on every row — the graphed arm advances GDN state bit-identically to eager. Logs: `/tmp/stage3_matrix_{4,2,1}.log` (all exit=0).
   - The FI float workspace (`fi.float_ws`, aliased into `wrap4`) MOVES when eager runs interleave — exonerated: zeroing it before replays changed nothing; its contents are eager-side scratch, never a capture-read dependency. Same for moe bank caches / recurrent states movement (states advance in place by design).
   - The old "bs=4 failure" chain, fully decomposed: (a) real bug = freed capture-time `cu_seqlens` (fact 4, fixed); (b) bs=4-only signal = harness slot overlap at `+3` (fixed, `+bs`); (c) an intermediate harness iteration (`slots+100`) IMA'd — OOB vs the `4·mr+1` pool (gotcha below).

## Next steps (in order, when resuming)

1. ~~Mini-gate / layer bisect / GDN fix / probe matrix~~ **ALL DONE.** Root cause: freed capture-time `cu_seqlens` (fact 4, fixed in `_capture_graphs`). Probe matrix **GREEN** 2026-09-16 evening — bs=1/2/4 all `maxdiff=0` across the full seqlen sweep (fact 5, logs `/tmp/stage3_matrix_{4,2,1}.log`).
2. ~~Full stage-3 gate~~ **Gate 1 PASSED. Gate 2 root cause: BATCH-SHAPE asymmetry between the
   gate's arms (M-dependent cuBLASLt kernel choice) — NOT the verify graph, NOT the allocator,
   NOT MoE-cache sizing.** Full falsification chain, each step proven by an identical-divergence
   control (same fingerprint fib@98 / ww2@46 / py@230 every time):
   - Verify graph bypassed (`FT_SPEC_VERIFY_EAGER=1`): still diverges → verify graph exonerated.
   - Allocator matched (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in spec): still → ES exonerated.
   - `FT_SPEC_DRAFT_EAGER=1` (zero CUDA graphs at all): still → ALL graph machinery exonerated.
   - Stage-1-era worktree (`f202681`, pre-stage-2): still → NOT a stage-2/3 regression at all.
   - `moe_cache_size=4096` pinned in both arms: still → MoE-cache-auto theory falsified (an
     earlier "confirmed mechanism" note here was wrong; superseded).
   - **Actual mechanism**: plain built `mr=1/cgmbs=1` (prompts queue → M=1 GEMMs) vs spec
     `mr=4/cgmbs=4` (3 prompts batched → M=3). cuBLASLt bf16 kernel choice is M-dependent — the
     documented known-fake — and near-tie greedy argmaxes flip at arbitrary positions. Fully
     deterministic per config (fingerprint stable across runs and trees).
   - **Fix (gate, not product)**: both arms generate ONE PROMPT AT A TIME (bs=1, M=1 everywhere);
     the spec build keeps `mr/cgmbs=MAX_BS` only for Gate 1's bs 1..4 bit-equality families.
     E2E byte-identity is only well-defined at matched batch shape. **FINAL: gate rerun 2026-09-17
     ALL PASS** (`/tmp/stage3_gate6.log`) — one wrinkle: at the `max_tokens` cap the spec arm emits
     2 tokens/accepted-iteration and ends one token SHORT with identical content (the stage-2 gate's
     documented benign boundary); the gate contract now allows spec `len in {len(p), len(p)-1}`.

## Harness staging gotchas learned (for whoever writes the next probe)
- Hybrid GDN pool = `4·max_running_req + 1` slots (`_linear_pool_num_slots`). At `max_running_req=2` that's **9 slots (0..8)** — any band displacement must stay in bounds or you get IMAs that look like engine bugs (this bit me 3 times: `+11` OOB, `+7` OOB at mr=2, `+3` fine).
- The engine's dummy page-table row is filled with a constant pool row (`num_tokens`) at init — **every decode KV write goes to the same shared page** unless you restage the table. Displacing `out_loc` by +100 is legal for write-row *selection* but both arms' reads resolve through the same table; comparing "displaced vs undisplaced" rows directly compares two different physical rows — must read each arm's OWN rows (probe's `dump_states(tag, out_loc_off)` handles this).
- `PROBE_BS` env picks probe bs; `dump_states` needs `out_loc_off` arg (0 for eager, 100 for graphed).
- Full-pool `.clone()` dumps OOM at 1.23 GiB free — dump to CPU (`.cpu()`), don't clone on GPU.
- `OPList` has no `named_children()` — iterate `eng.model.model.layers.op_list`.
- 300s bash-job deadline kills model loading — pass explicit `timeout` ≥ 1800s for anything that boots the engine.
- `PROBE_BS` defaults to **1** in `mtp_stage3_probe.py` — an unset env silently turns a bs=2 test into bs=1. This invalidated one "passing" run and cost hours; always set it explicitly and check the log echoes the intended bs.
- **Cross-process A/B gates must pin every auto-sized resource** (`moe_cache_auto`, cache auto-sizers generally): both arms derive sizes from free VRAM, and any VRAM asymmetry (extra graph families, allocator mode) silently changes numerics via hybrid-MoE CPU-overflow participation. Symptom: byte-divergence at sparse, arbitrary first-diff offsets that survives every graph/allocator control. Check both logs' `resolved moe_cache_size=` lines FIRST.
- **Gate-2 byte-divergence (98/46/230 fingerprint) root cause: BATCH-SHAPE asymmetry between the
  gate's arms** — plain built `mr=1/cgmbs=1` (prompts queue, M=1 GEMMs), spec built `mr=4/cgmbs=4`
  (3 prompts batched, M=3 GEMMs). cuBLASLt bf16 kernel choice is M-dependent (the documented
  known-fake, tcq-coordination 2026-09-14): different kernels differ by 1 ULP on some rows and
  greedy near-tie argmaxes flip at arbitrary positions. Every earlier falsification (allocator,
  MoE-cache pin, fully-eager) kept this asymmetry and reproduced the same fingerprint — and the
  stage-1-era tree reproduces it too, so it was NEVER a stage-2/3 regression. Fix: both arms
  bs=1-per-request, one prompt at a time (worker keeps mr/cgmbs=4 in the spec build only for
  Gate 1's bs 1..4 families). Gate 2's E2E contract is only well-defined at matched M.
  Logs: /tmp/g_spec_full_eager*.log, /tmp/s1_{plain,spec}.log (worktree repro), /tmp/stage3_gate5.log (fixed gate).
- `QLIST` env (comma-separated) selects the sweep's q values; `MINI_Q/MINI_STEPS/MINI_BAND/MINI_RANDBS/MINI_MIX` configure the mini-gate.


## Environment notes for the resumed session

- RAM: **123 GiB installed, swap empty** — expert banks now load at disk speed; each probe run is ~4 min instead of ~13.
- Checkpoint: `~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5` (the knoopx GGUF snapshot is NOT the right one for these scripts).
- Log artifacts: `/tmp/m1.log`, `/tmp/m2.log` (fix-v2 matrix, in flight), `/tmp/stage3_gate.log` (pre-fix run), `/tmp/stage3_gate2.log` (post-ES-flip-v1, still IMA), `/tmp/stage3_probe_sweep*.log` (passing seqlen sweeps), `/tmp/tail_probe.log` (tail-op exoneration).
- `/tmp/stage3_gate3.log` (gate, pre-M-fix: fib[98] fingerprint), `/tmp/stage3_gate5.log` (pinned-cache
  still diverging → killed moe_cache theory), `/tmp/stage3_gate6.log` (**FINAL ALL PASS**), replaydiff
  chain `/tmp/replaydiff{3,4}.log`, probe matrices `/tmp/stage3_matrix_{4,2,1}.log`.