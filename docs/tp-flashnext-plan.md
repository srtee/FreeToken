# Flash-Next tensor parallelism — work list

Status: Phase 1+2 landed 2026-10-07 (TP2 `--dummy-weight` acceptance PASSED; real-weights
bench pending a free GPU). Target: serve Qwen3.8-Flash-Next (`qwen4_exp`, NVFP4) with
TP=2 on the desktop pair (2×16 GB Blackwell). All paths relative to the repo root.

## TL;DR

The TP scaffolding is real and launches (one process per GPU, NCCL/pynccl, sharded
linears/embeddings/pools) — `qwen3_moe` proves the pattern end to end. Flash-Next is
locked out by four `qwen4_exp`-specific gates, none of them framework-wide:

1. the weight loader refuses TP>1 (`models/qwen4_exp/weight.py:267`),
2. NVFP4 experts are `tp_ok=False` (`layers/quantization/moe/nvfp4.py:44`),
3. QSA sparse attention has no head/pool sharding (`attention/qsa_sparse.py`,
   `kvcache/qsa_pool.py`),
4. the MTP head is declared TP=1 (`models/qwen4_exp/mtp.py:8`).

The work is: shard the trunk the `qwen3_moe` way, make NVFP4 expert banks split along
the intermediate axis, wire the QSA pool for ranks, and carry the MTP head along.
Phases 1-2 deliver serving; 3 restores speculative decode; 4 is the lossless gate.

## Current state (what already works at TP>1)

- Launch: `server/launch.py` spawns one process per rank, `--gpu` takes one entry per
  rank (UUID prefixes fine); process group gloo+PyNCCL or NCCL (`engine/engine.py:593`).
- Linears: `div_even` output/input sharding + all-reduce (`layers/linear.py`).
- Embedding/lm_head: vocab-parallel embedding, all-gather logits (`layers/`).
- Paged KV: kv-head sharding with `allow_replicate` replication
  (`kvcache/hybrid_swa_pool.py:92`).
- GDN state: key/value-head-local pool dims (`kvcache/linear_state_pool.py`,
  `_linear_local_dims`).
- MoE scaffolding: experts already conceptually split by `intermediate // tp_size`
  (`layers/quantization/moe/base.py`); the reject path is the single `tp_ok` flag.
- Engine memory baselines: cross-rank MIN/MAX (`engine/engine.py:361`) — our pair is
  two identical 16 GB cards, so no skew handling needed.
- Reference family: `models/qwen3_moe` (attention `div_even`, loader
  `iter_merged_tensors` + `MergeRule`, sharded expert iteration).

Flash-Next geometry, census-verified from the real checkpoint config
(`~/models/Qwen3.8-Flash-Next-NVFP4`, via `models/qwen4_exp/config.py`
`parse_config`): hidden 2560; 48 layers = 36 GDN + 12 QSA (+1 MTP draft folded into
the QSA group, `config.py:147-152` → 13 pool layers); QSA 24 qo × 256 on 2 replicated
kv heads; GDN 16 key / 48 value heads × 128; indexer 4 n-heads × 128 + 1 kv head,
ratio 4 / budget 2048; hc_count=4 streams (hc_lowrank 320); 512 NVFP4 experts top-10
with `moe_intermediate=640` (+ 640 shared expert); PLE 320M × 160 B = 47.7 GiB host
side; vocab 248,320.

## Design: sharding table

| Component | Axis | Per rank | Comms | Notes |
|---|---|---|---|---|
| QSA attention qo | heads, `div_even` | q_heads/2 | none | mirror `models/qwen3_moe/attention.py` |
| QSA kv (2 heads) | replicate | full | none | `allow_replicate=True` precedent |
| QSA indexer | replicate | full | none | slab is slot-indexed, not head-indexed; shard only if profiling demands |
| QSAKVCache pool | kv heads / slots | per rank | none | clone the `hybrid_swa_pool` tp math; pending ring is per-request → per-rank |
| GDN projections | qkvz/ba heads | heads/2 | out-proj all-reduce | fla kernels take local head counts — verify, don't assume |
| GDN state pool | already local | — | none | `_linear_local_dims` is TP-aware today |
| Hyper-connections | replicate | full | none | resolved: the fused `input_mix_weight_down_block_inject` is the HC's own lowrank GEMM (`LinearReplicated`, hc.py:99-101), not the MoE down-proj — replication is the module type; zero coupling with expert sharding |
| Shared expert | intermediate (row-parallel) | I/2 | all-reduce | same treatment as the routed experts' down axis |
| Routed experts NVFP4 | intermediate | gate_up `[E, 2·I/2, H]`, down `[E, H, I/2]` | none (router replicated, top-k identical per rank) | **the load-bearing item**: NVFP4 scale groups run along K, and down's K is the sharded axis → scales split cleanly iff `moe_intermediate % (16·tp)`; flip `tp_ok`, split banks per rank in the offload cache (`moe/offload_cache.py` bank schemas) |
| PLE table | v1: replicate | full store per rank | none | disk backend is natural per-rank (RowStore per process); `pinned` costs 47.7 GiB × ranks — document, don't solve. Row-space sharding is a follow-up |
| Token embedding / lm_head | as built | — | all-gather | no work |
| MTP head | mirrors trunk | — | all-gather (logits) | draft layer clones trunk sharding; draft MoE `[E, …]` pair shards by intermediate; its KV row rides the QSA group (already in `full_ids`); `draft-vocab` mask applies after the gathered logits → per-rank identical |
| Weight loader | — | — | — | replace the `weight.py:267` raise with `qwen3_moe`-style sharded iteration; fuse-then-shard order (packed fusions run on full tensors, shard after) |
| `kernel/aot_models.py` + offload schemas | — | — | — | resolved: descriptors feed only the prebuilt kernel-cache name list; runtime specs are named from live shapes (`aot.py:43-87`), so TP-sharded shapes miss by name and JIT-fall back. No runtime consumer reads the descriptors — rank-aware descriptors are an optional later optimization, not a blocker |

## Work list

### Phase 0 — audit (no behavior change)

1. Divisibility census — **done 2026-10-07**: tp∈{2,4} fully green (qo 24→12/6;
   GDN k 16→8/4, v 48→24/12 — no replication needed; indexer n 4→2/1, kv=1 →
   replicate; `moe_intermediate=640 % (16·tp)` → 320/160; marlin N bounds hold:
   2I/tp %64 and H=2560 %64; shared expert 640→320/160; vocab 248,320 even). No pads
   or floors; the only replication is the trivial kv=1 indexer head.
2. HC packed-fusion axis — **resolved**: the fusion is the HC module's own lowrank
   GEMM — `input_mix_weight_down [320, 10240]` concat `block_inject_weight [4, 10240]`
   + 12 zero-pad rows (weight.py:81 `_PAD_TO`, hc.py:81-101) — built as
   `LinearReplicated`. It never touches the MoE down-proj, so `_QWEN4_EXP_PACKED`
   needs no TP-aware merge rule; fused tensors replicate whole.
3. AoT descriptor consumers — **resolved**: no runtime consumer reads the descriptor
   shapes (`aot_models.py:21-27` states the table targets TP=1 by design). Runtime
   kernel specs encode live shapes in their names (`aot.py` `_store_spec` /
   `_index_spec` / `_fast_index_copy_spec`), so a TP-sharded store row (kv 2→1 heads)
   or halved fast_index_copy feature sizes misses the prebuilt cache and JIT-compiles
   once (needs runtime nvcc). Embedding-index shapes are TP-invariant → AOT hit.
   Making the table rank-aware is an optional later win.
4. CUDA-graph capture under TP — **resolved with one new work item**: `get_free_memory`
   is per-rank local (`graph.py:104`); each rank captures its own graphs on its own
   device; the FI capture-scratch handoff (trunk disabled under `--spec-mtp` → draft
   owns it, verify reuses the arm, `graph.py:400-405`, `:598-605`) is per-rank by
   construction. Trunk and verify graphs capture `model.forward()` through the
   lm_head all-gather, so captured logits are full-vocab per rank — TP-safe as built.
   The exception: the draft graph's IN-GRAPH argmax (`graph.py:444`) — new Phase 3
   item below.

### Phase 1 — trunk TP (serving without MTP) — **landed 2026-10-07**

1. Loader — **done**: `_tp_sharder` in `models/qwen4_exp/weight.py` (fused-name dispatch:
   qkv segmented q|gate col-sharded + kv per-head chunk/replicate; GDN `in_proj_qkvz`
   head-grouped row split, `conv1d`/`A_log`/`dt_bias` head-row slices; `o_proj`/`out_proj`
   row-parallel col chunks; embeddings vocab-parallel), applied post-fuse to the fused
   dense stream; `weight.py` raise dropped. Unit-proven by a shape/semantics harness.
2. QSA backend + pool — **no code needed**: `QSAKVCache` extends `MHAKVCache` (kv-sharded
   internally); index slab + pending ring replicate by design; the backend is shape-agnostic
   over local head counts flowing from the module.
3. GDN — **done**: local head counts via `div_even(..., allow_replicate=True)` matching
   `linear_state_pool._linear_local_dims`; input projections now `LinearColLocalMerged`
   (explicit rank-local output sizes — the qkvz packing's k/v head blocks do not chunk
   contiguously); `out_proj` → `LinearRowParallel`. fla kernels are shape-agnostic.
4. HC blocks replicated (already `LinearReplicated`); shared expert row/col-parallel by
   inheritance from `qwen3_5_moe._SharedExpert`.
5. PLE disk backend is rank-agnostic (per-rank fd reads).
6. Acceptance — **PASSED**: shrunken-config TP2 serve (4 layers, 8 experts, 1-indexed
   `ple_layer_ids` — HF validates) on both 5070 Ti + 5060 Ti: both ranks built KV pools +
   qsa backends, NVFP4 triton banks engaged, pynccl all-reduce carried a real greedy
   completion (200 OK). Machine recipe: pynccl JIT needs an unversioned libnccl —
   `ln -s .../nvidia/nccl/lib/libnccl.so.2 .tp2-nccl/libnccl.so` + `LIBRARY_PATH=$PWD/.tp2-nccl`.
   Regression: TP1 qwen4_exp/scheduler suites green except the pre-existing PLE boundary
   snapshot failure (reproduces on baseline `3c28947`; unrelated).

### Phase 2 — NVFP4 experts — **landed 2026-10-07** (acceptance covered above)

1. `nvfp4.py` — **done**: all three kernels (triton/marlin/b12x) `tp_ok=True` behind the
   `moe_intermediate % (16·tp)` gate; every bank layout switched to `cfg.local_intermediate`
   (verified: intermediate-axis shapes halve exactly tp1→tp2, K axes stay full).
2. Bank split — **done**: `_tp_slice_pieces` in `models/nvfp4_banks.py` slices each expert
   piece to the rank's intermediate rows before the per-expert assembly.
3. Acceptance: dummy-weights part done; the full-size real-weights serve (per-rank bank
   bytes = half, KV ≈ 2×) waits for a free GPU (pw.x co-tenants hold ~12.4 GiB/card).

### Phase 3 — MTP under TP

1. Draft layer/attention clone trunk sharding; draft MoE intermediate shard; lift the
   `mtp.py:8` declaration once the trunk holds.
2. Draft CUDA graph: the captured tail argmaxes LOCAL logits (`graph.py:444`); under
   TP the draft lm_head is vocab-sharded, so the captured step must gather first —
   in-graph all-gather before the fp32 cast + argmax (the trunk graph already captures
   the lm_head all-gather, so the pattern is proven capturable), or local argmax +
   all-gather of int ids. Verify graphs are unaffected (argmax runs outside).
3. Draft-vocab mask: no change (post-all-gather).
4. Acceptance: `--spec-mtp --tp 2` engages the arm (per `docs/QSA.md` the arm must
   first engage on QSA at TP1 — that prerequisite is unchanged); acceptance-rate
   telemetry within noise of TP1.

### Phase 4 — verification + docs

1. Numerics gate: TP2 vs TP1 greedy — **allclose + 100% argmax agreement on a fixed
   prompt set**, NOT byte-identical (all-reduce order differs from single-rank math;
   the byte-identical convention stays reserved for same-device comparisons).
2. QSA/GDN snapshot tests re-run at tp=1 (no regression) + new rank-split unit tests
   (pure logic: shard math, bank slicing, merge rules).
3. Bench on the desktop pair: tok/s, acceptance rate, KV capacity, host RAM (PLE
   backend both ways) → `docs/` numbers file, `docs/cli.md` TP section,
   `docs/models.md` Flash-Next row.
4. The TP=1-only guard sentences in `docs/` (gguf adapters keep theirs — out of scope).

## Risks / open questions

- 2 kv heads: TP2 replicates KV (fine); TP4 would replicate 4× qo into 1× kv — fine
  for memory, wasteful for compute; TP>2 is not a goal.
- QSA indexer under TP is v1-replicated; if it becomes the profiled bottleneck,
  head-sharding it is a contained follow-up (its own div_even + slab layout).
- Host RAM: pinned-PLE ×2 = 95.4 GiB — desktop has it, but disk backend is the
  documented TP default.
- The scheduler's runtime-rebuild gate under TP (`scheduler.py:638`) is inherited,
  not fixed here.
- GGUF adapters stay TP=1 (`models/qwen3_moe/gguf.py:32`,
  `models/qwen3_5_moe/gguf.py:55`) — NVFP4 safetensors checkpoints only.

## Reference

- `models/qwen3_moe/` — the in-repo TP template (attention, loader, experts).
- `server/launch.py`, `engine/engine.py:593` — rank launch + process group.
- `layers/linear.py`, `layers/embedding.py` — row/col-parallel + vocab-parallel.
- `kvcache/hybrid_swa_pool.py:92`, `kvcache/linear_state_pool.py` — pool sharding.
- `layers/quantization/moe/base.py` (`tp_ok`), `nvfp4.py`, `moe/offload_cache.py`.
- Strata `--mmap-experts` (2-GPU expert split, 1.3-1.6× per their users) — prior art
  that Flash-Next shards by expert across cards; our design shards by intermediate
  instead (top-k routers want whole experts per rank to keep gathers local).
