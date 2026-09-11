# FreeToken TurboQuant/TCQ KV Cache — Implementation Plan

Port buun-llama-cpp's TurboQuant KV codecs (FWHT rotation + Lloyd-Max/Trellis-Coded
Quantization, 8-bit down to 2.25 bpv) into FreeToken, staged so a local coding agent
can execute each wave independently with its own verification gate.

## Why this shape

FreeToken stores KV as raw `torch.empty((2, L, pages, page, heads, head_dim), dtype=fp16/bf16)`.
The port replaces *storage + quantize + dequantize*; attention backends, pagers, radix
trees, and the scheduler stay untouched in the early waves. The reference
implementation lives in `/home/sherntee/20llms/buun-llama-cpp` (Apache/MIT — verify
license headers before copying; TurboQuant files carry no third-party claims, codebooks
are MIT-licensed per `codebooks/README.md`).

Key FreeToken integration points (all verified in-tree):

| Point | File | Role |
|---|---|---|
| Pool factory | `python/freetoken/kvcache/__init__.py` → `MHAKVCache(...)` | Where a quantized pool subclass slots in |
| Pool base contract | `python/freetoken/kvcache/base.py` | `store_kv`, `k_cache`, `v_cache`, `unit_bytes`, `kv_cost` |
| KV write | `python/freetoken/kvcache/mha_pool.py::store_kv` → `freetoken.kernel.store_cache` | The quantize hook replaces this scatter |
| KV read | `python/freetoken/attention/{fa,fi,triton}.py::forward` → `kvcache.k_cache(layer_id)` | The dequant hook — either a fused FA kernel or a decode-into-fp16 materializer |
| Backend registry | `python/freetoken/attention/__init__.py::SUPPORTED_ATTENTION_BACKENDS` | Register a turbo-capable backend variant |
| JIT kernels | `python/freetoken/kernel/utils.py::load_jit` (tvm-ffi `load_inline`, nvcc JIT) | Turbo CUDA kernels land here, same pattern as `kernel/csrc/jit/store.cu` |
| Cost model | `python/freetoken/kvcache/base.py::spec_kv_bytes_per_token`, `kv_cost`, `unit_bytes` | Must report the compressed bpv or cache sizing lies |
| Cache rebuild | `ft ctl cache --kv N` → `rebuild(num_pages)` | Must stay working (resize without restart) |

Reference kernels to port (buun-llama-cpp paths):

| File | Contents |
|---|---|
| `ggml/src/ggml-cuda/turbo-wht.cu` | FWHT butterfly, 128-group, constant sign tables (68 lines) |
| `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` | set_rows kernels: turbo3/turbo2/turbo4/turbo8 + TCQ Viterbi encoders (`k_set_rows_turbo3_tcq`, `k_set_rows_turbo2_tcq`), codebook constants, per-channel InnerQ equalization |
| `ggml/src/ggml-cuda/turbo-tcq-alpha.cuh` | context-adaptive `alpha_V` dequant scale — single source of truth for all consumers |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | `flash_attn_ext_turbo{4,8,...}_load_tile` — GMEM→half2 shmem tile loaders for fused FA |
| `ggml/src/ggml-cuda/fattn-mma-turbo.cuh` | MMA turbo case launcher (nstages=0, shmem layout) |
| `ggml/src/ggml-common.h` | Block structs: `block_turbo4_0` (66 B), `block_turbo8_0` (130 B), `block_turbo3_tcq` (52 B = 2-byte norm + 49-byte bitstream + 1 pad), `block_turbo2_tcq` |
| `codebooks/{3bit,2bit}/*.bin` | Trained TCQ codebooks (raw f32 arrays: 3-bit = 512 floats, 2-bit = 256 floats) |

## Codec primer (what the agent is implementing)

Per 128-element rotation group (one head_dim):

1. **Norm**: compute group L2 norm; store fp16.
2. **FWHT rotation** with fixed ±1 sign vectors (seed 42, `d_turbo_wht_s1/s2`): makes
   post-rotation values i.i.d.-ish Gaussian; normalize by 1/sqrt(128).
3. **Quantize** each of 128 values against a codebook:
   - `turbo8`: uniform 256-level grid + per-block absmax → 130 B/128 = 8.125 bpv.
   - `turbo4`: 16 Lloyd-Max centroids (N(0, 1/√128)) → 66 B/128 = 4.125 bpv.
   - `turbo3_tcq`: 512-state right-shift trellis, 3-bit outputs, 390-bit bitstream →
     52 B/128 = 3.25 bpv. Encode = Viterbi (per-thread serial over 128 steps, fully
     parallel across groups). Decode = O(1): `state = read_9_bits(qs, t*3)`,
     `recon = codebook[state] * norm`.
   - `turbo2_tcq`: 256-state, 2-bit outputs → 2.25 bpv.
4. **TCQ alpha_V**: V side carries a context-adaptive dequant scale
   (`log(1 + n/α₀)/log(1 + 1/α₀)` form) so deep-context V blocks get a mild boost;
   K does not (softmax-neutral). Read `turbo-tcq-alpha.cuh` for the exact formula and
   the three consumer sites that must stay in sync.

Block layout (from `ggml-common.h`, port verbatim into the CUDA kernel header):
```c
// turbo4: 2-byte fp16 norm + 64-byte nibble-packed 4-bit indices
// turbo3_tcq: 2-byte fp16 norm + 49-byte bitstream + 1-byte pad
// turbo2_tcq: 2-byte fp16 norm + 33-byte bitstream + ... (see source)
```

---

# Wave 0 — Groundwork: correctness oracle and eval harness

**Goal**: establish ground truth so every later wave has a numeric acceptance gate.
No production code changed.

## Tasks

0.1 **KLD oracle script** — `scripts/tcq_oracle.py` (throwaway-friendly):
- Load the nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot config; run 3 wikitext-2-style
  prompts of 2K/8K/16K tokens through the already-working FreeToken serve stack
  (fi backend) in a Python harness; capture per-layer K/V tensors at `store_kv`.
- Store reference logits + final outputs. This is the "f16 KV" reference.
- Implement pure-PyTorch FWHT (matrix multiply, 128×128 Hadamard) + Lloyd-Max
  quantizer + a scalar Viterbi for the 512-state trellis. Validate against
  `codebooks/*.bin` reconstruction MSE on synthetic Gaussian data first.

0.2 **KLD metric tool** — for a captured (K,V) dump and a codec config, compute
median KL-divergence of attention logits vs the f16 reference, per layer, plus PPL
delta. Mirrors buun's `README.md` codec table so numbers are comparable.

0.3 **Acceptance thresholds** (write into `scripts/tcq_oracle.py --assert`):
- turbo4: median KLD ≤ 0.001, PPL delta ≤ +0.05
- turbo3_tcq: median KLD ≤ 0.002, PPL delta ≤ +0.10
- turbo2_tcq: median KLD ≤ 0.007, PPL delta ≤ +0.35
- (from buun's measured table on Qwen3.6-27B; expect similar order on 35B-A3B)

## Verification
- `python scripts/tcq_oracle.py --codec f16` reproduces itself (KLD=0).
- Pure-torch turbo3_tcq on dumped K/V hits KLD ≤ 0.003 before any CUDA work starts.
- Deliverable: `docs/tcq-baseline-numbers.md` with the per-layer KLD table.

---

# Wave 1 — Turbo8 + Turbo4 as storage codecs (simplest path, no fused FA)

**Goal**: first real compression in production, minimal kernel surface. Decode
attention reads happen through a **materializer**: quantized slabs are the
authoritative storage; a decode kernel expands pages to a scratch fp16 buffer that
the existing FA/FI backend reads. Encode happens in `store_kv`.

## Design

1.1 **New pool class** `TurboKVCache(MHAKVCache)` in
`python/freetoken/kvcache/turbo_pool.py`:
- Storage `torch.uint8` slab of `ceil(head_dim * bpv / 8) + 2` bytes per value, per
  (2, L, pages*page, heads) — i.e. same 6-D logical layout, byte-typed last dim
  replaced by packed blocks. Keep `page_size=1` constraint initially (matches
  FreeToken default; page-quantized boundaries get hairy otherwise).
- `store_kv(k, v, out_loc, layer_id)`: launch quantize kernel
  (f32/bf16 → packed blocks) writing into the slab at `out_loc` rows.
- `k_cache(layer_id)` / `v_cache(layer_id)`: **changed contract** — returns a lazy
  handle, not a dense tensor. This is the crux; see 1.3.

1.2 **Quantize/dequant kernels** — `python/freetoken/kernel/csrc/jit/turbo_kv.cu`
(loaded via `load_jit`, mirrors `store.cu`):
- `turbo_quantize(k_f32, out_u8, out_loc, codebook_mode)` — one block per
  (row, head): 32-warp-group, per-row L2 norm, FWHT butterfly in shmem (port
  `k_turbo_wht` verbatim — 68 lines), scalar quantize against Lloyd-Max centroids
  (`d_turbo_centroids_4bit` constants), pack.
- `turbo_dequantize(in_u8, out_f16, locs, norm_mode)` — inverse; O(1) per element
  for turbo4/turbo8 (lookup + scale).
- Do **not** port InnerQ per-channel equalization yet (Wave 3).

1.3 **Attention read path** — modify `fa.py`/`fi.py` forward to branch:
```python
if pool.is_turbo:
    k_dense, v_dense = pool.materialize(layer_id, metadata.page_table, metadata.cache_seqlens)
else:
    k_dense, v_dense = self.kvcache.k_cache(layer_id), ...
```
`materialize` gathers the needed pages, dequants into a persistent scratch buffer
(allocated at `max_seqlen_k × heads × head_dim × f16`, sized from the resolved
context), and returns views. Correct but costs a full gather-dequant per layer per
step — decode tok/s will regress ~15–30%. That is acceptable for Wave 1; the point
is plumbing + correctness. Fused kernels land in Wave 2.

1.4 **Config + plumbing**:
- `ft serve --kv-codec {f16,bf16,turbo8,turbo4}` (default f16; auto→f16).
- Thread through: `cli.py` → ServerArgs → engine pool factory
  (`kvcache/__init__.py`): when codec is turbo and attn_type == FULL, instantiate
  `TurboKVCache`; refuse (typed error) for MLA/DSA/DSV4/QSA/LINEAR pools in this wave.
- `unit_bytes()` / `kv_cost()` return compressed bpv math: `(head_dim*bpv + 16)/8`
  bytes per value. This automatically fixes `--num-tokens` sizing and the
  `/v1/cache/status` geometry report.
- CUDA graph capture: `store_kv` kernel must be capture-safe (no sync, no dynamic
  allocs — same constraints `store_cache` already meets). Materializer runs inside
  capture too (fixed scratch, indexed by metadata tensors — check `fa.py` capture
  path carefully; `FACaptureData` pins buffer addresses).

1.5 **Radix cache interplay**: prefix-cache hits return `MatchResult` indices into
pages; pages are codec-uniform (whole pool is one codec), so matching is unchanged.
Assert `insert_prefix` never mixes codecs by refusing rebuild with a different
codec without a pool reset (log a clear error).

## Verification
- Round-trip unit test: random tensors through quantize→dequant; assert MSE within
  codebook bound; assert exact byte layout matches `block_turbo4_0`.
- Wave 0 oracle on the real model with `--kv-codec turbo4`: KLD ≤ 0.001 gate.
- E2E: `ft serve --kv-codec turbo4` on Qwen3.6; run the omp agent loop from the
  earlier session (bash tool call) — output coherent.
- Perf: record decode t/s before/after in `docs/tcq-baseline-numbers.md`.
  Expected regression; gate is correctness + VRAM math (49K tokens of Qwen3.6 KV
  drops from ~0.9 GiB to ~0.25 GiB at turbo4).
- VRAM accounting test: `ft ctl stats` KV numbers match `unit_bytes()` × tokens.

---

# Wave 2 — TCQ codecs (turbo3_tcq, turbo2_tcq) with fused decode FA

**Goal**: the 3.25/2.25 bpv sweet spot, and recover the Wave 1 decode regression
with a fused GMEM→shmem dequant in the FA backend.

2.1 **Viterbi encoder kernel** — port `k_set_rows_turbo3_tcq` /
`k_set_rows_turbo2_tcq` into `turbo_kv.cu`:
- One thread per 128-group; serial 128-step trellis over a 512-entry codebook held
  in `__constant__` (512 floats) — port buun's constants verbatim, plus runtime
  override via `TURBO_TCQ_CB=<path>` (read file, 512 f32, `cudaMemcpyToSymbol`).
- Right-shift bitstream writer: 390 bits per block (6-bit zero prefix + 128×3-bit).
- Codebook training scripts stay in buun (`scripts/tcq_train_*.py`); ship the
  compiled-in defaults from `codebooks/{3bit,2bit}/` bests
  (`cb_50iter_finetuned.bin`, `tcq_2bit_cuda_200iter.bin`).

2.2 **O(1) decoder** — port bitstream reader + `codebook[state] * norm * alpha`:
- `turbo-tcq-alpha.cuh`: port the single-source-of-truth alpha_V (context-adaptive
  scale). In FreeToken the two consumers are (a) the materializer (Wave 1 path,
  still supported for FI/triton backends) and (b) the fused FA loader. Assert
  equality in a unit test — buun's comment warns prefill/decode alpha desync is a
  silent correctness bug.

2.3 **Fused FA tile loaders** — FreeToken's `fa` backend calls
`sgl_kernel.flash_attn.flash_attn_with_kvcache` (closed source; cannot add tile
loaders). Two options, in order of preference:
- **(a) Triton fused decode** — extend
  `python/freetoken/kernel/triton/attention.py::decode_paged_attention` with a
  turbo3_tcq K/V loader variant: read packed bytes, dequant in registers/shmem
  before the QK^T dot. Triton can express the bit-ops (shifts/masks) and the
  codebook gather (`tl.load` from a 512-entry device tensor). This keeps
  page-table handling identical. The fa (sgl_kernel) path keeps using the Wave 1
  materializer; make triton the recommended backend for turbo codecs
  (`--attention-backend triton --kv-codec turbo3_tcq`).
- **(b) CUDA fused kernel** — port `fattn-mma-f16.cuh` tile loaders +
  `fattn-mma-turbo.cuh` launcher as a new FreeToken backend `fa_turbo` in
  `SUPPORTED_ATTENTION_BACKENDS` (FULL type only). Bigger job (MMA kernel is
  template-heavy); only if (a) leaves >10% decode on the table vs buun's tg
  numbers. Port `fattn-mma-turbo.cuh` wholesale, `nstages=0` shmem layout and all.

2.4 **Graph capture with TCQ**: the alpha_V context-length term changes with
seqlen — verify it reads length from metadata tensors (capturable), not from
host-side python ints baked at capture time. Buun's fused path handles this;
mirror whatever they do (`tcq_compute_alpha_v` reads a runtime count).

## Verification
- Bit-exact trellis test: pure-torch Viterbi (Wave 0) vs CUDA encoder on 10K random
  groups → identical bitstreams. This is the strongest possible check.
- Decode O(1) test: for random bitstreams, CUDA decoder output == torch reference
  decoder bit-exactly (same codebook, same alpha).
- KLD gate from Wave 0: turbo3_tcq ≤ 0.002, turbo2_tcq ≤ 0.007.
- E2E omp agent loop on `--kv-codec turbo3_tcq --attention-backend triton`.
- Perf vs Wave 1: fused decode should beat the materializer path ≥ 25% at 8K+.
- Alpha-sync test: prefill-then-decode and decode-only on the same tokens produce
  identical stored/decoded values (guards the desync bug class).

---

# Wave 3 — Production hardening

3.1 **InnerQ per-channel equalization** (port from `turbo-quant-cuda.cuh`):
calibrate per-channel K scales during first N prefills (`d_innerq_*` accumulators),
apply before L2-norm/FWHT, invert on Q in the FA kernel. Buun's calibration data
shows this matters for anisotropic K. Gate behind `--kv-codec-tune innerq`.

3.2 **Multi-GPU (TP)**: KV slabs are per-rank after `div_even(num_kv_heads, tp)` —
quantization is per-head-group so no cross-rank state. Test with TP=2 if a second
GPU is available; otherwise assert single-rank correctness and document.

3.3 **`ft ctl cache` integration**: live KV resize
(`cache --kv N`) must dequantize→re-quantize or copy packed bytes on rebuild;
packed bytes copy verbatim (codec is pool-wide), so rebuild is a straight byte
gather. Add codec fields to `/v1/cache/status` geometry output.

3.4 **Checkpoint/FTW format**: FTW files carry KV? (No — weights only. Confirm and
document that KV codec is runtime-only.)

3.5 **Docs**: extend `docs/models.md` MoE/KV notes + `docs/cli.md` with the new
flags and the codec table (bpv / KLD / measured t/s from this machine).

## Verification
- Full pytest suite (`tests/kernels`, `tests/kvcache`, `tests/engine`) green.
- 30-minute soak: serve Qwen3.6 at turbo3_tcq, drive the omp agent through
  multi-turn tool use, watch for drift/loops (buun's "quants will misbehave"
  caveat); compare outputs to f16 server on the same prompts.
- Final table in `docs/tcq-baseline-numbers.md`: f16 vs turbo8/4/3tcq/2tcq —
  VRAM/token, KLD, PPL delta, pp t/s, tg t/s at 2K/8K/16K/32K.

---

# Non-goals (this plan)

- VBR dynamic ladder + runtime transcoding (Wave 2's `vbr-transcode.cu`): follow-up
  plan; needs the artifact/prompt-cache redesign first.
- Projected prompt artifacts (authenticated prefix snapshots).
- MTP/DSpark speculative sidecars: separate plan, bigger decode win.
- Classic ladder for BailingMoE3/Ling geometries.
- ROCm/HIP: CUDA-only; TurboQuant types on non-CUDA backends refuse at init
  (mirror buun's typed refusal).

# Agent execution notes

- One wave = one agent session. Each wave's Verification section is the session's
  acceptance criteria; do not start a wave with the previous wave's gates red.
- Kernel work happens in `python/freetoken/kernel/csrc/jit/turbo_kv.cu` + a
  `python/freetoken/kernel/turbo_kv.py` loader module following `store.py`'s
  `_jit_store_module` pattern exactly (KernelConfig, `load_jit`,
  `check_nvcc_matches_torch`). nvcc is at `/usr/local/cuda/bin` (already wired into
  the venv activate).
- Reference code to copy from: `/home/sherntee/20llms/buun-llama-cpp/ggml/src/ggml-cuda/`
  (turbo-* and the set_rows kernels in `turbo-quant-cuda.cuh`) and
  `ggml/src/ggml-common.h` block structs. Keep the block struct bit layouts
  byte-identical so codebooks and dump tooling stay interoperable.
- The 5070 Ti is sm_120 (Blackwell). FWHT/Viterbi kernels are simple enough to run
  on any arch, but add `-arch=sm_120` awareness via the existing toolchain module
  rather than hardcoding.
- Local coder model: `omp --model freetoken/qwen3.6-35b-a3b` (server on 1919, config
  in `~/.omp/agent/models.yml`). Keep high-level decisions in the orchestrating
  session; the agent handles file-level work.