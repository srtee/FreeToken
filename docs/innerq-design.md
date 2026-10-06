# InnerQ per-channel equalization — design notes (TCQ plan item 3.1)

## What buun does (reference: `buun-llama-cpp/ggml/src/ggml-cuda/turbo-quant-cuda.cuh`)

- Per-channel scale `s[128]` shared by K and V, applied BEFORE L2-norm
  and FWHT at encode (`x *= s; norm; rotate`).
- Calibration: during a probe window, `d_innerq_channel_sq[j] +=
  x[j]^2` (atomicAdd), `d_innerq_channel_max[j] = max(|x[j]|)` (CAS
  loop), `d_innerq_count += 1` per 128-group, accumulated over both K
  and V groups pre-norm.
- Finalize: `s[i] = (mean_rms / channel_rms)^strength`, clamped to
  `[1/2, 2]` (max-based mode: `1/sqrt(max)`); scales with max ratio
  < 1.2 are considered balanced and disabled. Q-side inverse
  `scale_inv` is applied in the FA kernel when Q meets dequantized K.
- Armed non-identity scales force the decode off the fused kernel onto
  the materialize path in buun.

## ft adaptation (this repo's structure differs in one key way)

ft's decode path is a **materializer**: `turbo_dequant_kernel` produces
original-domain K/V in fp16 scratch, consumed by stock FA/FI kernels.
Buun's decode dequantizes inside its fused FA and multiplies by
`scale_inv` there. In ft the equivalent place is the tail of
`turbo_dequant_kernel`: after the inverse rotation and `* inv_sqrt *
s1`, multiply channel-wise by `1/s[j]` before the final store. The
norm slot already carries the scaled-domain group norm, so the dequant
output is in the scaled domain; unscale restores the original domain.

Mathematically: encode error `e` is added to `x·s` in the scaled
domain; after decode `x̂ = x + e/s` — relative quantization noise is
equalized across channels, which is exactly the equalization goal.
No Q-side change, no attention backend changes, no softmax
considerations (original domain restored).

## Where the taps go

- `turbo_quant_kernel` (turbo4/turbo8) and the TCQ Viterbi encoder:
  before the L2-norm reduction, `x[tid] *= s[tid]`; during calibration
  also accumulate `x²` and `max|x|` on the pre-scale values (matching
  buun).
- `turbo_dequant_kernel`: at the tail, `v *= s_inv[tid]` — buun's
  dequant order is `val * inv_sqrt_128 * s1 * scale_inv * norm`
  (commutative; place anywhere before the store).
- Calibration accumulators: `__device__ float
  d_innerq_channel_sq[128]; __device__ float
  d_innerq_channel_max[128]; __device__ int d_innerq_count;
  __device__ int d_innerq_calibrate;` — one global set shared by K and
  V, same as buun.

## Calibration state machine (ft)

- `TurboKVCache` arms calibration on the first `store_kv` after pool
  creation when `config.kv_codec_tune == "innerq"`.
- Every quantize launch during the window sets `calibrate=1`; each
  128-group contributes one count.
- After the first `N` tokens quantized (pool-side counter, default
  2048 tokens), the pool copies the accumulators to CPU, computes `s`
  (RMS mode, strength 0.5, clamp 2.0), uploads `scale`/`scale_inv` to
  the device symbols, and disarms calibration. If max ratio < 1.2,
  scales reset to identity (buun's auto-disable) and the pool logs it.
- KLD gates: after calibration, run the Wave-0 oracle harness with
  `--kv-codec-tune innerq` and compare KLD against the no-tune
  baseline on the same checkpoint.

## Non-goals for this pass

- No K-mean subtraction (buun's `TURBO_KMEAN_SUB` — separate probe).
- No fused-FA decode path (ft has no fused TCQ decode yet; the
  materializer is the only path, so buun's fused-kernel caveat does
  not apply).
- No max-based mode (paper's formula) — RMS mode only, matching buun's
  default; the mode knob stays available for later.