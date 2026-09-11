// TurboQuant/TCQ KV cache quantize + dequantize kernels (FreeToken port).
//
// Ported from buun-llama-cpp (MIT, ggml authors / spiritbuun):
//   turbo-wht.cu            FWHT butterfly with seed-42 sign tables
//   turbo-quant-cuda.cuh    set_rows quantizers (turbo4/turbo8 + TCQ Viterbi)
//   fattn.cu                sliding-window TCQ dequantizers
//   ggml-common.h           block struct byte layouts (kept byte-identical)
//
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <tvm/ffi/container/tensor.h>
//   turbo3_tcq: norm fp16 (2B) + 49 B bitstream + 1 B pad        = 52 B
//   turbo2_tcq: norm fp16 (2B) + 33 B bitstream + 1 B pad        = 36 B
//
// Quantize kernel (one block per (row, head) group, 128 threads):
//   load f32 group -> L2 norm -> normalize -> FWHT(signs1, signs2) ->
//   scalar quantize against the codec table -> pack -> store norm.
// Dequant kernel: inverse scatter, O(1) per element for turbo4/turbo8; for TCQ
// a sliding-window bitstream read (width 9/8 bits at offset t*3/t*2).
//
// The Wave-1 pool stores packed slabs in (2, L, tokens, heads, block_bytes)
// uint8 buffers; K uses the K codebook, V the V codebook (TCQ split books).

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kGroup = 128;
constexpr float kInvSqrt128 = 0.08838834764831845f;

// ---- seed-42 FWHT sign tables (turbo-wht.cu, verbatim) ----------------------
__constant__ float d_turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
__constant__ float d_turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// ---- codec tables (turbo-quant-cuda.cuh, verbatim) --------------------------
__constant__ float d_turbo_centroids_4bit[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
__constant__ float d_turbo_mid_4bit[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};
// turbo8: uniform grid centroid[i] = (i - 127.5) / 127.5; no constant needed
// (computed on the fly). Per-block absmax scale lives in the norm slot.

__device__ __forceinline__ uint8_t turbo_find_nearest_4bit(float val) {
    // Binary search over the 15 midpoints (turbo-quant-cuda.cuh verbatim).
    if (val < d_turbo_mid_4bit[7]) {
        if (val < d_turbo_mid_4bit[3]) {
            if (val < d_turbo_mid_4bit[1]) {
                return val < d_turbo_mid_4bit[0] ? 0 : 1;
            } else {
                return val < d_turbo_mid_4bit[2] ? 2 : 3;
            }
        } else {
            if (val < d_turbo_mid_4bit[5]) {
                return val < d_turbo_mid_4bit[4] ? 4 : 5;
            } else {
                return val < d_turbo_mid_4bit[6] ? 6 : 7;
            }
        }
    } else {
        if (val < d_turbo_mid_4bit[11]) {
            if (val < d_turbo_mid_4bit[9]) {
                return val < d_turbo_mid_4bit[8] ? 8 : 9;
            } else {
                return val < d_turbo_mid_4bit[10] ? 10 : 11;
            }
        } else {
            if (val < d_turbo_mid_4bit[13]) {
                return val < d_turbo_mid_4bit[12] ? 12 : 13;
            } else {
                return val < d_turbo_mid_4bit[14] ? 14 : 15;
            }
        }
    }
}

// TCQ codebooks: runtime-loaded device buffers (512/256 f32), K and V books.
// Uploaded by turbo_kv.py at module init via cudaMemcpyToSymbol on the pointer
// holder below (simpler than __constant__ for runtime override support).
__device__ float d_tcq3_codebook_k[512];
__device__ float d_tcq3_codebook_v[512];
__device__ float d_tcq2_codebook_k[256];
__device__ float d_tcq2_codebook_v[256];

extern "C" void turbo_upload_codebooks(const float* cb3k, const float* cb3v,
                                       const float* cb2k, const float* cb2v,
                                       void* stream) {
    cudaMemcpyToSymbolAsync(d_tcq3_codebook_k, cb3k, 512 * sizeof(float), 0,
                            cudaMemcpyHostToDevice, (cudaStream_t)stream);
    cudaMemcpyToSymbolAsync(d_tcq3_codebook_v, cb3v, 512 * sizeof(float), 0,
                            cudaMemcpyHostToDevice, (cudaStream_t)stream);
    cudaMemcpyToSymbolAsync(d_tcq2_codebook_k, cb2k, 256 * sizeof(float), 0,
                            cudaMemcpyHostToDevice, (cudaStream_t)stream);
    cudaMemcpyToSymbolAsync(d_tcq2_codebook_v, cb2v, 256 * sizeof(float), 0,
                            cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

// ---- kernel params ----------------------------------------------------------
//
// src: (L, heads, 128) f32/fp16/bf16 KV values to quantize.
// dst: (L, heads, block_bytes) uint8 slabs.
// locs: (L,) int32/int64 destination rows (the out_loc scatter).
// The cache slab is (tokens, heads, block_bytes) uint8, indexed dst_row = locs[l].

// dtype dispatch helper: fp16/bf16/f32 all convert to float.
template <typename T> __device__ __forceinline__ float turbo_load_as_float(const T *p) {
  return (float)*p;
}
template <> __device__ __forceinline__ float turbo_load_as_float<__nv_bfloat16>(const __nv_bfloat16 *p) {
  return __bfloat162float(*p);
}

struct TurboQuantParams {
  const void *__restrict__ src;   // (L, heads, 128)
  void *__restrict__ dst;         // (tokens, heads, block_bytes)
  const void *__restrict__ locs;  // (L,)
  int length;                     // L
  int heads;
  int block_bytes;
  int codec;                      // 4=turbo4, 8=turbo8, 32=turbo3_tcq, 22=turbo2_tcq
  int is_v;                       // selects the V codebook for TCQ
  int locs_is_int32;
};

// ---- quantize: one CUDA block per (row, head) group -------------------------
//
// 128 threads: load+sign+FWHT via warp shuffles (first 5 stages in-warp, last
// two cross-warp through shared memory, same shape as buun's TCQ encoders).
template <typename T>
__global__ void turbo_quant_kernel(const __grid_constant__ TurboQuantParams p) {
  const int group = blockIdx.x;
  if (group >= p.length * p.heads) return;
  const int row = group / p.heads;
  const int head = group % p.heads;
  const int tid = threadIdx.x;

  const T *src = (const T *)p.src;
  const int64_t dst_row = p.locs_is_int32
      ? (int64_t)((const int32_t *)p.locs)[row]
      : ((const int64_t *)p.locs)[row];

  __shared__ float x[128];

  x[tid] = turbo_load_as_float(&src[(int64_t)row * p.heads * 128 + head * 128 + tid]);
  __syncthreads();

  // L2 norm reduction
  __shared__ float red[128];
  red[tid] = x[tid] * x[tid];
  __syncthreads();
#pragma unroll
  for (int stride = 64; stride >= 1; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  const float grp_norm = sqrtf(red[0]);
  const float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
  x[tid] *= inv_norm;
  __syncthreads();

  // FWHT: first five stages within a warp via shuffles, last two via smem.
  {
    const int lane = tid & 31;
    float v = x[tid] * d_turbo_wht_s1[tid];
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, v, h);
      v = (lane & h) ? (other - v) : (v + other);
    }
    x[tid] = v;
  }
  __syncthreads();
  if (tid < 64) {
    const int j = ((tid >> 5) << 6) + (tid & 31);
    float a = x[j], b = x[j + 32];
    x[j] = a + b; x[j + 32] = a - b;
  }
  __syncthreads();
  if (tid < 64) {
    float a = x[tid], b = x[tid + 64];
    x[tid] = a + b; x[tid + 64] = a - b;
  }
  __syncthreads();
  x[tid] *= kInvSqrt128 * d_turbo_wht_s2[tid];
  __syncthreads();

  uint8_t *dst_base = (uint8_t *)p.dst +
      dst_row * (int64_t)p.heads * p.block_bytes + head * p.block_bytes;

  if (p.codec == 4) {
    // turbo4: nibble pack (low nibble first, even j in low nibble)
    uint8_t idx = turbo_find_nearest_4bit(x[tid]);
    if ((tid & 1) == 0) {
      dst_base[2 + (tid >> 1)] = idx;
    } else {
      dst_base[2 + (tid >> 1)] |= idx << 4;
    }
    // recon norm for correction: each thread contributes its centroid^2
    red[tid] = d_turbo_centroids_4bit[idx] * d_turbo_centroids_4bit[idx];
    __syncthreads();
#pragma unroll
    for (int stride = 64; stride >= 1; stride >>= 1) {
      if (tid < stride) red[tid] += red[tid + stride];
      __syncthreads();
    }
    const float recon_norm = sqrtf(red[0]);
    if (tid == 0) {
      const float corrected = recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm;
      const uint16_t bits = __half_as_ushort(__float2half(corrected));
      dst_base[0] = (uint8_t)(bits & 0xFF);
      dst_base[1] = (uint8_t)((bits >> 8) & 0xFF);
    }
  } else if (p.codec == 8) {
    // turbo8: per-block absmax + uniform grid
    __shared__ float absmax_sh;
    float ax = fabsf(x[tid]);
    red[tid] = ax;
    __syncthreads();
#pragma unroll
    for (int stride = 64; stride >= 1; stride >>= 1) {
      if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
      __syncthreads();
    }
    if (tid == 0) absmax_sh = fmaxf(red[0], 1e-10f);
    __syncthreads();
    const float scale = absmax_sh;
    const float inv_scale = 1.0f / scale;
    float q = x[tid] * inv_scale * 127.5f + 127.5f;
    int idx = (int)lrintf(q);
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    dst_base[2 + tid] = (uint8_t)idx;
    if (tid == 0) {
      const float corrected = grp_norm * scale;  // norm slot carries ||x|| * absmax
      const uint16_t bits = __half_as_ushort(__float2half(corrected));
      dst_base[0] = (uint8_t)(bits & 0xFF);
      dst_base[1] = (uint8_t)((bits >> 8) & 0xFF);
    }
  }
}

// ---- TCQ Viterbi encoder: one block per group, kStates threads ---------------
//
// Serial over 128 steps; each thread owns one trellis state. Double-buffered
// costs, one barrier per step, thread-0 backtrack (buun's k_set_rows_turbo*_tcq
// layout, minus the InnerQ/mean-subtract taps). Forward from state 0; the
// bitstream is the 6-bit (zero) prefix + 128 output symbols; the sliding-window
// decode reads states at bit offset t*kBits.
template <int kBits, typename T>
__global__ void __launch_bounds__(1 << (6 + kBits), 2)
turbo_tcq_quant_kernel(const __grid_constant__ TurboQuantParams p) {
  constexpr int kStates = 1 << (6 + kBits);
  constexpr int kLow = 64;
  const int group = blockIdx.x;
  if (group >= p.length * p.heads) return;
  const int row = group / p.heads;
  const int head = group % p.heads;
  const int sid = threadIdx.x;

  const T *src = (const T *)p.src;
  const int64_t dst_row = p.locs_is_int32
      ? (int64_t)((const int32_t *)p.locs)[row]
      : ((const int64_t *)p.locs)[row];

  __shared__ float x[128];
  __shared__ float cost[kStates];
  __shared__ float cost_b[kStates];
  __shared__ float warp_min_cost[kStates / 32];
  __shared__ int warp_min_idx[kStates / 32];
  __shared__ int shared_initial_state;
  __shared__ uint8_t bt[128 * kLow];

  if (sid < 128) {
    x[sid] = turbo_load_as_float(&src[(int64_t)row * p.heads * 128 + head * 128 + sid]);
  }
  __syncthreads();

  // L2 norm via cost buffer (128 values over kStates threads)
  if (sid < 128) cost[sid] = x[sid] * x[sid];
  else cost[sid] = 0.0f;
  __syncthreads();
  for (int stride = kStates / 2; stride >= 128; stride >>= 1) {
    if (sid < stride) cost[sid] += cost[sid + stride];
    __syncthreads();
  }
  if (sid < 128) {
    float v = cost[sid];
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    if ((sid & 31) == 0) cost[sid] = v;
  }
  __syncthreads();
  // Cross-warp combine: the shuffle left 4 partials (cost[0,32,64,96]).
  if (sid == 0) {
    cost[0] += cost[32] + cost[64] + cost[96];
  }
  __syncthreads();
  const float grp_norm = sqrtf(cost[0]);
  const float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

  // Normalize, rotate (same shape as turbo_quant_kernel's FWHT)
  if (sid < 128) {
    float v = x[sid] * inv_norm;
    const int lane = sid & 31;
    v *= d_turbo_wht_s1[sid];
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, v, h);
      v = (lane & h) ? (other - v) : (v + other);
    }
    x[sid] = v;
  }
  __syncthreads();
  if (sid < 64) {
    const int j = ((sid >> 5) << 6) + (sid & 31);
    float a = x[j], b = x[j + 32];
    x[j] = a + b; x[j + 32] = a - b;
  }
  __syncthreads();
  if (sid < 64) {
    float a = x[sid], b = x[sid + 64];
    x[sid] = a + b; x[sid + 64] = a - b;
  }
  __syncthreads();
  if (sid < 128) x[sid] *= kInvSqrt128 * d_turbo_wht_s2[sid];
  __syncthreads();

  const float *cb = p.is_v
      ? (kBits == 3 ? d_tcq3_codebook_v : d_tcq2_codebook_v)
      : (kBits == 3 ? d_tcq3_codebook_k : d_tcq2_codebook_k);
  const float cb_sid = cb[sid];

  // Viterbi forward from state 0: cost[0] = 0, rest INF.
  cost[sid] = (sid == 0) ? 0.0f : 3.4028234663852886e38f;
  __syncthreads();

  for (int t = 0; t < 128; t++) {
    float *cost_rd = (t & 1) ? cost_b : cost;
    float *cost_wr = (t & 1) ? cost : cost_b;
    const float xt = x[t];

    // Predecessors of state s are the kStates/kOut states whose high bits ==
    // s's low 6 bits (ns = (prev >> kBits) | (out << 6) => prev >> kBits
    // covers ns & 0x3F after shifting right... in this encoding the predecessor
    // GROUP of s is (s & 0x3F) << kBits .. +kOut-1? No: ns low 6 bits = prev
    // high 6 bits (prev >> kBits). So prev group g = s & 0x3F gives prev =
    // (g << kBits) | p for p in [0, kOut). That's the buun scan:
    //   base_prev = (s & 0x3F) << kBits; scan cost_rd[base_prev + p].
    const int g = sid & 0x3F;
    const int base_prev = g << kBits;
    float best = cost_rd[base_prev];
    int best_p = 0;
#pragma unroll
    for (int pp = 1; pp < (1 << kBits); pp++) {
      const float c = cost_rd[base_prev | pp];
      if (c < best) { best = c; best_p = pp; }
    }
    if (sid < kLow) bt[t * kLow + sid] = (uint8_t)best_p;

    const float dist = xt - cb[sid];
    cost_wr[sid] = best + dist * dist;
    __syncthreads();
  }
  // 128 steps (even): final costs in cost[].

  // Final-state argmin over kStates values.
  {
    float my_cost = cost[sid];
    int my_idx = sid;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float oc = __shfl_xor_sync(0xFFFFFFFFu, my_cost, offset);
      const int oi = __shfl_xor_sync(0xFFFFFFFFu, my_idx, offset);
      if (oc < my_cost) { my_cost = oc; my_idx = oi; }
    }
    if ((sid & 31) == 0) {
      warp_min_cost[sid >> 5] = my_cost;
      warp_min_idx[sid >> 5] = my_idx;
    }
  }
  __syncthreads();
  if (sid < 32) {
    float best = (sid < kStates / 32) ? warp_min_cost[sid] : 3.4028234663852886e38f;
    int best_idx = (sid < kStates / 32) ? warp_min_idx[sid] : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float oc = __shfl_down_sync(0xFFFFFFFFu, best, offset);
      const int oi = __shfl_down_sync(0xFFFFFFFFu, best_idx, offset);
      if (oc < best) { best = oc; best_idx = oi; }
    }
    if (sid == 0) shared_initial_state = best_idx;
  }
  __syncthreads();

  // Backtrack by thread 0: outputs[t] = state >> 6, rewind via bt.
  __shared__ uint8_t outputs[128];
  if (sid == 0) {
    int state = shared_initial_state;
    for (int t = 127; t >= 0; t--) {
      outputs[t] = (uint8_t)(state >> 6);
      const int pp = bt[t * kLow + (state & 0x3F)];
      state = ((state & 0x3F) << kBits) | pp;
    }
    shared_initial_state = state;  // == 0 (forward started there)
  }
  __syncthreads();

  // Parallel recon norm: t >= kOut windows reconstruct the state from symbols.
  if (sid < 128) {
    int cur_state;
    if (sid < 6 / kBits + 1) {
      cur_state = shared_initial_state;
      for (int t = 0; t <= sid; t++) {
        cur_state = (cur_state >> kBits) | ((int)outputs[t] << 6);
      }
    } else {
      // state_t = out_{t} << 6 | out_{t-1} << (6-kBits) | ... | out_{t-6/kBits} << 0
      // (buun's parallel recon: outputs[sid-N] low ... outputs[sid] << 6 high).
      cur_state = 0;
      for (int j = 0; j <= 6 / kBits; j++) {
        const int shift = 6 - j * kBits;
        cur_state |= ((int)outputs[sid - j] & ((1 << kBits) - 1)) << shift;
      }
    }
    cost[sid] = cb[cur_state] * cb[cur_state];
  } else {
    cost[sid] = 0.0f;
  }
  __syncthreads();
  for (int stride = kStates / 2; stride >= 128; stride >>= 1) {
    if (sid < stride) cost[sid] += cost[sid + stride];
    __syncthreads();
  }
  if (sid < 128) {
    float v = cost[sid];
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    if ((sid & 31) == 0) cost[sid] = v;
  }
  __syncthreads();
  if (sid == 0) {
    cost[0] += cost[32] + cost[64] + cost[96];
  }
  __syncthreads();
  const float recon_norm = sqrtf(cost[0]);
  float corrected = recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm;

  // Bitpack: thread 0 packs the 6-bit prefix (zeros — forward starts at
  // state 0, so shared_initial_state after rewind == 0 and the prefix is 0);
  // threads 2..(2 + nbytes) each own one stream byte.
  uint8_t *dst_base = (uint8_t *)p.dst +
      dst_row * (int64_t)p.heads * p.block_bytes + head * p.block_bytes;
  const int n_bytes = p.block_bytes - 3;
  if (sid == 0) {
    const uint16_t bits = __half_as_ushort(__float2half(corrected));
    dst_base[0] = (uint8_t)(bits & 0xFF);
    dst_base[1] = (uint8_t)((bits >> 8) & 0xFF);
    dst_base[2 + n_bytes] = 0;  // pad byte
  }
  if (sid < n_bytes) {
    uint8_t packed = 0;
#pragma unroll
    for (int bit = 0; bit < 8; bit++) {
      const int pos = sid * 8 + bit;
      int v = 0;
      if (pos < 6) {
        v = 0;  // prefix: initial state 0
      } else {
        const int sym_bit_pos = pos - 6;
        const int sym_idx = sym_bit_pos / kBits;
        if (sym_idx < 128) {
          v = (outputs[sym_idx] >> (sym_bit_pos % kBits)) & 1;
        }
      }
      packed |= (uint8_t)(v << bit);
    }
    dst_base[2 + sid] = packed;
  }
}

// ---- dequant: gather+expand, 128 threads per (row, head) group ---------------
struct TurboDequantParams {
  const void *__restrict__ src;   // (tokens, heads, block_bytes) uint8
  void *__restrict__ dst;         // (L, heads, 128) f16
  const void *__restrict__ locs;  // (L,) source rows
  int length;
  int heads;
  int block_bytes;
  int codec;                      // 4=turbo4, 8=turbo8, 32=turbo3_tcq, 22=turbo2_tcq
  int is_v;
  int locs_is_int32;
  int dst_is_bf16;
};

__device__ __forceinline__ float turbo8_centroid(int idx) {
  return ((float)idx - 127.5f) / 127.5f;
}

__global__ void turbo_dequant_kernel(const __grid_constant__ TurboDequantParams p) {
  const int group = blockIdx.x;
  if (group >= p.length * p.heads) return;
  const int row = group / p.heads;
  const int head = group % p.heads;
  const int tid = threadIdx.x;

  const uint8_t *src = (const uint8_t *)p.src;
  const int64_t src_row = p.locs_is_int32
      ? (int64_t)((const int32_t *)p.locs)[row]
      : ((const int64_t *)p.locs)[row];
  const uint8_t *blk = src + src_row * (int64_t)p.heads * p.block_bytes
                     + head * p.block_bytes;
  // dst element type: fp16 or bf16 (16-bit float both ways)
  const bool dst_is_bf16 = p.dst_is_bf16 != 0;
  float *out_f = nullptr;
  __half *out_h = nullptr;
  __nv_bfloat16 *out_b = nullptr;
  if (dst_is_bf16) {
    out_b = (__nv_bfloat16 *)p.dst
          + (int64_t)row * p.heads * 128 + head * 128 + tid;
  } else {
    out_h = (__half *)p.dst
          + (int64_t)row * p.heads * 128 + head * 128 + tid;
  }

  const uint16_t norm_raw = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
  const float norm = __half2float(__ushort_as_half(norm_raw));

  float val;
  if (p.codec == 4) {
    const uint8_t byte = blk[2 + (tid >> 1)];
    const uint8_t idx = (tid & 1) ? (byte >> 4) : (byte & 0xF);
    val = d_turbo_centroids_4bit[idx] * norm;
  } else if (p.codec == 8) {
    val = turbo8_centroid(blk[2 + tid]) * norm;
  } else {
    const int bits = (p.codec == 32) ? 3 : 2;
    const int bit_pos = tid * bits;
    const int byte_idx = bit_pos >> 3;
    const int bit_off = bit_pos & 7;
    // Sliding window: the +1 byte is only needed when the window crosses a
    // byte boundary — the last byte of the block (qs[N-1]) has no successor
    // and unconditionally reading it walks past the row.
    const uint16_t lo = blk[2 + byte_idx];
    const uint16_t hi = (bit_off + 6 + bits > 8) ? blk[2 + byte_idx + 1] : 0;
    const uint16_t raw = lo | (hi << 8);
    const int state = (raw >> bit_off) & ((1 << (6 + bits)) - 1);
    const float *cb = (p.codec == 32)
        ? (p.is_v ? d_tcq3_codebook_v : d_tcq3_codebook_k)
        : (p.is_v ? d_tcq2_codebook_v : d_tcq2_codebook_k);
    val = cb[state] * norm;
  }

  // Inverse rotation to the original (pre-FWHT) domain — buun's
  // *_dequant_f16_inv_fwht: multiply by s2, butterfly, then * inv_sqrt * s1.
  // The materializer serves stock FA/FI kernels, which need original-domain K/V.
  __shared__ float smem[128];
  float v = val * d_turbo_wht_s2[tid];
  {
    const int lane = tid & 31;
    #pragma unroll
    for (int h = 1; h <= 16; h <<= 1) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, v, h);
      v = (tid & h) ? (other - v) : (v + other);
    }
  }
  __syncthreads();
  smem[tid] = v;
  __syncthreads();
  v = (tid & 32) ? (smem[tid - 32] - v) : (v + smem[tid + 32]);
  __syncthreads();
  smem[tid] = v;
  __syncthreads();
  v = (tid & 64) ? (smem[tid - 64] - v) : (v + smem[tid + 64]);
  __syncthreads();
  v = v * kInvSqrt128 * d_turbo_wht_s1[tid];

  if (dst_is_bf16) {
    *out_b = __float2bfloat16(v);
  } else {
    *out_h = __float2half(v);
  }
}

} // namespace

// Host launchers must be visible for TVM_FFI_DLL_EXPORT_TYPED_FUNC.
// ---- host-side launchers (tvm-ffi entry points) ------------------------------
template <int kCodec, int kBlockBytes>
struct TurboQuantLaunch {
  static void run(const tvm::ffi::TensorView src,
                  const tvm::ffi::TensorView dst,
                  const tvm::ffi::TensorView locs,
                  int64_t is_v) {
    using namespace host;
    auto L = SymbolicSize{"L"};
    auto H = SymbolicSize{"H"};
    auto D = SymbolicSize{"D"};   // 128
    auto Rows = SymbolicSize{"Rows"};
    auto dtype_ = SymbolicDType{};
    auto locs_dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L, H, D})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(src);
    TensorMatcher({Rows, H, kBlockBytes})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(dst);
    TensorMatcher({L})
        .with_dtype<int32_t, int64_t>(locs_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(locs);

    RuntimeCheck(D.unwrap() == 128);
    const auto device = device_.unwrap();
    const auto length = static_cast<int>(L.unwrap());
    const auto heads = static_cast<int>(H.unwrap());

    TurboQuantParams params{
        .src = src.data_ptr(),
        .dst = dst.data_ptr(),
        .locs = locs.data_ptr(),
        .length = length,
        .heads = heads,
        .block_bytes = kBlockBytes,
        .codec = kCodec,
        .is_v = static_cast<int>(is_v),
        .locs_is_int32 = locs_dtype_.unwrap().bits == 32 ? 1 : 0,
    };
    const int n_groups = length * heads;
    const auto dt = dtype_.unwrap();
    if (dt.code == kDLFloat && dt.bits == 32) {
      if (kCodec == 32) {
        LaunchKernel(dim3(n_groups), dim3(512), device)(turbo_tcq_quant_kernel<3, float>, params);
      } else if (kCodec == 22) {
        LaunchKernel(dim3(n_groups), dim3(256), device)(turbo_tcq_quant_kernel<2, float>, params);
      } else {
        LaunchKernel(dim3(n_groups), dim3(128), device)(turbo_quant_kernel<float>, params);
      }
    } else if (dt.code == kDLBfloat && dt.bits == 16) {
      if (kCodec == 32) {
        LaunchKernel(dim3(n_groups), dim3(512), device)(turbo_tcq_quant_kernel<3, __nv_bfloat16>, params);
      } else if (kCodec == 22) {
        LaunchKernel(dim3(n_groups), dim3(256), device)(turbo_tcq_quant_kernel<2, __nv_bfloat16>, params);
      } else {
        LaunchKernel(dim3(n_groups), dim3(128), device)(turbo_quant_kernel<__nv_bfloat16>, params);
      }
    } else if (dt.code == kDLFloat && dt.bits == 16) {
      if (kCodec == 32) {
        LaunchKernel(dim3(n_groups), dim3(512), device)(turbo_tcq_quant_kernel<3, __half>, params);
      } else if (kCodec == 22) {
        LaunchKernel(dim3(n_groups), dim3(256), device)(turbo_tcq_quant_kernel<2, __half>, params);
      } else {
        LaunchKernel(dim3(n_groups), dim3(128), device)(turbo_quant_kernel<__half>, params);
      }
    } else {
      RuntimeCheck(false, "turbo quantize: unsupported src dtype");
    }
  }
};

struct TurboDequantLaunch {
  static void run(const tvm::ffi::TensorView src,
                  const tvm::ffi::TensorView dst,
                  const tvm::ffi::TensorView locs,
                  int64_t codec,
                  int64_t is_v) {
    using namespace host;
    auto L = SymbolicSize{"L"};
    auto H = SymbolicSize{"H"};
    auto D = SymbolicSize{"D"};
    auto Rows = SymbolicSize{"Rows"};
    auto B = SymbolicSize{"B"};
    auto device_ = SymbolicDevice{};
    auto locs_dtype_ = SymbolicDType{};
    auto dst_dtype_ = SymbolicDType{};
    TensorMatcher({L, H, D})
        .with_dtype(dst_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(dst);
    TensorMatcher({Rows, H, B})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(src);
    TensorMatcher({L})
        .with_dtype<int32_t, int64_t>(locs_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(locs);

    RuntimeCheck(D.unwrap() == 128);
    RuntimeCheck(dst_dtype_.unwrap().bits == 16, "turbo dequantize: dst must be 16-bit float");
    const auto device = device_.unwrap();
    const auto length = static_cast<int>(L.unwrap());
    const auto heads = static_cast<int>(H.unwrap());
    RuntimeCheck(B.unwrap() == block_bytes_for(codec), "block_bytes mismatch");

    const auto locs_dt = locs_dtype_.unwrap();
    RuntimeCheck(
        (locs_dt.code == kDLInt || locs_dt.code == kDLUInt) && locs_dt.bits == 32 ||
        (locs_dt.code == kDLInt) && locs_dt.bits == 64,
        "turbo dequantize: locs must be int32 or int64");
    TurboDequantParams params{
        .src = src.data_ptr(),
        .dst = dst.data_ptr(),
        .locs = locs.data_ptr(),
        .length = length,
        .heads = heads,
        .block_bytes = static_cast<int>(B.unwrap()),
        .codec = static_cast<int>(codec),
        .is_v = static_cast<int>(is_v),
        .locs_is_int32 = locs_dt.bits == 32 ? 1 : 0,
        .dst_is_bf16 = dst_dtype_.unwrap().code == kDLBfloat ? 1 : 0,
    };
    const int n_groups = length * heads;
    LaunchKernel(dim3(n_groups), dim3(128), device)(turbo_dequant_kernel, params);
  }

  static int block_bytes_for(int64_t codec) {
    switch (codec) {
      case 4: return 66;
      case 8: return 130;
      case 32: return 52;
      case 22: return 36;
      default: return -1;
    }
  }
};

struct TurboCodebookUpload {
  static void run(const tvm::ffi::TensorView cb3k,
                  const tvm::ffi::TensorView cb3v,
                  const tvm::ffi::TensorView cb2k,
                  const tvm::ffi::TensorView cb2v) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    TensorMatcher({512}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(cb3k);
    TensorMatcher({512}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(cb3v);
    TensorMatcher({256}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(cb2k);
    TensorMatcher({256}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(cb2v);
    const auto device = device_.unwrap();
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
    turbo_upload_codebooks((const float *)cb3k.data_ptr(), (const float *)cb3v.data_ptr(),
                           (const float *)cb2k.data_ptr(), (const float *)cb2v.data_ptr(),
                           stream);
  }
};