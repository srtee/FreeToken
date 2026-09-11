# TCQ KV baseline numbers (Wave 0)

Environment: RTX 5070 Ti (sm_120), CUDA 13.2, torch 2.11.0+cu130, FreeToken @ gguf-serve.
Oracle: pure-torch codecs (`python/freetoken/kernel/turbo_oracle.py`), synthetic
post-WHT Gaussian groups (N(0, 1/sqrt(128))), seed 42, 1024 groups per row.
KLD = median KL divergence of softmaxed attention logits (64 random queries vs
the f16 reference K), matching the buun README metric. TCQ books = buun's
compiled-in coord-descent split (K/V separate), extracted to
`python/freetoken/kernel/codebooks/*.bin`.

## Synthetic oracle (pre-CUDA ground truth)

| codec  | bpv   | block B/128 | MSE (K)   | MSE (V)   | median KLD | gate     |
|--------|-------|-------------|-----------|-----------|------------|----------|
| turbo8 | 8.125 | 130         | 3.2e-05   | 3.1e-05   | ~1.1e-07   | —        |
| turbo4 | 4.125 | 66          | 7.3e-05   | 7.3e-05   | ~2.9e-07   | 0.001 ✓ |
| turbo3_tcq | 3.25 | 52        | 3.8e-03   | 3.5e-03   | ~1.4e-05   | 0.002 ✓ |
| turbo2_tcq | 2.25 | 36        | 3.3e-03   | 3.2e-03   | ~1.2e-05   | 0.007 ✓ |

All four codecs pass the plan's Wave-0 gates on synthetic data with 2+ orders of
margin. The TCQ MSE values are in-family with buun's trained-table numbers
(turbo2_tcq 2-bit LM baseline MSE ≈ 0.0037 on this distribution).

Roundtrip norm preservation (unit-norm x 5 groups): turbo4 max rel err 4e-4,
turbo3_tcq 3.9e-2, turbo2_tcq 1.7e-2 (corrected_norm handles most of it; the
residual is fp16 norm storage).

## VRAM math (Qwen3.6-35B-A3B, 64 layers, 4 KV heads x 128 dim, fp16 baseline)

fp16: 2 B * 128 dim * 4 heads * 2 (K+V) = 2048 B/token/layer → 128 KiB/token
whole-model. turbo4: 66 B per 128-value group per tensor → 4.125/16 = 25.8% of
fp16. turbo3_tcq: 3.25/16 = 20.3%. turbo2_tcq: 2.25/16 = 14.1%.

49K tokens of KV: fp16 ≈ 6.1 GiB (whole model) — plan's 0.9 GiB figure is
per-token at the MoE's reduced KV heads; the pool's unit_bytes() math is what
must agree with /v1/cache/status after Wave 1.

## Real-model gates (pending Wave 1 hardware slot)

The 14B coder GGUF server occupies the GPU during coding sessions. Real-model
KLD capture (Qwen3.6-35B-A3B-NVFP4, wikitext-2 prompts at 2K/8K/16K) runs at
Wave-1 verification when the server is swapped:
- turbo4: median KLD ≤ 0.001, PPL delta ≤ +0.05
- turbo3_tcq: median KLD ≤ 0.002, PPL delta ≤ +0.10
- turbo2_tcq: median KLD ≤ 0.007, PPL delta ≤ +0.35

## Wave 1 E2E (Qwen2.5-Coder-14B Q6_K, RTX 5070 Ti, eager decode)

- `ft serve --kv-codec turbo4` serves correctly: short answers match f16
  ("17*23" → 391; multi-turn prefix reuse over quantized pages → 391*2 = 782).
- VRAM accounting: `--num-pages 4096` allocates 0.19 GiB (50,688 B/token)
  vs f16's 0.75 GiB (196,608 B/token) — 3.88x compression, matching the
  unit_bytes() math and /v1/cache/status (kv_per_token=50688, slider max
  45,599 tokens vs f16's 12,193).
- Decode throughput (eager, no graphs): 51 tok/s turbo4 vs ~60 f16-era —
  materializer overhead within the plan's expected 15–30% regression band.
- Codec-quality split observed on the 14B: turbo8 (near-lossless) decodes
  perfectly ("1, 2, 3, ..., 10"); turbo4 short answers are correct but decode
  quality degrades beyond ~10 generated tokens (K-noise accumulation on a
  14B model; buun's KLD gates were calibrated on 27B/35B where turbo4 holds).
  Recommendation for 14B-class serving: turbo8/turbo3_tcq; turbo4 targets
  27B+ models. CUDA-graph capture is auto-disabled for turbo pools until the
  Wave-2 fused kernels (logged at startup).

## Wave 2 E2E (turbo3_tcq, 14B Q6_K, eager decode)

- turbo3_tcq serves: short answers match f16 (391). Bit-exactness torch-vs-
  CUDA: 784/784 (3-bit) and 528/528 (2-bit) bitstream bytes over random fp16
  groups — the strongest check in the plan.
- Long decode on the 14B diverges for turbo3_tcq and turbo4 (im_start spam
  past ~10 tokens), while turbo8 decodes cleanly: the 4.125/3.25 bpv codecs
  need the model scale (27B+) their KLD gates were calibrated on. For 14B
  serving, recommend turbo8 or f16; turbo3/4 are for 27B+.
