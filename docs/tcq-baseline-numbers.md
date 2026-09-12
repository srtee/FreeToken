# TCQ KV codec — measured baseline numbers

Machine: RTX 5070 Ti 16 GB (driver 595.58.03, CUDA 13.2), 28-core CPU.
Soak checkpoint: Qwen2.5-Coder-32B-Instruct IQ3_XXS (64 layers, 8 kv heads,
head_dim 128 — the largest head_dim-128 checkpoint that fits the card; the
plan's Qwen3.6 35B pick has head_dim 256, which the turbo pool rejects by
design). Battery: timed 2K/8K-context prefill + 128-token decode at
temperature 0; VRAM sampled post-load. Run through the Slurm chain in
`soak/` (see `soak/SLURM-PLAN.md`).

## Compression (arithmetic, codec-exact)

| Codec | bpv | K+V bytes/token | vs f16 |
|---|---|---|---|
| f16 | 16 | 262144 | 1.0x |
| turbo8 | 8.125 | 133120 | 1.97x |
| turbo4 | 4.125 | 67584 | 3.88x |
| turbo3_tcq | 3.25 | 53248 | 4.92x |
| turbo2_tcq | 2.25 | 36864 | 7.11x |

## Server battery (measured)

| Codec | VRAM MiB (weights+KV 8192 tok) | 2K prefill+128 dec (s) | 8K prefill+128 dec (s) |
|---|---|---|---|
| f16 | 14940 | 11.78 | 22.93 |
| turbo8 | pending | pending | pending |
| turbo4 | pending | pending | pending |
| turbo3_tcq | pending | pending | pending |

## 30-min turbo3_tcq soak

Pending — `soak/soak30.sh` under Slurm (`soak30-turbo3` job). Drift signal:
repeated output hashes across turns on identical prompts, monotonic length
decay, or ERROR lines in `/tmp/soak30_results.txt`.

## Quality gates already measured (torch oracle, synthetic)

- TCQ bitstreams (turbo3_tcq / turbo2_tcq) byte-exact vs the torch Viterbi
  oracle; turbo4/turbo8 roundtrips match the Lloyd-Max oracle to fp16
  rounding (tests/kernels).
- InnerQ (per-channel equalization) shrinks weak-channel scaled-domain
  relative error >=2x on anisotropic inputs; real-checkpoint calibration on
  Qwen2.5-Coder-14B: 32752 groups, max ratio 1.42-1.51.
- 14B turbo4 known limitation: quality degrades past ~10 decode tokens on
  14B-class models; turbo8 is safe at any scale (models.md table).