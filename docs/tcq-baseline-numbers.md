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
| f16 | 14940 | 11.78 | 22.94 |
| turbo8 | 13920 | 23.15 | 10.48 |
| turbo4 | 13420 | 23.18 | 24.02 |
| turbo3_tcq | 13300 | 23.52 | 24.30 |

Battery anomaly: turbo8's 8K wall (10.48s) is FASTER than f16 (22.94s) and
faster than its own 2K (23.15s) — inconsistent with the other codecs and
with prefill scaling; treat the battery wall-clock numbers as unreliable
(prefix-cache interaction suspected) until re-run with cache-busting.
VRAM deltas track the compression table (turbo3: 13300 vs f16 14940 MiB =
1.6 GiB saved on 8192 tokens ~= the predicted 172 KiB/token x 8192).

## 30-min turbo3_tcq soak — PASS (2026-09-13 rerun on fixed path)

31-min soak, 20 turns x 3 prompts, 32B IQ3_XXS: per-prompt outputs
deterministic, lengths identical across every turn (780/698/485 chars),
zero drift/decay/loops. Prior soak failure attributed to the materializer
page-id bug (fixed in 73f2bff).

## (historical) 30-min soak — RESOLVED (materializer page-id bug)

**ROOT CAUSE FOUND AND FIXED** (commit after b72236f): `materialize()`
dequantized rows COMPACTED into `scratch[:n]`, but the attention wrapper
indexes the returned tensor by ORIGINAL page id. Request 1 worked by luck
(fresh pool: page ids 0..12 < n=13); request 2+ wrote new tokens into
reused pages (id >= n) and FlashInfer read out-of-bounds scratch rows —
every request after the first was corrupted, on ALL turbo codecs, at ANY
precision. The A/B that framed this as "turbo3 quality" was wrong: turbo8
(0.64% relerr) showed the identical junk. Fix: dequant into staging,
scatter to page-id positions of the full-width scratch (extra
n x heads x 128 x 2B copy per layer per step). Verified: two-request repro
now byte-stable; turbo8/turbo4/turbo3_tcq all coherent on the soak prompts
(P1 609-720 chars vs f16 609; P2 367-432 vs f16 367); 490 tests green.

Original failure record (kept for the investigation trail):

17 turns x 3 prompts at temperature 0, 0 transport errors. Per-prompt drift
check against an f16 A/B on the identical checkpoint + prompts:

- P0 (dedup function): turbo3 == f16 quality (733 vs 734 chars, same
  opening). Long decodes CAN stay coherent.
- P1 (explain list comprehension): f16 609 chars coherent; turbo3 diverges
  from the FIRST decode token ('彻底.' + whitespace, 200 chars).
- P2 (refactor sum): f16 367 chars coherent; turbo3 whitespace-degenerate
  (identical md5 across all 16 turns — deterministic degeneration).

Token-1 divergence means the PREFILL KV is corrupted for these prompts —
not cumulative decode drift. The battery's padding prompts (x-repeat) hid
this: degenerate all-same-token KV encodes fine; real text KV does not.
turbo3_tcq is NOT shippable at 32B; investigation belongs in the decode
path (alpha_v adaptive scale, trellis encode of non-synthetic KV). turbo8
and turbo4 batteries completed; turbo4 quality on 14B is already flagged
(10-token limit), so treat turbo8 as the only soak-passing codec until
each is A/B-verified.

## Quality gates already measured (torch oracle, synthetic)

- TCQ bitstreams (turbo3_tcq / turbo2_tcq) byte-exact vs the torch Viterbi
  oracle; turbo4/turbo8 roundtrips match the Lloyd-Max oracle to fp16
  rounding (tests/kernels).
- InnerQ (per-channel equalization) shrinks weak-channel scaled-domain
  relative error >=2x on anisotropic inputs; real-checkpoint calibration on
  Qwen2.5-Coder-14B: 32752 groups, max ratio 1.42-1.51.
- 14B turbo4 known limitation: quality degrades past ~10 decode tokens on
  14B-class models; turbo8 is safe at any scale (models.md table).