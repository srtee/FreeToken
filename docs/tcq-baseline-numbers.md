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

## Server battery — cache-busted rerun (2026-09-13, triton fused decode)

32B IQ3_XXS, unique prompts per round (radix cache never short-circuits
prefill), server+client pinned to E-cores 8-11 during the LAMMPS run.
Warmup round discarded; second round measured.

| Codec | VRAM MiB | 2K prefill+128 dec (s) | 8K prefill+128 dec (s) |
|---|---|---|---|
| f16 | 15092 | 29.86 | 56.72 |
| turbo8 | 14852 | 24.12 | 28.34 |
| turbo4 | 14332 | 24.20 | 28.52 |
| turbo3_tcq | 14232 | 24.92 | 30.35 |

turbo8 is 2.0x f16 at 8K (fused packed-slab decode + CUDA graphs vs the
f16 kernel path) and 1.24x at 2K; VRAM tracks the compression table
(turbo3 -860 MiB vs f16 on 8192 tokens). Decode dominates the 8K wall —
the fused path is the win, not the codec's bandwidth alone.

## (superseded 2026-09-13) Server battery (measured)

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

## Qwen3.6-35B-A3B (head_dim 256) — turbo3_tcq soak + cross-config matrix (2026-09-21)

First soak of a head_dim-256 (dual rotation group) checkpoint. Findings:

- **Pool economics**: turbo3_tcq @ 262144 tokens (the full model cap) costs
  1.12 GiB — vs turbo8's 1.40 GiB @ 131072. Turbo3 doubles the addressable
  context at half the KV footprint.
- **Spec acceptance (quality canary)**: r = 0.57–0.90 per decode window,
  aggregate ~0.75 — indistinguishable from turbo8 (0.65–0.92). The 3.25-bpv
  TCQ storage does not degrade MTP draft quality on the 35B.
- **Byte-stability**: NOT soak-passing in the strict sense — but the
  investigation exonerated the codec. Greedy multi-turn soaks (3 prompts ×
  10 turns, 512 tok) diverge identically on turbo8 and turbo3_tcq, with and
  without --spec-mtp; concurrent duplicate requests are always identical.

### The cold-first-request artifact — RESOLVED (2026-09-22); not a cache or codec bug

The "first-reuse flip" was the **first HTTP request to a freshly started
server returning an empty response**. It only correlated with cache reuse
because in a cold soak pass-1 turn-0 IS request #1, and pass-2 turn-0 is the
radix-hot reply (9 chars) to the same prompt.

Evidence trail:
1. fp32 snapshot hardening (this commit): h_track boundary snapshots are now
   bit-exact vs the kernel's internal fp32 registers
   (`tests/kernels/test_fla_track_snapshot_fp32.py`). The flip persisted with
   the IDENTICAL signature `[[0,0,0,9]]` — the bf16-rounded-`h` theory is dead,
   and the hardening stands as correctness work.
2. Post-fix turbo8 plain cold soak: same `[[0,0,0,9]]`. 8-bit quant noise
   cannot flip a greedy tie identically to 3-bit — refutes quantization
   noise and, being deterministic, any noise model at all.
3. Decisive control: cold restart, one trivial warmup ping before the soak.
   The warmup returned **empty text** (0 chars on a "Say OK" prompt); the
   subsequent 10-turn × 3-prompt turbo3-plain soak was byte-STABLE
   (2026-09-22, `/tmp/soak_t3_after.json` run with warmup).

This also reconciles the old matrix: every DIVERGED cell ran with a cold
server (request #1 inside the soak); every STABLE cell (warm-tree, cache-hot
rerun) had a warm first request.

Follow-up ticket (separate, pre-existing): first request after server start
completes with 0 generated tokens and no error — suspected cold triton
autotune path dropping the response. One empty reply per restart until fixed.

## (historical) 30-min soak — RESOLVED (materializer page-id bug)

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