# Qwen3.8-Flash-Next MTP verification runbook

Verifying the `qwen38-mtp` branch — the MTP speculative-decode draft head for
Qwen3.8-Flash-Next — on any CUDA box, in three tiers:

1. **CPU suites** — no GPU, no weights, ~30 s. Catches code regressions.
2. **Checkpoint-gated tests** — needs the model download, CPU is fine.
   Catches weight-mapping drift against the real checkpoint.
3. **GPU serve with `--spec-mtp`** — needs weights + a free GPU. The real
   acceptance: speculative decode engages and speeds up greedy decode.

## Prerequisites

- NVIDIA GPU, CUDA-13 capable (verified on Blackwell `sm_120`). 16 GB VRAM is
  enough: routed experts offload to host by default. See `docs/install.md`.
- Host RAM **≥ 64 GB**: the 47.7 GiB PLE n-gram table pins in host RAM at
  launch, on top of expert banks.
- Disk ~130 GB free for the checkpoint.
- Python 3.12 and `uv` (install: `docs/install.md`).

## Setup

```bash
git fetch <this-remote> qwen38-mtp && git checkout qwen38-mtp   # base: gguf-serve @ c833dfa
uv sync          # or bash install.sh; creates .venv
source .venv/bin/activate
```

## Model download (~126 GB)

```bash
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4
```

Any downloader works (aria2 with a manifest and a 4-connection cap was used
originally; the HF CDN caps ~5 MiB/s per session regardless of connection
count). Done when no `*.aria2` control files remain; the MTP tiers need
`model-bf16-00010/00011/00012.safetensors` complete, the serve needs
everything.

```bash
export FREETOKEN_QWEN4EXP_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4
```

## Tier 1 — CPU suites (no GPU, no weights)

```bash
pytest tests/models/qwen4_exp tests/layers tests/kernels/test_rotary.py -q
```

Expected: everything green except **one known pre-existing failure**:

```
tests/models/qwen4_exp/test_ple.py::test_track_snapshot_equals_a_prefill_stopped_at_the_boundary
```

It fails identically on the base commit (`c833dfa`) — a PLE tracker issue,
unrelated to this branch, do not chase it here. Reference counts from the
source box: 115 passed, 53 skipped (CUDA-gated cases), 1 failed.

## Tier 2 — checkpoint-gated tests (weights, CPU ok)
```bash
pytest tests/models/qwen4_exp/test_mtp_ckpt.py -q          # new: MTP mapping
pytest tests/models/qwen4_exp/test_weight_ckpt.py -q       # pre-existing: full loader (heavy)
```

`test_mtp_ckpt` replays the production loader remap over the real MTP shards
and strict-loads them into the draft head. Passing means:

- all 31 checkpoint `mtp.*` tensors map onto the module, key-for-key
  (including the qkv fusion and the hyper-connection down+inject fusion),
- every tensor is bf16 and finite,
- the fc_embedding/fc_hidden neck runs finite on real weights.

`test_weight_ckpt` exercises the whole loader including the 47.7 GiB PLE pin
(needs all `model-plefp8-*` shards).

## Tier 3 — GPU suites + live serve (weights, GPU free)

```bash
pytest tests/models/qwen4_exp -q      # CUDA cases un-skip: QSA kernels/backend, GDN
```

Expected: Tier 1's failure list again (the PLE test), nothing new.

Then the acceptance test — speculative decode end to end:

```bash
ft serve --model $FREETOKEN_QWEN4EXP_MODEL --spec-mtp
```

- Startup must load the MTP head cleanly (mapping drift fails loudly in
  `load_state_dict` here).
- Send a **greedy** request (`temperature: 0` — the spec arm only engages on
  greedy; request shape in `docs/cli.md`) and check the output is coherent.
- A/B decode speed: rerun without `--spec-mtp` on a long-generation prompt.
  Expect a solid uplift on warm caches (coding-style repeated prefixes are
  the friendly case), muted on the very first request (expert banks cold).
- Watch host RAM during startup: PLE pin + expert banks must fit the pin
  budget or launch fails sizing it.

## Failure triage

| Symptom | Meaning |
|---|---|
| Tier 2 strict-load `Unexpected keys` / `missing` | Checkpoint `config.json` drifted vs the remap gate (`_checkpoint_mtp_layers` in `models/qwen4_exp/weight.py`) |
| `cos_sin_cache must be a CUDA tensor` | A device-blind rope cache was reintroduced (`layers/rotary.py` — the cache is keyed on build device) |
| Serve startup OOM sizing the pin budget | Host RAM < PLE (47.7 GiB) + banks; free page cache or add RAM |
| Slow first tokens, fast after | Normal: expert banks warm into the GPU cache; not an MTP bug |
