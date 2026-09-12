# TCQ wave-3 soak — Slurm job plan (pending GPU undrain)

Workstation Slurm: single node `MS7E06`, partition `local*`, configured
8 CPUs / 30 GB / `Gres=gpu:1`. Node currently DRAINED
("gres/gpu count reported lower than configured (0 < 1)") — a cloud agent
(omp/glm-5.3, agent id `c0af0d81-67fc-45c4-a296-3c317014fc57`) is fixing the
gres config. Once `scontrol show node MS7E06` shows IDLE with gpu count 1,
submit the jobs below.

## What these jobs are

- **Codec battery** (`battery.sh`): per KV codec (f16 baseline, turbo8,
  turbo4, turbo3_tcq), serve Qwen2.5-Coder-32B IQ3_XXS (head_dim 128, 12 GiB
  weights — fits the 16 GiB card; the plan's Qwen3.6 pick is impossible for
  the soak because the turbo pool hard-requires head_dim == 128 and the 35B
  has 256), run timed 2K and 8K-context prefill + 128-token decode rounds,
  record VRAM and wall time. Fills the pp/tg columns of
  `docs/tcq-baseline-numbers.md`. Each serve is sequential (one GPU).
- **30-min soak** (`soak30.sh`): turbo3_tcq server driven by alternating
  codegen prompts at temperature 0 for 1800 s; every output is length+hash
  logged so drift/loop detection is a diff of the sample file. This is the
  plan's "quants will misbehave" gate.

LAMMPS note: the `lmp` run (13 workers) lives outside Slurm; jobs request
`--cpus-per-task=4` so Slurm never overbooks the box, and the ft server
itself is GPU-bound/CPU-light. The soak server binds port 1920.

## Job scripts

`sbatch --gres=gpu:1 --cpus-per-task=4 --mem=24g -J soak-<codec> soak-batch.sh <codec> <tune>`

`soak-batch.sh` wraps `battery.sh <codec> <tune>`; the 30-min soak runs as
`soak-batch.sh turbo3_tcq innerq --soak`. Jobs are chained with `--dependency`
(singleton on job name is fine too) so the GPU serves one codec at a time:
f16 → turbo8 → turbo4 → turbo3_tcq → soak30. Output lands in
`FreeToken/soak/results/<codec>-<tune>.txt` (battery lines
`codec=... ctx=... wall_s=...` + `vram_mib=...`).

## After completion

- Parse results into the final `docs/tcq-baseline-numbers.md` table
  (VRAM/token, pp/tg t/s at 2K/8K per codec).
- Check soak30 sample hashes: any repeat-hash across turns on the same
  prompt, or monotonic length decay, is the drift signal the plan warns
  about (buun caveat).

## Submit checklist (once undrained)

```
sinfo                          # MS7E06 state=idle, gres gpu=1
sbatch --gres=gpu:1 --wrap='nvidia-smi -L'   # smoke: job runs, sees GPU
bash FreeToken/soak/submit_all.sh             # chains the 5 jobs
```