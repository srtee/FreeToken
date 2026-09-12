#!/bin/bash
# Chain all soak jobs on the GPU, one at a time (dependency chain).
set -e
cd "$(dirname "$0")"
J() { sbatch --gres=gpu:1 --cpus-per-task=4 --mem=24g -J "$1" soak-batch.sh "${@:2}"; }
LAST=$(J soak-f16 f16 none | grep -oE '[0-9]+')
LAST=$(sbatch --dependency=afterok:$LAST -J soak-turbo8 soak-batch.sh turbo8 none | grep -oE '[0-9]+')
LAST=$(sbatch --dependency=afterok:$LAST -J soak-turbo4 soak-batch.sh turbo4 none | grep -oE '[0-9]+')
LAST=$(sbatch --dependency=afterok:$LAST -J soak-turbo3 soak-batch.sh turbo3_tcq none | grep -oE '[0-9]+')
sbatch --dependency=afterok:$LAST -J soak30-turbo3 soak-batch.sh turbo3_tcq innerq --soak
echo "queued: f16 -> turbo8 -> turbo4 -> turbo3_tcq -> soak30"
squeue