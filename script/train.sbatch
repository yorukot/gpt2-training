#!/usr/bin/env bash
#SBATCH --job-name=lab4-yoru
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch script/train.sbatch <run ID: AA, AB, AC, ...>
export RUN_NAME="lab4-yoru-${1:?pass the run ID, e.g. sbatch script/train.sbatch AA}-$SLURM_JOB_ID" && echo "Run name: $RUN_NAME"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
# Keep torch.compile's cache in the project, not node-local /tmp, so later jobs skip most compiling.
export TORCHINDUCTOR_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/inductor"

exec .venv-gpu/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 src/train_gpt2.py
