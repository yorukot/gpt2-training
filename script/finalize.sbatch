#!/usr/bin/env bash
#SBATCH --job-name=finalize-yoru
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn146,25a-hgpn157
#SBATCH --output=finalize-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch script/finalize.sbatch <run name, e.g. lab4-yoru-AX-490841>
# train.sbatch queues this itself behind each training job.
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src

exec .venv-gpu/bin/python script/finalize.py "${1:?pass a run name, e.g. lab4-yoru-AX-490841}"
