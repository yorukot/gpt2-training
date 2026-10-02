#!/usr/bin/env bash
#SBATCH --job-name=teacher-yoru
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157
#SBATCH --output=teacher-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch script/eval_teacher.sbatch <HF model id, e.g. EleutherAI/gpt-j-6b>
# The model must already be in the HF cache (hf download on the login node).
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src

exec .venv-gpu/bin/python script/eval_teacher.py "${1:?pass a teacher model id, e.g. EleutherAI/gpt-j-6b}"
