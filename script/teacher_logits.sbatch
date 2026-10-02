#!/usr/bin/env bash
#SBATCH --job-name=logits-yoru
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157
#SBATCH --output=logits-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch script/teacher_logits.sbatch
# Resumable: chunks that already have a done marker are skipped, so resubmitting continues.
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src

exec .venv-gpu/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 script/teacher_logits.py
