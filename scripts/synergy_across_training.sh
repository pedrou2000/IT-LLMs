#!/bin/bash
#SBATCH --job-name=phiid
#SBATCH --output=logs/phiid-%j.out
#SBATCH --error=logs/phiid-%j.out
#SBATCH --partition=h100-agentS-train
#SBATCH --gres=gpu:1
# h100-agentS-train or cpu

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate int
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: $(pwd)"

python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/synergy_across_training.py
python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/plot_synergy_across_training.py
