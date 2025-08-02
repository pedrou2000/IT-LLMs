#!/bin/bash

# Manually hardcoded values (safe to edit before submission)
MODEL="L32-1"                               # L32-1, D2-16-A2, P-1
GENERATION="original_prompts"               # subset, original_prompts
TIME_SERIES="attention_outputs"             # attention_outputs, expert_output
PHYID="base"                                # discrete
DEACTIVATION_ANALYSIS="reverse_kl"

# Generate a unique job script at submission time
TIMESTAMP=$(date +%s)
JOB_NAME="${MODEL}_${GENERATION}_${TIME_SERIES}_${PHYID}"
JOB_SCRIPT="scripts/run_${JOB_NAME}.sh"
LOG_FILE="logs/${JOB_NAME}_%j.out"

mkdir -p slurm_jobs logs

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=phiid-${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate int
export HF_ALLOW_CODE_EVAL=1

echo "Working directory: \$(pwd)"
echo "Running with config: model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS"

python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/record_activations.py model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS
python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/time_series_and_phyid.py model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS
python /home/p84400019/projects/consciousness-llms/IT-LLMs/scripts/ranked_deactivations.py model=$MODEL generation=$GENERATION time_series=$TIME_SERIES phyid=$PHYID deactivation_analysis=$DEACTIVATION_ANALYSIS
EOF

chmod +x "$JOB_SCRIPT"
sbatch "$JOB_SCRIPT"
rm "$JOB_SCRIPT"
