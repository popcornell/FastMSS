#!/bin/bash
#
# SLURM job array wrapper for FastMSS simulation (stage 4 only).
# Splits n_meetings across N_ARRAY_JOBS independent jobs, each with a unique seed.
#
# Prerequisites: stages 0-3 must be completed already (cuts, noise, RIRs prepared).
#
# Usage:
#   ./run_jobarray.sh <sim_script> <n_total_meetings> <n_array_jobs> [extra hydra overrides...]
#
# Examples:
#   ./run_jobarray.sh sim_ami_ihm.py 5000 50
#   ./run_jobarray.sh sim_librispeech.py 5000 50 duration=180 min_max_spk=[3,6]
#   ./run_jobarray.sh sim_oto_ihm.py 1000 10 add_noise=False

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <sim_script> <n_total_meetings> <n_array_jobs> [extra hydra overrides...]"
    echo ""
    echo "  sim_script       : Python simulation script (e.g. sim_ami_ihm.py)"
    echo "  n_total_meetings : Total number of meetings to generate"
    echo "  n_array_jobs     : Number of SLURM array jobs to split across"
    echo "  extra overrides  : Additional Hydra overrides passed to the sim script"
    exit 1
fi

SIM_SCRIPT="$1"
N_TOTAL="$2"
N_JOBS="$3"
shift 3
EXTRA_OVERRIDES="$@"

# Compute meetings per job (last job handles remainder)
MEETINGS_PER_JOB=$(( N_TOTAL / N_JOBS ))
REMAINDER=$(( N_TOTAL % N_JOBS ))

ARRAY_MAX=$(( N_JOBS - 1 ))

# Create a temporary sbatch script
SBATCH_SCRIPT=$(mktemp /tmp/fastmss_jobarray_XXXXXX.sh)

cat > "$SBATCH_SCRIPT" << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=fastmss-sim
#SBATCH --output=logs/fastmss_%A_%a.out
#SBATCH --error=logs/fastmss_%A_%a.err
HEREDOC

# Append the array directive
echo "#SBATCH --array=0-${ARRAY_MAX}" >> "$SBATCH_SCRIPT"

cat >> "$SBATCH_SCRIPT" << HEREDOC

set -euo pipefail

JOB_IDX=\$SLURM_ARRAY_TASK_ID
MEETINGS_PER_JOB=${MEETINGS_PER_JOB}
REMAINDER=${REMAINDER}
N_TOTAL=${N_TOTAL}

# Last job takes the remainder
if [ "\$JOB_IDX" -eq "${ARRAY_MAX}" ]; then
    N_MEETINGS=\$(( MEETINGS_PER_JOB + REMAINDER ))
else
    N_MEETINGS=\$MEETINGS_PER_JOB
fi

# Unique seed per job: base seed from SLURM array job ID + task index
SEED=\$(( SLURM_ARRAY_JOB_ID + JOB_IDX ))

# Set numpy/python random seed before launching
export PYTHONHASHSEED=\$SEED

python ${SIM_SCRIPT} \\
    stage=4 \\
    n_meetings=\$N_MEETINGS \\
    n_jobs=1 \\
    seed=\$SEED \\
    manifest_prefix=part_\${JOB_IDX} \\
    ${EXTRA_OVERRIDES}
HEREDOC

mkdir -p logs
echo "Submitting job array: ${N_JOBS} jobs, ${MEETINGS_PER_JOB} meetings/job (last job: $(( MEETINGS_PER_JOB + REMAINDER )))"
echo "Script: ${SIM_SCRIPT}"
echo "Total meetings: ${N_TOTAL}"
echo "Sbatch script: ${SBATCH_SCRIPT}"
echo ""
cat "$SBATCH_SCRIPT"
echo ""
echo "---"
echo "To submit: sbatch $SBATCH_SCRIPT"
echo "To merge manifests after completion, run:"
echo "  python merge_manifests.py <output_dir>/manifests"
