#!/bin/bash
#SBATCH --job-name fastmss-sim
#SBATCH --account DD-26-5
#SBATCH --partition qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --time 4:00:00
#SBATCH --output=logs/%x_%j.out
#
# Run one FastMSS simulation config on a single compute node and pre-segment the
# resulting cutset to 30s windows for training.
#
# Usage:
#   sbatch submit_sim.sh <config_path> <config_name> <sim_dir> <cutset_prefix> [hydra overrides...]
#
# Example:
#   sbatch submit_sim.sh config/table2/ordered librispeech \
#       librispeech/callhome_boosted_w_nr_ordered synth-librispeech-train-cuts

set -euo pipefail

CONFIG_PATH="$1"
CONFIG_NAME="$2"
SIM_DIR="$3"
CUTSET_PREFIX="$4"
shift 4

cd /mnt/proj1/fta-26-11/ipoloka/FastMSS
source local_env.sh

ml Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ts_asr

python sim.py --config-path="$CONFIG_PATH" --config-name "$CONFIG_NAME" "$@"

MANIFESTS="${SIM_OUT_ROOT}/${SIM_DIR}/manifests"
python "${MADP_ROOT}/src/pre_segment_using_alignments.py" \
    --input "${MANIFESTS}/${CUTSET_PREFIX}.jsonl.gz" \
    --output "${MANIFESTS}/${CUTSET_PREFIX}_30s.jsonl.gz" \
    --max_len 30 \
    --num_jobs 32
