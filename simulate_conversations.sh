#!/bin/bash

source local_env.sh
# Table 1
SIMULATED_OUT="${SIM_OUT_ROOT}/nsf"
CONFIGS=("flat" "ami_estimate" "nsf_estimate" "nsf_estimate_boosted")

for CONFIG in "${CONFIGS[@]}"; do
    echo "Processing config: $CONFIG"
    python "${FastMSS_ROOT}/sim_nsf_ihm.py" --config-name "$CONFIG"
    python "${MADP_ROOT}/src/pre_segment_using_alignments.py" \
        --input "$SIMULATED_OUT/${CONFIG}/manifests/synth-notsofar1_ihm-train-cuts.jsonl.gz" \
        --output "$SIMULATED_OUT/${CONFIG}/manifests/synth-notsofar1_ihm-train-cuts_30s.jsonl.gz" \
        --max_len 30
done