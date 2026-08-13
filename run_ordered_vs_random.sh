#!/bin/bash
#
# Clean (no noise, no reverberation) LibriSpeech simulation, run twice: once with
# the utterances of each speaker following the original LibriVox reading order and
# once with the previous random sampling, so the two sets differ only in that.
# Both reuse the same source cuts (manifests/all_cuts.jsonl.gz, symlinked).
#
# Runs locally with N_JOBS workers (~7 GB each), one config after the other.

set -euo pipefail

cd /mnt/proj1/fta-26-11/ipoloka/FastMSS
source local_env.sh

N_JOBS="${N_JOBS:-8}"
mkdir -p logs

for VARIANT in ordered random; do
    SIM_DIR="${SIM_OUT_ROOT}/librispeech/clean_${VARIANT}"
    LOG="logs/clean_${VARIANT}.log"
    echo "=== ${VARIANT}: simulating -> ${SIM_DIR} (log: ${LOG})"

    nice -n 10 python sim.py \
        --config-path="config/table2/${VARIANT}" --config-name librispeech \
        n_jobs="${N_JOBS}" >> "${LOG}" 2>&1

    echo "=== ${VARIANT}: pre-segmenting to 30s windows"
    nice -n 10 python "${MADP_ROOT}/src/pre_segment_using_alignments.py" \
        --input "${SIM_DIR}/manifests/synth-librispeech-train-cuts.jsonl.gz" \
        --output "${SIM_DIR}/manifests/synth-librispeech-train-cuts_30s.jsonl.gz" \
        --max_len 30 \
        --num_jobs "${N_JOBS}" >> "${LOG}" 2>&1

    echo "=== ${VARIANT}: done"
done
