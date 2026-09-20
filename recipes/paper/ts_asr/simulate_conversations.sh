#!/bin/bash
# Sweep all TS-ASR paper configs (Table 1 + Table 2) through recipes/sim.py.
#
# Requires the env vars referenced inside the YAMLs:
#   SIM_OUT_ROOT  - output root for generated meetings
#   MANIFEST_DIR  - root containing per-corpus lhotse manifests
#   MUSAN_ROOT    - only for table2/w_noise and table2/w_nr
#
# Extra args are forwarded to sim.py, e.g.:
#   ./simulate_conversations.sh n_jobs=16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SIM_PY="${REPO_ROOT}/recipes/sim.py"

TABLE1_CONFIGS=(
    flat
    callhome
    callhome_boosted
    ami_estimate
    nsf_estimate
    nsf_estimate_markov
    nsf_estimate_boosted
)

for CFG in "${TABLE1_CONFIGS[@]}"; do
    echo "=== Table 1: ${CFG} ==="
    python "${SIM_PY}" --config-name "paper/ts_asr/table1/${CFG}" "$@"
done

TABLE2_VARIANTS=(clean w_reverb w_nr w_noise)

for VARIANT in "${TABLE2_VARIANTS[@]}"; do
    VARIANT_DIR="${SCRIPT_DIR}/table2/${VARIANT}"
    for YAML in "${VARIANT_DIR}"/*.yaml; do
        CFG="$(basename "${YAML}" .yaml)"
        echo "=== Table 2 / ${VARIANT}: ${CFG} ==="
        python "${SIM_PY}" --config-name "paper/ts_asr/table2/${VARIANT}/${CFG}" "$@"
    done
done
