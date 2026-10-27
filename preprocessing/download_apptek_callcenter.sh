#!/bin/bash
# Download AppTek Call-Center Dialogues from HuggingFace and resample to 16 kHz.
#
# Layout produced (mirroring preprocessing/download_wham.sh):
#   ${INSTALL_DIR}_original_SR/  - original-SR audio + metadata as downloaded
#   ${INSTALL_DIR}/              - 16 kHz audio + metadata, ready for pseudo-labeling
#
# Usage: bash download_apptek_callcenter.sh [INSTALL_DIR]
#   INSTALL_DIR  Target dir for 16 kHz audio (default: /home/samco/Datasets/ApptekConv)
#   NUM_JOBS     Env var; parallel workers for resampling (default: nproc)
set -euo pipefail

INSTALL_DIR="${1:-/home/samco/Datasets/ApptekConv}"
REPO="apptek-com/apptek_callcenter_dialogues"
TARGET_SR=16000
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
ORIG_DIR="${INSTALL_DIR}_original_SR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preflight checks
for cmd in hf python3; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "Error: ${cmd} not found. Please install it before running this script."
        exit 1
    fi
done

echo "=== AppTek Call-Center Dialogues Download ==="
echo "Original SR dir : ${ORIG_DIR}"
echo "16 kHz dir      : ${INSTALL_DIR}"
echo "Parallel jobs   : ${NUM_JOBS}"
echo ""

mkdir -p "${ORIG_DIR}" "${INSTALL_DIR}"

echo "[1/3] Downloading ${REPO} from HuggingFace..."
hf download "${REPO}" --repo-type dataset --local-dir "${ORIG_DIR}"

WAV_COUNT=$(find "${ORIG_DIR}" -name "*_channel?.wav" | wc -l | tr -d ' ')
echo "Downloaded ${WAV_COUNT} channel files."

echo ""
echo "[2/3] Resampling audio to ${TARGET_SR} Hz (preserving accent/audio/ structure)..."
python3 "${SCRIPT_DIR}/resample_folder.py" "${ORIG_DIR}" "${INSTALL_DIR}" \
    --target_sr "${TARGET_SR}" --num_workers "${NUM_JOBS}" \
    --extensions .wav

echo ""
echo "[3/3] Copying metadata (test.jsonl) alongside resampled audio..."
while IFS= read -r -d '' src; do
    rel="${src#${ORIG_DIR}/}"
    dst="${INSTALL_DIR}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    cp "${src}" "${dst}"
done < <(find "${ORIG_DIR}" -name "*.jsonl" -print0)

CH1=$(find "${INSTALL_DIR}" -name "*_channel1.wav" | wc -l | tr -d ' ')
CH2=$(find "${INSTALL_DIR}" -name "*_channel2.wav" | wc -l | tr -d ' ')

echo ""
echo "=== Done ==="
echo "✓ 16 kHz audio       : ${INSTALL_DIR}/"
echo "✓ Original SR audio  : ${ORIG_DIR}/"
echo "  channel1 wavs: ${CH1}"
echo "  channel2 wavs: ${CH2}"
