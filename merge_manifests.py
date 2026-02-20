"""
Merge per-job manifests from job array runs into a single manifest set.

Usage:
    python merge_manifests.py <manifest_dir> [--prefix synth]
"""
import argparse
import glob
import logging
from pathlib import Path

import lhotse
from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.manipulation import combine as combine_manifests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge per-job FastMSS manifests")
    parser.add_argument("manifest_dir", type=str, help="Directory containing per-job manifests")
    parser.add_argument("--prefix", type=str, default="synth", help="Output prefix for merged manifests")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)

    # Find all per-job recording manifests
    rec_files = sorted(glob.glob(str(manifest_dir / "*part_*-recordings.jsonl.gz")))
    sup_files = sorted(glob.glob(str(manifest_dir / "*part_*-supervisions.jsonl.gz")))
    cut_files = sorted(glob.glob(str(manifest_dir / "*part_*-cuts.jsonl.gz")))

    if not rec_files:
        logger.error(f"No per-job recording manifests found in {manifest_dir}")
        return

    logger.info(f"Found {len(rec_files)} recording manifests, {len(sup_files)} supervision manifests, {len(cut_files)} cut manifests")

    # Merge recordings
    all_recs = combine_manifests([lhotse.load_manifest(f) for f in rec_files])
    out = manifest_dir / f"{args.prefix}-train-recordings.jsonl.gz"
    all_recs.to_file(out)
    logger.info(f"Merged {len(all_recs)} recordings -> {out}")

    # Merge supervisions
    all_sups = combine_manifests([lhotse.load_manifest(f) for f in sup_files])
    out = manifest_dir / f"{args.prefix}-train-supervisions.jsonl.gz"
    all_sups.to_file(out)
    logger.info(f"Merged {len(all_sups)} supervisions -> {out}")

    # Merge cuts if available
    if cut_files:
        all_cuts = combine_manifests([lhotse.load_manifest(f) for f in cut_files])
        out = manifest_dir / f"{args.prefix}-train-cuts.jsonl.gz"
        all_cuts.to_file(out)
        logger.info(f"Merged {len(all_cuts)} cuts -> {out}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
