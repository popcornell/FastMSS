"""On-disk layout shared by the TTS recipes.

Every TTS corpus (tts_sim.py, tts_staged.py, and the corpora derived from one by
derive_shuffled_turns.py / augment_tts_acoustics.py) has the same shape:

    <out>/
      bank.json                              LibriSpeech reference-voice bank
      audio/<id>.wav                         mixture
      audio/<id>-spk-<speaker>.wav           per-speaker stem (when written)
      manifests/<id>-{recordings,supervisions,cuts}.jsonl.gz
      manifests/<id>.rttm
      manifests/tts-{recordings,supervisions,cuts}.jsonl.gz   all meetings
      manifests/all.rttm

Per-meeting manifests are written as soon as a meeting finishes, so a killed run
never leaves audio without labels; the aggregate files are rebuilt from them at
the end. This module holds the helpers that write that layout.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import lhotse
from lhotse import RecordingSet, SupervisionSet

AGGREGATE_RECORDINGS = "tts-recordings.jsonl.gz"
AGGREGATE_SUPERVISIONS = "tts-supervisions.jsonl.gz"
AGGREGATE_CUTS = "tts-cuts.jsonl.gz"
AGGREGATE_RTTM = "all.rttm"


def default_cfg(**over) -> SimpleNamespace:
    """FastMSS mixing/RIR config used by the TTS recipes; keyword args override."""
    cfg = dict(samplerate=16000, base_gain=[-25.0, -12.0], rel_gain=[-6.0, 3.0],
               use_fir=True, reverberate=False, reverb_prob=1.0,
               add_noise=False, noise_rel_gain=[-20, -3],
               noise_probability_global=1.0, multi_noise_sampling=True,
               save_spk=True, save_anechoic=False,
               rt60=[0.2, 0.6], room_sz=[5, 8], room_ceiling=[2.7, 3.5],
               n_rirs=2, n_pos_rirs=6, delta_dist=0.5, use_rand_ism=False,
               rand_disp=0.0, max_position_change=0.5, mic_type="single",
               hmm_fit_transitions_to=None, boost_overlap_factor=None,
               use_markov=False, min_max_spk=[3, 4], duration=30)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def fmt_seconds(seconds: float) -> str:
    """Human-readable duration: 12.3s, 4m 5.0s, 1h 2m 3.0s."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def stable_seed(*parts: object) -> int:
    """64-bit seed derived from its arguments; identical across runs and machines."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def write_json_atomic(path: Path, data: object) -> None:
    """Write JSON through a temp file so readers never see a half-written file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def write_rttm(supervisions: Iterable, path: Path) -> None:
    with open(path, "w") as f:
        for s in supervisions:
            f.write(f"SPEAKER {s.recording_id} 1 {s.start:.3f} {s.duration:.3f} "
                    f"<NA> <NA> {s.speaker} <NA> <NA>\n")


def write_meeting_manifests(manifest_dir: Path, recording, supervisions: list) -> None:
    """Validate one meeting and write its recordings/supervisions/cuts + RTTM."""
    manifest_dir, mid = Path(manifest_dir), recording.id
    recs, sups = RecordingSet([recording]), SupervisionSet(supervisions)
    lhotse.validate_recordings_and_supervisions(recs, sups)
    recs.to_file(manifest_dir / f"{mid}-recordings.jsonl.gz")
    sups.to_file(manifest_dir / f"{mid}-supervisions.jsonl.gz")
    lhotse.CutSet.from_manifests(recordings=recs, supervisions=sups) \
        .to_file(manifest_dir / f"{mid}-cuts.jsonl.gz")
    write_rttm(supervisions, manifest_dir / f"{mid}.rttm")


def write_combined_manifests(manifest_dir: Path, recordings: list, supervisions: list) -> None:
    """Validate and write the corpus-level tts-* manifests and all.rttm."""
    manifest_dir = Path(manifest_dir)
    recs, sups = RecordingSet(recordings), SupervisionSet(supervisions)
    lhotse.validate_recordings_and_supervisions(recs, sups)
    recs.to_file(manifest_dir / AGGREGATE_RECORDINGS)
    sups.to_file(manifest_dir / AGGREGATE_SUPERVISIONS)
    lhotse.CutSet.from_manifests(recordings=recs, supervisions=sups) \
        .to_file(manifest_dir / AGGREGATE_CUTS)
    write_rttm(supervisions, manifest_dir / AGGREGATE_RTTM)


def combine_meeting_manifests(manifest_dir: Path, recording_ids: Iterable[str]) -> None:
    """Rebuild the aggregate manifests from already-written per-meeting files."""
    manifest_dir = Path(manifest_dir)
    recordings, supervisions = [], []
    for mid in recording_ids:
        recordings.extend(RecordingSet.from_file(manifest_dir / f"{mid}-recordings.jsonl.gz"))
        supervisions.extend(SupervisionSet.from_file(manifest_dir / f"{mid}-supervisions.jsonl.gz"))
    write_combined_manifests(manifest_dir, recordings, supervisions)
