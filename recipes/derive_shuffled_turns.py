#!/usr/bin/env python3
"""Derive a discourse-shuffled FastMSS TTS corpus without rerunning TTS.

Each output slot retains its original meeting, clone voice, event type, and
measured speech level. Its waveform and transcript are replaced with a
different, duration-compatible turn from the same clone voice and event class.
By default its onset is reflowed to preserve source pause and overlap timing.
The donor mapping is a deterministic derangement, so every selected source turn
is used exactly once and no turn is assigned to itself.

The source must be a completed dry ``tts_staged.py`` corpus: the script uses
``manifests/*_realized.json`` and the corresponding ``clips_final`` WAVs.
It writes mixed audio and Lhotse/RTTM manifests only by default. Use
``--write-stems`` when creating a corpus that will later need RIR/noise
augmentation, or ``--write-stems-only`` to recover stems for an existing
shuffled corpus without changing its mixtures.

Matching rules, in priority order:
  1. required:  same clone voice (LibriSpeech id) and same event type
                (switch / hold / interrupt / backchannel);
  2. preferred: same delivery level (relaxed only for singleton level buckets);
  3. preferred: similar duration (derangements are drawn inside blocks of
                --duration-block-size turns sorted by duration);
  4. avoided:   donors from the same meeting (repaired by swaps when possible).

Outputs, besides the usual corpus layout (see fastmss/tts/corpus.py):
  manifests/shuffled-turn.json              configuration + mapping statistics
  manifests/shuffled-turn-mapping.jsonl.gz  one row per target turn -> donor turn
  manifests/<id>_realized.json              records with `shuffle` / `timing` provenance
Check a finished derivation with recipes/validate_shuffled_turns.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lhotse
import numpy as np
import soundfile as sf
from lhotse import AudioSource, Recording, RecordingSet, SupervisionSet
from lhotse.supervision import SupervisionSegment
from scipy.signal import convolve, firwin, resample_poly

from fastmss.tts.corpus import (combine_meeting_manifests, fmt_seconds, stable_seed,
                                write_json_atomic, write_meeting_manifests)


# Must match the source mixer (fastmss/tts/simulator.py): 16 kHz output and the
# same 63-tap, 40 Hz FIR high-pass, so donor turns are processed identically.
SR = 16000
FIR_TAPS = 63
FIR_CUTOFF_HZ = 40.0
SCHEMA_VERSION = 2       # of shuffled-turn.json / *_realized.json


@dataclass(frozen=True)
class Turn:
    """One realized source turn and the target slot it initially occupied."""

    recording_id: str
    uid: int
    voice_id: str
    event: str
    level: str
    duration: float
    clip_path: Path
    clip_relpath: str
    record: dict[str, Any]
    speech_level_db: float

    @property
    def key(self) -> tuple[str, int]:
        return self.recording_id, self.uid


@dataclass
class Meeting:
    """A source meeting: its realized plan and its turns sorted by uid."""
    recording_id: str
    source_plan: dict[str, Any]
    turns: list[Turn]


# --------------------------------------------------------------------------
# Loading the source corpus
# --------------------------------------------------------------------------

def _meeting_sort_key(recording_id: str) -> tuple[int, str]:
    """Sort ttsmeeting_<n> numerically (so _1000 comes after _999)."""
    prefix = "ttsmeeting_"
    if recording_id.startswith(prefix):
        suffix = recording_id[len(prefix):]
        if suffix.isdigit():
            return int(suffix), recording_id
    return math.inf, recording_id


def _required_source_paths(source: Path, recording_id: str) -> tuple[Path, Path]:
    manifest_dir = source / "manifests"
    return (
        manifest_dir / f"{recording_id}_realized.json",
        manifest_dir / f"{recording_id}-supervisions.jsonl.gz",
    )


def _read_selected_ids(path: Path) -> set[str]:
    """Recording ids from a text file, one per line; '#' lines are comments."""
    ids = {line.strip() for line in path.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")}
    if not ids:
        raise ValueError(f"{path} does not contain any recording IDs")
    return ids


def _supervision_levels(recording_id: str, records: list[dict[str, Any]], path: Path) -> dict[int, float]:
    """Read measured target levels, validating the staged-plan ordering contract."""
    supervisions = list(SupervisionSet.from_file(path))
    ordered = sorted(records, key=lambda record: int(record["uid"]))
    if len(supervisions) != len(ordered):
        raise RuntimeError(
            f"{recording_id}: {len(records)} realized records but {len(supervisions)} supervisions"
        )
    levels: dict[int, float] = {}
    for index, supervision in enumerate(sorted(supervisions, key=lambda item: item.id)):
        expected_id = f"{recording_id}_{index:04d}"
        if supervision.id != expected_id:
            raise RuntimeError(
                f"{recording_id}: expected supervision {expected_id}, found {supervision.id}; "
                "cannot safely recover per-turn measured levels"
            )
        value = (supervision.custom or {}).get("speech_level_db")
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"{recording_id}: {supervision.id} has no numeric speech_level_db"
            )
        levels[int(ordered[index]["uid"])] = float(value)
    return levels


def _resolve_clip(source: Path, raw_path: object, recording_id: str, uid: int) -> tuple[Path, str]:
    """Absolute path of a final turn clip, plus its path relative to the source corpus."""
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{recording_id}/uid{uid}: realized record has no clip path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = source / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{recording_id}/uid{uid}: missing final turn clip {path}")
    try:
        relative = str(path.relative_to(source))
    except ValueError:
        relative = str(path)
    return path, relative


def _load_meetings(source: Path, limit: int | None, recording_ids: set[str] | None) -> list[Meeting]:
    """Load and validate the selected source meetings (fails on any missing clip/label)."""
    manifest_dir = source / "manifests"
    aggregate = manifest_dir / "tts-recordings.jsonl.gz"
    if not aggregate.is_file():
        raise FileNotFoundError(f"Missing aggregate recording manifest: {aggregate}")

    plan_paths = sorted(manifest_dir.glob("ttsmeeting_*_realized.json"),
                        key=lambda path: _meeting_sort_key(path.name.removesuffix("_realized.json")))
    if not plan_paths:
        raise FileNotFoundError(f"No *_realized.json files under {manifest_dir}")
    plans: list[tuple[str, Path]] = []
    for plan_path in plan_paths:
        recording_id = plan_path.name.removesuffix("_realized.json")
        if recording_ids is not None and recording_id not in recording_ids:
            continue
        plans.append((recording_id, plan_path))
    if recording_ids is not None:
        found = {recording_id for recording_id, _ in plans}
        missing = sorted(recording_ids - found)
        if missing:
            raise FileNotFoundError(f"No realized plan for selected recording {missing[0]}")
    if limit is not None:
        plans = plans[:limit]
    if not plans:
        raise RuntimeError("No source meetings selected")

    meetings: list[Meeting] = []
    for recording_id, plan_path in plans:
        plan = json.load(open(plan_path))
        if plan.get("id") != recording_id:
            raise RuntimeError(f"{plan_path}: plan id {plan.get('id')!r} does not match its filename")
        records = plan.get("records")
        personas = plan.get("personas")
        if not isinstance(records, list) or not records or not isinstance(personas, dict):
            raise RuntimeError(f"{recording_id}: incomplete realized plan")
        _, supervision_path = _required_source_paths(source, recording_id)
        if not supervision_path.is_file():
            raise FileNotFoundError(f"{recording_id}: missing supervision manifest {supervision_path}")
        levels = _supervision_levels(recording_id, records, supervision_path)
        seen_uids: set[int] = set()
        turns: list[Turn] = []
        for record in records:
            uid = int(record["uid"])
            if uid in seen_uids:
                raise RuntimeError(f"{recording_id}: duplicate realized uid {uid}")
            seen_uids.add(uid)
            speaker = record.get("speaker")
            persona = personas.get(speaker)
            if not isinstance(persona, dict) or not persona.get("ls_id"):
                raise RuntimeError(f"{recording_id}/uid{uid}: speaker {speaker!r} has no LibriSpeech voice ID")
            event, level = record.get("event"), record.get("level")
            if not isinstance(event, str) or not isinstance(level, str):
                raise RuntimeError(f"{recording_id}/uid{uid}: missing event or level")
            duration = float(record["dur"])
            if duration <= 0:
                raise RuntimeError(f"{recording_id}/uid{uid}: non-positive duration {duration}")
            clip_path, clip_relpath = _resolve_clip(source, record.get("path"), recording_id, uid)
            turns.append(Turn(
                recording_id=recording_id,
                uid=uid,
                voice_id=str(persona["ls_id"]),
                event=event,
                level=level,
                duration=duration,
                clip_path=clip_path,
                clip_relpath=clip_relpath,
                record=record,
                speech_level_db=levels[uid],
            ))
        meetings.append(Meeting(recording_id=recording_id, source_plan=plan,
                                turns=sorted(turns, key=lambda turn: turn.uid)))
    return meetings


# --------------------------------------------------------------------------
# Building the donor mapping
# --------------------------------------------------------------------------

def _derangement(size: int, rng: random.Random) -> list[int]:
    """Random permutation of range(size) with no fixed points (no turn maps to itself)."""
    if size < 2:
        raise ValueError("a derangement requires at least two turns")
    permutation = list(range(size))
    rng.shuffle(permutation)
    fixed = [index for index, donor in enumerate(permutation) if index == donor]
    if len(fixed) == 1:
        index = fixed[0]
        other = 0 if index != 0 else 1
        permutation[index], permutation[other] = permutation[other], permutation[index]
    elif len(fixed) > 1:
        donors = [permutation[index] for index in fixed]
        for index, donor in zip(fixed, donors[1:] + donors[:1]):
            permutation[index] = donor
    if any(index == donor for index, donor in enumerate(permutation)):
        raise AssertionError("failed to construct a derangement")
    return permutation


def _duration_blocks(turns: list[Turn], block_size: int, seed: int, label: str) -> list[list[Turn]]:
    """Form local duration strata; the last singleton joins its predecessor."""
    ordered = sorted(turns, key=lambda turn: (
        turn.duration, stable_seed(seed, label, turn.recording_id, turn.uid)))
    blocks = [ordered[index:index + block_size] for index in range(0, len(ordered), block_size)]
    if len(blocks) > 1 and len(blocks[-1]) == 1:
        blocks[-2].extend(blocks.pop())
    if any(len(block) < 2 for block in blocks):
        raise AssertionError(f"{label}: singleton shuffle block")
    return blocks


def _repair_same_meeting(targets: list[Turn], donors: list[Turn], rng: random.Random) -> int:
    """Remove accidental within-meeting donors where a compatible swap exists."""
    for _ in range(len(targets)):
        bad = [index for index, target in enumerate(targets)
               if donors[index].recording_id == target.recording_id]
        if not bad:
            return 0
        changed = False
        for index in bad:
            candidates = [other for other in range(len(targets)) if other != index
                          and donors[other].recording_id != targets[index].recording_id
                          and donors[index].recording_id != targets[other].recording_id]
            rng.shuffle(candidates)
            if candidates:
                other = candidates[0]
                donors[index], donors[other] = donors[other], donors[index]
                changed = True
        if not changed:
            break
    return sum(donor.recording_id == target.recording_id
               for target, donor in zip(targets, donors))


def _partitions(turns: list[Turn]) -> tuple[list[tuple[str, list[Turn]]], int]:
    """Keep level exact when possible; only sparse singleton strata are relaxed."""
    by_voice_event: dict[tuple[str, str], list[Turn]] = defaultdict(list)
    for turn in turns:
        by_voice_event[turn.voice_id, turn.event].append(turn)

    result: list[tuple[str, list[Turn]]] = []
    level_relaxed = 0
    for (voice_id, event), group in sorted(by_voice_event.items()):
        by_level: dict[str, list[Turn]] = defaultdict(list)
        for turn in group:
            by_level[turn.level].append(turn)
        exact = [items for _, items in sorted(by_level.items()) if len(items) >= 2]
        singletons = [items[0] for _, items in sorted(by_level.items()) if len(items) == 1]
        if len(singletons) >= 2:
            result.append((f"{voice_id}:{event}:sparse-levels", singletons))
            level_relaxed += len(singletons)
        elif len(singletons) == 1:
            if not exact:
                raise RuntimeError(f"{voice_id}/{event}: cannot shuffle a single source turn")
            anchor = min(exact, key=len)
            anchor.append(singletons[0])
            level_relaxed += 1
        for items in exact:
            level = items[0].level if len({turn.level for turn in items}) == 1 else "mixed-levels"
            result.append((f"{voice_id}:{event}:{level}", items))
    return result, level_relaxed


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))]


def build_mapping(turns: list[Turn], seed: int, duration_block_size: int) -> tuple[dict[tuple[str, int], Turn], dict[str, Any]]:
    """Map every target turn key to a donor Turn; returns (mapping, statistics).

    Partition by (voice, event[, level]), derange inside duration blocks, then
    repair same-meeting donors first per block and again per partition. The
    invariants (one-to-one, no identity, voice/event preserved) are asserted.
    """
    mapping: dict[tuple[str, int], Turn] = {}
    partitions, sparse_level_turns = _partitions(turns)
    same_meeting_after_local_blocks = 0
    block_count = 0
    for label, partition in partitions:
        for block_index, block in enumerate(_duration_blocks(partition, duration_block_size, seed, label)):
            rng = random.Random(stable_seed(seed, label, block_index))
            donors = [block[index] for index in _derangement(len(block), rng)]
            same_meeting_after_local_blocks += _repair_same_meeting(block, donors, rng)
            block_count += 1
            for target, donor in zip(block, donors):
                if target.key in mapping:
                    raise AssertionError(f"duplicate shuffle target {target.key}")
                mapping[target.key] = donor

    source_keys = {turn.key for turn in turns}
    for label, partition in partitions:
        donors = [mapping[target.key] for target in partition]
        repair_rng = random.Random(stable_seed(seed, label, "global-meeting-repair"))
        _repair_same_meeting(partition, donors, repair_rng)
        for target, donor in zip(partition, donors):
            mapping[target.key] = donor
    donor_keys = [turn.key for turn in mapping.values()]
    if set(mapping) != source_keys or len(donor_keys) != len(source_keys):
        raise AssertionError("shuffle mapping does not cover every selected turn")
    if len(set(donor_keys)) != len(donor_keys):
        raise AssertionError("shuffle mapping reuses a donor turn")
    if any(target == donor.key for target, donor in mapping.items()):
        raise AssertionError("shuffle mapping contains an identity assignment")
    if any((target.voice_id != donor.voice_id or target.event != donor.event)
           for target in turns for donor in [mapping[target.key]]):
        raise AssertionError("shuffle mapping violated voice/event matching")

    duration_delta = [abs(turn.duration - mapping[turn.key].duration) for turn in turns]
    same_level = sum(turn.level == mapping[turn.key].level for turn in turns)
    same_meeting = sum(turn.recording_id == mapping[turn.key].recording_id for turn in turns)
    stats = {
        "turns": len(turns),
        "partitions": len(partitions),
        "duration_blocks": block_count,
        "duration_block_size": duration_block_size,
        "voice_match_rate": 1.0,
        "event_match_rate": 1.0,
        "level_match_rate": same_level / len(turns),
        "level_relaxed_target_turns": sparse_level_turns,
        "same_meeting_donor_turns": same_meeting,
        "same_meeting_after_local_duration_blocks": same_meeting_after_local_blocks,
        "duration_abs_difference_seconds": {
            "mean": sum(duration_delta) / len(duration_delta),
            "median": _quantile(duration_delta, 0.5),
            "p95": _quantile(duration_delta, 0.95),
            "max": max(duration_delta),
        },
    }
    return mapping, stats


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _highpass_filter() -> np.ndarray:
    """The source mixer's FIR high-pass (spectral inversion of a Hamming low-pass)."""
    lowpass = firwin(FIR_TAPS, FIR_CUTOFF_HZ / (SR / 2.0), window="hamming")
    highpass = -lowpass
    highpass[FIR_TAPS // 2] += 1.0
    return highpass.astype("float32")


def _load_clip(path: Path) -> tuple[np.ndarray, int]:
    wav, sampling_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if not wav.size:
        raise ValueError(f"{path}: empty audio")
    return wav.astype("float32", copy=False), int(sampling_rate)


def _normalize_to(wav: np.ndarray, target_db: float) -> np.ndarray:
    """Scale to an RMS of target_db dBFS (silence is returned unchanged)."""
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms <= 0.0:
        return wav.astype("float32", copy=False)
    return (wav * ((10.0 ** (target_db / 20.0)) / rms)).astype("float32")


def _interval_overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_end = float(first["onset"]) + float(first["dur"])
    second_end = float(second["onset"]) + float(second["dur"])
    return max(0.0, min(first_end, second_end) - max(float(first["onset"]), float(second["onset"])))


def _retime_records(meeting: Meeting, records: list[dict[str, Any]], timing_mode: str) -> None:
    """Reflow donor durations while retaining realized source gaps and overlaps.

    gaps-overlaps (default): each floor turn keeps the source pause after (or
    overlap with) the previous floor turn; a backchannel keeps its overlap
    with its host's end, capped by the shorter of the two clips.
    fixed-onsets: every turn starts at its source onset (overlaps may change).
    Records are modified in place and get a `timing` provenance block.
    """
    if timing_mode not in {"gaps-overlaps", "fixed-onsets"}:
        raise ValueError(f"unsupported timing mode: {timing_mode}")
    source_by_uid = {turn.uid: turn.record for turn in meeting.turns}
    slots = {int(slot["uid"]): slot for slot in meeting.source_plan.get("skeleton", {}).get("slots", [])}
    placed: dict[int, dict[str, Any]] = {}
    previous_floor_uid: int | None = None

    for record in records:
        uid = int(record["uid"])
        source_record = source_by_uid[uid]
        source_onset = float(source_record["onset"])
        event = str(record["event"])
        relation = "initial"
        target_value = 0.0

        if timing_mode == "fixed-onsets":
            onset = source_onset
            relation = "fixed_onset"
        elif event == "backchannel":
            host_uid = slots.get(uid, {}).get("host_uid")
            if host_uid is None or int(host_uid) not in placed:
                onset = source_onset
                relation = "backchannel_fixed_onset_missing_host"
                record["checks"].append("backchannel host unavailable; source onset retained")
            else:
                host_uid = int(host_uid)
                source_host = source_by_uid[host_uid]
                new_host = placed[host_uid]
                target_value = _interval_overlap(source_host, source_record)
                capacity = min(float(new_host["dur"]), float(record["dur"]))
                desired_overlap = min(target_value, capacity)
                onset = max(0.0, float(new_host["onset"]) + float(new_host["dur"]) - desired_overlap)
                relation = "host_overlap"
                if desired_overlap + 1e-9 < target_value:
                    record["checks"].append(
                        f"backchannel overlap limited from {target_value:.3f}s to {desired_overlap:.3f}s")
        elif previous_floor_uid is None:
            onset = source_onset
        else:
            source_previous = source_by_uid[previous_floor_uid]
            new_previous = placed[previous_floor_uid]
            source_previous_end = float(source_previous["onset"]) + float(source_previous["dur"])
            new_previous_end = float(new_previous["onset"]) + float(new_previous["dur"])
            if event == "interrupt":
                target_value = source_previous_end - source_onset
                onset = new_previous_end - target_value
                relation = "preceding_overlap"
            else:
                target_value = source_onset - source_previous_end
                onset = new_previous_end + target_value
                relation = "preceding_pause"
            if onset < 0.0:
                record["checks"].append(f"placement clamped from {onset:.3f}s to zero")
                onset = 0.0

        record["onset"] = float(onset)
        if relation == "host_overlap" and host_uid is not None and int(host_uid) in placed:
            realized_value = _interval_overlap(placed[int(host_uid)], record)
        elif relation == "preceding_overlap" and previous_floor_uid is not None:
            realized_value = (float(placed[previous_floor_uid]["onset"])
                              + float(placed[previous_floor_uid]["dur"]) - float(onset))
        elif relation == "preceding_pause" and previous_floor_uid is not None:
            realized_value = (float(onset) - float(placed[previous_floor_uid]["onset"])
                              - float(placed[previous_floor_uid]["dur"]))
        else:
            realized_value = float(onset)
        record["timing"] = {
            "mode": timing_mode,
            "relation": relation,
            "target_seconds": round(target_value, 6),
            "realized_seconds": round(realized_value, 6),
            "source_onset": round(source_onset, 6),
            "placed_onset": round(float(onset), 6),
        }
        record["shuffle"]["placed_onset"] = float(onset)
        placed[uid] = record
        if event != "backchannel":
            previous_floor_uid = uid


def _build_output_records(meeting: Meeting, mapping: dict[tuple[str, int], Turn], timing_mode: str) -> list[dict[str, Any]]:
    """Target slot records with the donor's audio/text and the target's level, retimed."""
    records: list[dict[str, Any]] = []
    for target in meeting.turns:
        donor = mapping[target.key]
        record = copy.deepcopy(target.record)
        record.update({
            "dur": donor.duration,
            "text": donor.record["text"],
            "instruct": donor.record["instruct"],
            "level": donor.level,
            "src": donor.record.get("src"),
            "path": donor.clip_relpath,
            "target_speech_level_db": target.speech_level_db,
            "checks": [f"shuffled-turn donor={donor.recording_id}/uid{donor.uid}"],
            "shuffle": {
                "source_recording_id": donor.recording_id,
                "source_uid": donor.uid,
                "source_clip": donor.clip_relpath,
                "source_voice_id": donor.voice_id,
                "source_event": donor.event,
                "source_level": donor.level,
                "source_duration": donor.duration,
                "target_voice_id": target.voice_id,
                "target_event": target.event,
                "target_level": target.level,
                "target_duration": target.duration,
                "target_onset": float(target.record["onset"]),
            },
        })
        records.append(record)
    _retime_records(meeting, records, timing_mode)
    return records


def _write_wav_atomic(path: Path, wav: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, wav, SR, subtype="PCM_16")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# --write-stems-only: recover stems for an existing shuffled corpus
# --------------------------------------------------------------------------

def _load_stem_recovery_plans(source: Path, out: Path, limit: int | None,
                              recording_ids: set[str] | None) -> list[dict[str, Any]]:
    """Load saved shuffled realizations without recomputing their turn mapping."""
    manifest_dir = out / "manifests"
    provenance_path = manifest_dir / "shuffled-turn.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Missing shuffled-turn provenance: {provenance_path}. "
            "--write-stems-only requires a completed shuffled-turn corpus."
        )
    provenance = json.load(open(provenance_path))
    if provenance.get("method") != "shuffled-turn":
        raise RuntimeError(f"{provenance_path} is not a shuffled-turn corpus")
    recorded_source = provenance.get("source")
    if not isinstance(recorded_source, str) or Path(recorded_source).resolve() != source:
        raise RuntimeError(
            "The supplied --source does not match the source recorded in "
            f"{provenance_path}; refusing to reconstruct stems from different clips."
        )

    plan_paths = sorted(manifest_dir.glob("ttsmeeting_*_realized.json"),
                        key=lambda path: _meeting_sort_key(path.name.removesuffix("_realized.json")))
    if recording_ids is not None:
        plan_paths = [path for path in plan_paths
                      if path.name.removesuffix("_realized.json") in recording_ids]
        found = {path.name.removesuffix("_realized.json") for path in plan_paths}
        missing = sorted(recording_ids - found)
        if missing:
            raise FileNotFoundError(f"No shuffled realization for selected recording {missing[0]}")
    if limit is not None:
        plan_paths = plan_paths[:limit]
    if not plan_paths:
        raise RuntimeError("No shuffled meetings selected for stem recovery")

    plans: list[dict[str, Any]] = []
    for plan_path in plan_paths:
        plan = json.load(open(plan_path))
        recording_id = plan_path.name.removesuffix("_realized.json")
        records = plan.get("records")
        mix = plan.get("mix")
        if plan.get("method") != "shuffled-turn" or plan.get("id") != recording_id:
            raise RuntimeError(f"{plan_path}: invalid shuffled-turn realization")
        if not isinstance(records, list) or not records or not isinstance(mix, dict):
            raise RuntimeError(f"{plan_path}: missing records or mix metadata")
        if mix.get("sample_rate") != SR or mix.get("highpass_taps") != FIR_TAPS:
            raise RuntimeError(f"{plan_path}: incompatible saved mixer configuration")
        gain = mix.get("headroom_gain")
        if not isinstance(gain, (float, int)) or not 0.0 < gain <= 1.0:
            raise RuntimeError(f"{plan_path}: invalid saved headroom gain")
        speakers = sorted({str(record.get("speaker")) for record in records})
        if not speakers or "None" in speakers:
            raise RuntimeError(f"{plan_path}: records have no valid speakers")
        mix_path = out / "audio" / f"{recording_id}.wav"
        if not mix_path.is_file():
            raise FileNotFoundError(f"{recording_id}: missing existing mixture {mix_path}")
        info = sf.info(mix_path)
        if info.samplerate != SR or info.channels != 1 or info.frames <= 0:
            raise RuntimeError(f"{mix_path}: expected non-empty mono {SR}-Hz audio")
        plans.append({"recording_id": recording_id, "records": records,
                      "speakers": speakers, "mix_path": mix_path,
                      "num_samples": info.frames, "headroom_gain": float(gain)})
    return plans


def _stem_paths(out: Path, plan: dict[str, Any]) -> list[Path]:
    recording_id = str(plan["recording_id"])
    return [out / "audio" / f"{recording_id}-spk-{speaker}.wav"
            for speaker in plan["speakers"]]


def _stems_complete(out: Path, plan: dict[str, Any]) -> bool:
    for path in _stem_paths(out, plan):
        if not path.is_file() or path.stat().st_size == 0:
            return False
        info = sf.info(path)
        if info.samplerate != SR or info.channels != 1 or info.frames != plan["num_samples"]:
            return False
    return True


def _recover_stems_one(plan: dict[str, Any], source: Path, out: Path,
                       highpass: np.ndarray) -> tuple[str, float, float, float]:
    """Reconstruct stems and verify them against an immutable existing mixture."""
    started = time.perf_counter()
    recording_id = str(plan["recording_id"])
    mix, mix_sr = _load_clip(Path(plan["mix_path"]))
    if mix_sr != SR or len(mix) != plan["num_samples"]:
        raise RuntimeError(f"{recording_id}: existing mixture changed during stem recovery")
    stems = {speaker: np.zeros_like(mix) for speaker in plan["speakers"]}
    for record in plan["records"]:
        speaker = str(record.get("speaker"))
        if speaker not in stems:
            raise RuntimeError(f"{recording_id}: record has unknown speaker {speaker!r}")
        clip = Path(str(record.get("path", "")))
        if not clip.is_absolute():
            clip = source / clip
        wav, source_sr = _load_clip(clip)
        if source_sr != SR:
            wav = resample_poly(wav, SR, source_sr).astype("float32")
        raw_duration = len(wav) / SR
        expected_duration = float(record["dur"])
        if abs(raw_duration - expected_duration) > 1.0 / SR:
            raise RuntimeError(
                f"{recording_id}/uid{record['uid']}: manifest duration "
                f"{expected_duration:.6f}s differs from donor file {raw_duration:.6f}s"
            )
        wav = wav - np.mean(wav, dtype="float32")
        if len(wav) >= len(highpass):
            wav = convolve(wav, highpass, mode="full", method="fft").astype("float32")
        wav = _normalize_to(wav, float(record["target_speech_level_db"]))
        offset = int(float(record["onset"]) * SR)
        if offset < 0 or offset + len(wav) > len(mix):
            raise RuntimeError(f"{recording_id}/uid{record['uid']}: turn exceeds existing mixture")
        stems[speaker][offset:offset + len(wav)] += wav

    gain = float(plan["headroom_gain"])
    if gain < 1.0:
        stems = {speaker: stem * gain for speaker, stem in stems.items()}
    reconstructed = np.zeros_like(mix)
    for stem in stems.values():
        reconstructed += stem
    max_error = float(np.max(np.abs(reconstructed - mix)))
    tolerance = 3.0 / 32768.0  # PCM-16 quantization plus summation roundoff.
    if max_error > tolerance:
        raise RuntimeError(
            f"{recording_id}: reconstructed stems do not match the existing mixture "
            f"(max error {max_error:.8f} > {tolerance:.8f}); mixture was not modified"
        )
    for speaker, stem in stems.items():
        _write_wav_atomic(out / "audio" / f"{recording_id}-spk-{speaker}.wav", stem)
    return recording_id, len(mix) / SR, time.perf_counter() - started, max_error


def _recover_stems(source: Path, out: Path, args, recording_ids: set[str] | None) -> None:
    plans = _load_stem_recovery_plans(source, out, args.limit, recording_ids)
    estimated = sum(int(plan["num_samples"]) * 2 * len(plan["speakers"]) for plan in plans)
    free = shutil.disk_usage(out.parent).free
    safety_margin = max(5 * 2**30, int(0.02 * estimated))
    print(f"stem recovery: {len(plans)} meetings; estimated new audio: {estimated / 2**30:.1f} GiB; "
          f"required free space: {(estimated + safety_margin) / 2**30:.1f} GiB; "
          f"free: {free / 2**30:.1f} GiB")
    if free < estimated + safety_margin:
        raise RuntimeError("insufficient free space for recovered stems plus safety margin")
    if args.dry_run:
        print("dry run complete; no stems were written")
        return

    metadata_path = out / "manifests" / "shuffled-turn-stems.json"
    metadata = {
        "schema_version": 1,
        "method": "shuffled-turn-stem-recovery",
        "source": str(source),
        "sample_rate": SR,
        "subtype": "PCM_16",
        "mix_verification_max_abs_tolerance": 3.0 / 32768.0,
    }
    if metadata_path.is_file():
        previous = json.load(open(metadata_path))
        if previous != metadata:
            raise RuntimeError("Refusing stem recovery with different provenance settings")
    elif any(path.exists() for plan in plans for path in _stem_paths(out, plan)):
        raise RuntimeError(
            "Existing stems have no recovery provenance; refusing to overwrite them. "
            "Move them aside or add a matching manifests/shuffled-turn-stems.json."
        )
    if not args.resume and any(path.exists() for plan in plans for path in _stem_paths(out, plan)):
        raise FileExistsError("Recovered stems already exist; use --resume to skip valid meetings")
    write_json_atomic(metadata_path, metadata)

    pending = [plan for plan in plans if not (args.resume and _stems_complete(out, plan))]
    print(f"stem recovery: {len(plans) - len(pending)} complete, {len(pending)} pending")
    highpass = _highpass_filter()
    times: list[float] = []
    max_errors: list[float] = []
    wall_start = time.perf_counter()
    if args.jobs == 1:
        for plan in pending:
            recording_id, duration, elapsed, max_error = _recover_stems_one(plan, source, out, highpass)
            times.append(elapsed)
            max_errors.append(max_error)
            print(f"[{recording_id}] {duration:.1f}s audio, {fmt_seconds(elapsed)}, "
                  f"mix error {max_error:.8f}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_recover_stems_one, plan, source, out, highpass)
                       for plan in pending]
            for future in concurrent.futures.as_completed(futures):
                recording_id, duration, elapsed, max_error = future.result()
                times.append(elapsed)
                max_errors.append(max_error)
                print(f"[{recording_id}] {duration:.1f}s audio, {fmt_seconds(elapsed)}, "
                      f"mix error {max_error:.8f}")
    if times:
        print(f"OK: recovered stems for {len(plans)} meetings; "
              f"avg={fmt_seconds(sum(times) / len(times))}, "
              f"max mix error={max(max_errors):.8f}, "
              f"wall={fmt_seconds(time.perf_counter() - wall_start)}")
    else:
        print(f"OK: recovered stems already exist for {len(plans)} meetings")


# --------------------------------------------------------------------------
# Main derivation
# --------------------------------------------------------------------------

def _render_one(meeting: Meeting, mapping: dict[tuple[str, int], Turn], source: Path,
                out: Path, write_stems: bool, highpass: np.ndarray, timing_mode: str) -> tuple[Recording, list[SupervisionSegment], float]:
    """Mix one shuffled meeting and write its audio, manifests and realization."""
    started = time.perf_counter()
    records = _build_output_records(meeting, mapping, timing_mode)
    speakers = sorted({str(record["speaker"]) for record in records})
    rendered: list[tuple[dict[str, Any], np.ndarray]] = []
    final_samples = 0
    for record in records:
        clip = Path(record["path"])
        if not clip.is_absolute():
            clip = source / clip
        wav, source_sr = _load_clip(clip)
        if source_sr != SR:
            wav = resample_poly(wav, SR, source_sr).astype("float32")
        raw_duration = len(wav) / SR
        if abs(raw_duration - float(record["dur"])) > 1.0 / SR:
            raise RuntimeError(
                f"{meeting.recording_id}/uid{record['uid']}: manifest duration "
                f"{record['dur']:.6f}s differs from donor file {raw_duration:.6f}s"
            )
        wav = wav - np.mean(wav, dtype="float32")
        if len(wav) >= len(highpass):
            wav = convolve(wav, highpass, mode="full", method="fft").astype("float32")
        wav = _normalize_to(wav, float(record["target_speech_level_db"]))
        if not np.any(wav):
            record["checks"].append("silent donor waveform retained")
        offset = int(float(record["onset"]) * SR)
        final_samples = max(final_samples, offset + len(wav))
        rendered.append((record, wav))

    # This matches the source TTS simulator's trailing one-second dry pad.
    mixed = np.zeros(final_samples + SR, dtype="float32")
    stems = {speaker: np.zeros_like(mixed) for speaker in speakers} if write_stems else {}
    for record, wav in rendered:
        offset = int(float(record["onset"]) * SR)
        mixed[offset:offset + len(wav)] += wav
        if write_stems:
            stems[str(record["speaker"])][offset:offset + len(wav)] += wav
    peak = float(np.abs(mixed).max())
    headroom_gain = min(1.0, 0.99 / peak) if peak else 1.0
    if headroom_gain < 1.0:
        mixed *= headroom_gain
        for stem in stems.values():
            stem *= headroom_gain

    audio_dir, manifest_dir = out / "audio", out / "manifests"
    mix_path = audio_dir / f"{meeting.recording_id}.wav"
    _write_wav_atomic(mix_path, mixed)
    for speaker, stem in stems.items():
        _write_wav_atomic(audio_dir / f"{meeting.recording_id}-spk-{speaker}.wav", stem)

    recording = Recording(
        id=meeting.recording_id,
        sources=[AudioSource(type="file", channels=[0], source=str(mix_path.resolve()))],
        sampling_rate=SR,
        num_samples=len(mixed),
        duration=len(mixed) / SR,
    )
    supervisions: list[SupervisionSegment] = []
    for index, record in enumerate(records):
        duration = max(round(float(record["dur"]), 4), 0.0001)
        supervisions.append(SupervisionSegment(
            id=f"{meeting.recording_id}_{index:04d}", recording_id=meeting.recording_id,
            start=round(float(record["onset"]), 4), duration=duration, channel=0,
            speaker=str(record["speaker"]), text=str(record["text"]), language="English",
            custom={
                "transition_type": str(record["event"]).upper(),
                "speech_level_db": round(float(record["target_speech_level_db"]), 2),
                "instruct": str(record["instruct"]),
                "level": str(record["level"]),
                "lombard_db": round(float(record.get("lombard_db", 0.0)), 2),
                "checks": record["checks"],
            },
        ))
    write_meeting_manifests(manifest_dir, recording, supervisions)

    derived = {
        "schema_version": SCHEMA_VERSION,
        "method": "shuffled-turn",
        "id": meeting.recording_id,
        "source_recording_id": meeting.recording_id,
        "target_duration": meeting.source_plan.get("target_duration"),
        "n_speakers": meeting.source_plan.get("n_speakers"),
        "personas": meeting.source_plan.get("personas"),
        "records": records,
        "mix": {
            "sample_rate": SR,
            "highpass_hz": FIR_CUTOFF_HZ,
            "highpass_taps": FIR_TAPS,
            "headroom_gain": round(headroom_gain, 8),
            "rendered_duration": round(recording.duration, 6),
            "timing_mode": timing_mode,
        },
    }
    write_json_atomic(manifest_dir / f"{meeting.recording_id}_realized.json", derived)
    return recording, supervisions, time.perf_counter() - started


def _meeting_output_paths(out: Path, recording_id: str, speakers: Iterable[str], write_stems: bool) -> list[Path]:
    manifest_dir = out / "manifests"
    paths = [
        out / "audio" / f"{recording_id}.wav",
        manifest_dir / f"{recording_id}_realized.json",
        manifest_dir / f"{recording_id}-recordings.jsonl.gz",
        manifest_dir / f"{recording_id}-supervisions.jsonl.gz",
        manifest_dir / f"{recording_id}-cuts.jsonl.gz",
        manifest_dir / f"{recording_id}.rttm",
    ]
    if write_stems:
        paths.extend(out / "audio" / f"{recording_id}-spk-{speaker}.wav" for speaker in speakers)
    return paths


def _meeting_complete(out: Path, meeting: Meeting, write_stems: bool) -> bool:
    speakers = {str(turn.record["speaker"]) for turn in meeting.turns}
    paths = _meeting_output_paths(out, meeting.recording_id, speakers, write_stems)
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        return False
    try:
        recordings = RecordingSet.from_file(out / "manifests" / f"{meeting.recording_id}-recordings.jsonl.gz")
        supervisions = SupervisionSet.from_file(out / "manifests" / f"{meeting.recording_id}-supervisions.jsonl.gz")
        lhotse.validate_recordings_and_supervisions(recordings, supervisions)
        return len(recordings) == 1 and len(supervisions) == len(meeting.turns)
    except Exception:
        return False


def _write_mapping(path: Path, turns: list[Turn], mapping: dict[tuple[str, int], Turn]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as handle:
        for target in sorted(turns, key=lambda turn: (_meeting_sort_key(turn.recording_id), turn.uid)):
            donor = mapping[target.key]
            handle.write(json.dumps({
                "recording_id": target.recording_id,
                "uid": target.uid,
                "donor_recording_id": donor.recording_id,
                "donor_uid": donor.uid,
                "voice_id": target.voice_id,
                "event": target.event,
                "target_level": target.level,
                "donor_level": donor.level,
                "target_duration": target.duration,
                "donor_duration": donor.duration,
            }) + "\n")
    os.replace(tmp, path)


def _estimated_output_bytes(meetings: list[Meeting], mapping: dict[tuple[str, int], Turn], write_stems: bool, timing_mode: str) -> int:
    total = 0
    for meeting in meetings:
        end = max(float(record["onset"]) + float(record["dur"]) for record in _build_output_records(meeting, mapping, timing_mode)) + 1.0
        mix_bytes = int(math.ceil(end * SR * 2)) + 44
        speakers = len({turn.record["speaker"] for turn in meeting.turns})
        total += mix_bytes * (1 + speakers if write_stems else 1)
    return total


def _ensure_metadata(out: Path, metadata: dict[str, Any], resume: bool) -> None:
    """Write shuffled-turn.json; on --resume, refuse if the configuration changed."""
    path = out / "manifests" / "shuffled-turn.json"
    if path.is_file() and resume:
        previous = json.load(open(path))
        keys = ("schema_version", "method", "source", "seed", "duration_block_size",
                "selected_recordings", "write_stems", "timing_mode")
        changed = [key for key in keys if previous.get(key) != metadata.get(key)]
        if changed:
            raise RuntimeError(
                "Refusing to resume with a different shuffled-turn configuration: " + ", ".join(changed)
            )
    write_json_atomic(path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed dry TTS corpus directory.")
    parser.add_argument("--out", required=True, help="New directory for the shuffled-turn corpus.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed.")
    parser.add_argument("--duration-block-size", type=int, default=8,
                        help="Maximum turns per local duration-matched shuffle block (default: 8).")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CPU mixing jobs (default: 1).")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None,
                           help="Only derive the first N meetings. The selected split must contain at least two turns for every clone/event bucket.")
    selection.add_argument("--recording-ids", default=None,
                           help="Text file of recording IDs to derive; defines a leak-free split.")
    parser.add_argument("--write-stems", action="store_true",
                        help="Write per-speaker stems for later RIR/noise augmentation (large).")
    parser.add_argument("--write-stems-only", action="store_true",
                        help="Add missing speaker stems to an existing shuffled corpus without rewriting its mixtures.")
    parser.add_argument("--resume", action="store_true", help="Skip valid completed output meetings.")
    parser.add_argument("--timing-mode", choices=["gaps-overlaps", "fixed-onsets"],
                        default="gaps-overlaps",
                        help="Preserve realized pauses and overlaps (default), or use the legacy fixed-onset placement.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate source data and construct the deterministic mapping without writing output.")
    args = parser.parse_args()

    if args.duration_block_size < 2:
        parser.error("--duration-block-size must be at least 2")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.write_stems and args.write_stems_only:
        parser.error("--write-stems and --write-stems-only cannot be used together")

    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if source == out:
        parser.error("--out must be different from --source")
    if not source.is_dir():
        raise FileNotFoundError(source)
    recording_ids = _read_selected_ids(Path(args.recording_ids)) if args.recording_ids else None
    if args.write_stems_only:
        if not out.is_dir():
            raise FileNotFoundError(
                f"{out}: --write-stems-only requires an existing shuffled-turn output directory"
            )
        _recover_stems(source, out, args, recording_ids)
        return
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"{out} already exists and is not empty; use --resume or a new --out")

    print("loading and validating realized source turns ...")
    meetings = _load_meetings(source, args.limit, recording_ids)
    turns = [turn for meeting in meetings for turn in meeting.turns]
    print(f"source preflight: {len(meetings)} meetings, {len(turns)} turns, "
          f"{len({turn.voice_id for turn in turns})} clone voices")
    acoustic_markers = [source / "rirs", source / "manifests" / "augmentation.json"]
    if any(path.exists() for path in acoustic_markers):
        print("WARNING: source appears acoustically augmented. This derivation uses dry final turn clips; "
              "apply acoustics after shuffling for a matched acoustic variant.")

    mapping, stats = build_mapping(turns, args.seed, args.duration_block_size)
    print("shuffle: "
          f"voice={stats['voice_match_rate']:.3f}, event={stats['event_match_rate']:.3f}, "
          f"level={stats['level_match_rate']:.6f}, same-meeting={stats['same_meeting_donor_turns']}, "
          f"|duration delta| median/p95/max="
          f"{stats['duration_abs_difference_seconds']['median']:.3f}/"
          f"{stats['duration_abs_difference_seconds']['p95']:.3f}/"
          f"{stats['duration_abs_difference_seconds']['max']:.3f}s")

    if args.dry_run:
        print("dry run complete; no output was written")
        return

    estimate = _estimated_output_bytes(meetings, mapping, args.write_stems, args.timing_mode)
    free = shutil.disk_usage(out.parent).free
    safety_margin = max(5 * 2**30, int(0.02 * estimate))
    required_free = estimate + safety_margin
    print(f"estimated audio output: {estimate / 2**30:.1f} GiB; required free space: "
          f"{required_free / 2**30:.1f} GiB; free at {out.parent}: {free / 2**30:.1f} GiB")
    if free < required_free:
        raise RuntimeError("insufficient free space for the estimated audio plus safety margin")

    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    source_bank = source / "bank.json"
    if source_bank.is_file() and not (out / "bank.json").exists():
        shutil.copy2(source_bank, out / "bank.json")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "shuffled-turn",
        "source": str(source),
        "seed": args.seed,
        "duration_block_size": args.duration_block_size,
        "selected_recordings": [meeting.recording_id for meeting in meetings],
        "write_stems": args.write_stems,
        "timing_mode": args.timing_mode,
        "audio": {"sample_rate": SR, "subtype": "PCM_16", "highpass_hz": FIR_CUTOFF_HZ,
                  "highpass_taps": FIR_TAPS, "trailing_silence_seconds": 1.0},
        "matching": {
            "required": ["clone_voice_id", "event"],
            "preferred": ["delivery_level", "local_duration_stratum"],
            "description": "Deterministic one-to-one turn derangement within clone voice and event class.",
        },
        "statistics": stats,
    }
    _ensure_metadata(out, metadata, args.resume)
    _write_mapping(out / "manifests" / "shuffled-turn-mapping.jsonl.gz", turns, mapping)

    pending = [meeting for meeting in meetings if not (args.resume and _meeting_complete(out, meeting, args.write_stems))]
    print(f"shuffled-turn render: {len(meetings) - len(pending)} complete, {len(pending)} pending")
    highpass = _highpass_filter()
    timings: list[float] = []
    wall_start = time.perf_counter()
    if args.jobs == 1:
        for meeting in pending:
            print(f"[{meeting.recording_id}] mix")
            _, _, elapsed = _render_one(meeting, mapping, source, out, args.write_stems, highpass, args.timing_mode)
            timings.append(elapsed)
            print(f"  {fmt_seconds(elapsed)}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_render_one, meeting, mapping, source, out,
                                       args.write_stems, highpass, args.timing_mode): meeting for meeting in pending}
            for future in concurrent.futures.as_completed(futures):
                meeting = futures[future]
                _, _, elapsed = future.result()
                timings.append(elapsed)
                print(f"[{meeting.recording_id}] mix {fmt_seconds(elapsed)}")
    combine_meeting_manifests(out / "manifests", [meeting.recording_id for meeting in meetings])

    duration = sum(recording.duration for recording in RecordingSet.from_file(
        out / "manifests" / "tts-recordings.jsonl.gz"))
    print(f"OK: {len(meetings)} meetings, {len(turns)} supervisions, {duration / 3600.0:.2f} h -> {out}")
    if timings:
        print(f"Shuffled-turn mixing: n={len(timings)}, avg={fmt_seconds(sum(timings) / len(timings))}, "
              f"min={fmt_seconds(min(timings))}, max={fmt_seconds(max(timings))}, "
              f"wall={fmt_seconds(time.perf_counter() - wall_start)}")


if __name__ == "__main__":
    main()
