#!/usr/bin/env python3
"""Compare a completed Shuffled-Turn corpus with its Full source corpus.

Hard checks (any failure raises): same recording ids and uids on both sides;
every turn has a donor that is not itself, each donor is used exactly once, and
clone voice and event type are preserved; recorded durations/onsets match both
realizations; the gzip mapping covers exactly the same turns. --check-audio also
verifies every mixture's WAV header (mono, 16 kHz, duration = manifest).

The report (stdout, and --report-json) quantifies what is allowed to differ:
rendered and speech duration, same-meeting donors, delivery-level mismatches,
and the reflow error of pauses/overlaps per timing relation. Only reads the
manifests (plus WAV headers), so it is fast even on thousands of meetings.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


EPS = 1e-6       # tolerance (seconds) for durations/onsets copied between manifests


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * q)] if values else 0.0


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "median": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "max": max(values, default=0.0),
    }


def recordings(path: Path) -> dict[str, float]:
    """Recording id -> duration from a Lhotse recordings manifest (read as plain JSON)."""
    with gzip.open(path, "rt") as handle:
        return {item["id"]: float(item["duration"])
                for line in handle if line.strip()
                for item in [json.loads(line)]}


def plan(path: Path) -> tuple[dict, dict[int, dict]]:
    """A realized plan and its records keyed by uid."""
    data = json.loads(path.read_text())
    records = {int(record["uid"]): record for record in data["records"]}
    if len(records) != len(data["records"]):
        raise RuntimeError(f"duplicate UID in {path}")
    return data, records


def require_close(name: str, actual: float, expected: float) -> None:
    if abs(actual - expected) > EPS:
        raise RuntimeError(f"{name}: {actual:.6f} != {expected:.6f}")


def check_audio(root: Path, durations: dict[str, float]) -> None:
    import soundfile as sf
    for index, (recording_id, duration) in enumerate(sorted(durations.items()), 1):
        info = sf.info(root / "audio" / f"{recording_id}.wav")
        actual = info.frames / info.samplerate
        if info.samplerate != 16000 or info.channels != 1:
            raise RuntimeError(f"{recording_id}: expected mono 16 kHz")
        if abs(actual - duration) > 1.0 / info.samplerate + EPS:
            raise RuntimeError(f"{recording_id}: WAV/header duration mismatch")
        if index % 500 == 0:
            print(f"  audio headers: {index:,}/{len(durations):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="The dry corpus the shuffled one was derived from.")
    parser.add_argument("--shuffled", type=Path, required=True,
                        help="Output directory of derive_shuffled_turns.py.")
    parser.add_argument("--check-audio", action="store_true",
                        help="Also check every mixture's WAV header in both corpora.")
    parser.add_argument("--report-json", type=Path,
                        help="Write the report here (e.g. <shuffled>/manifests/validation.json).")
    args = parser.parse_args()
    source, shuffled = args.source.resolve(), args.shuffled.resolve()
    source_m, shuffled_m = source / "manifests", shuffled / "manifests"

    metadata = json.loads((shuffled_m / "shuffled-turn.json").read_text())
    if metadata.get("method") != "shuffled-turn":
        raise RuntimeError("not a Shuffled-Turn corpus")
    if Path(metadata["source"]).resolve() != source:
        raise RuntimeError("shuffled metadata source does not match --source")

    source_duration = recordings(source_m / "tts-recordings.jsonl.gz")
    # A derivation may cover only part of the source (--limit / --recording-ids);
    # compare against the meetings it actually selected.
    selected = set(metadata["selected_recordings"])
    source_duration = {k: v for k, v in source_duration.items() if k in selected}
    shuffled_duration = recordings(shuffled_m / "tts-recordings.jsonl.gz")
    if set(source_duration) != set(shuffled_duration):
        raise RuntimeError("source and shuffled recording IDs differ")
    if set(metadata["selected_recordings"]) != set(source_duration):
        raise RuntimeError("metadata selected_recordings differs from aggregate manifest")

    donor_keys: set[tuple[str, int]] = set()
    target_keys: set[tuple[str, int]] = set()
    meeting_delta, meeting_delta_percent, speech_delta, turn_delta = [], [], [], []
    largest_meeting_deltas: list[tuple[float, str, float, float]] = []
    source_speech_total = shuffled_speech_total = 0.0
    timing_error: dict[str, list[float]] = defaultdict(list)
    timing_count: Counter[str] = Counter()
    same_meeting = level_mismatch = limited_backchannels = 0

    for index, recording_id in enumerate(sorted(source_duration), 1):
        source_plan, source_records = plan(source_m / f"{recording_id}_realized.json")
        shuffled_plan, shuffled_records = plan(shuffled_m / f"{recording_id}_realized.json")
        if shuffled_plan.get("method") != "shuffled-turn":
            raise RuntimeError(f"{recording_id}: missing shuffled-turn method")
        if set(source_records) != set(shuffled_records):
            raise RuntimeError(f"{recording_id}: source/shuffled UIDs differ")

        source_speech = shuffled_speech = 0.0
        for uid, full in source_records.items():
            out = shuffled_records[uid]
            info, timing = out.get("shuffle"), out.get("timing")
            if not isinstance(info, dict) or not isinstance(timing, dict):
                raise RuntimeError(f"{recording_id}/uid{uid}: missing provenance")
            target = (recording_id, uid)
            donor = (str(info["source_recording_id"]), int(info["source_uid"]))
            if donor == target:
                raise RuntimeError(f"{recording_id}/uid{uid}: identity donor")
            if donor in donor_keys:
                raise RuntimeError(f"duplicate donor: {donor}")
            donor_keys.add(donor)
            target_keys.add(target)
            same_meeting += donor[0] == recording_id
            if info["target_voice_id"] != info["source_voice_id"]:
                raise RuntimeError(f"{recording_id}/uid{uid}: clone voice changed")
            if info["target_event"] != info["source_event"] or out["event"] != full["event"]:
                raise RuntimeError(f"{recording_id}/uid{uid}: event changed")
            level_mismatch += info["target_level"] != info["source_level"]
            require_close(f"{recording_id}/uid{uid} target duration",
                          float(info["target_duration"]), float(full["dur"]))
            require_close(f"{recording_id}/uid{uid} donor duration",
                          float(info["source_duration"]), float(out["dur"]))
            require_close(f"{recording_id}/uid{uid} source onset",
                          float(timing["source_onset"]), float(full["onset"]))
            require_close(f"{recording_id}/uid{uid} placed onset",
                          float(timing["placed_onset"]), float(out["onset"]))
            source_speech += float(full["dur"])
            shuffled_speech += float(out["dur"])
            turn_delta.append(abs(float(full["dur"]) - float(out["dur"])))
            relation = str(timing["relation"])
            timing_count[relation] += 1
            if relation in {"preceding_pause", "preceding_overlap", "host_overlap"}:
                timing_error[relation].append(
                    abs(float(timing["target_seconds"]) - float(timing["realized_seconds"])))
            limited_backchannels += any(
                "backchannel overlap limited" in str(check) for check in out.get("checks", []))
        delta = shuffled_duration[recording_id] - source_duration[recording_id]
        meeting_delta.append(abs(delta))
        meeting_delta_percent.append(100.0 * abs(delta) / source_duration[recording_id])
        largest_meeting_deltas.append(
            (abs(delta), recording_id, source_duration[recording_id], shuffled_duration[recording_id]))
        speech_delta.append(abs(source_speech - shuffled_speech))
        source_speech_total += source_speech
        shuffled_speech_total += shuffled_speech
        if index % 250 == 0:
            print(f"  plans: {index:,}/{len(source_duration):,}")

    if donor_keys != target_keys:
        raise RuntimeError("donor mapping is not a permutation of the selected turns")

    map_rows = 0
    with gzip.open(shuffled_m / "shuffled-turn-mapping.jsonl.gz", "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            target = (str(item["recording_id"]), int(item["uid"]))
            donor = (str(item["donor_recording_id"]), int(item["donor_uid"]))
            if target not in target_keys or donor not in donor_keys:
                raise RuntimeError(f"mapping contains unknown turn: {target} -> {donor}")
            map_rows += 1
    if map_rows != len(target_keys):
        raise RuntimeError(f"mapping rows={map_rows}, expected={len(target_keys)}")

    if args.check_audio:
        check_audio(source, source_duration)
        check_audio(shuffled, shuffled_duration)

    source_total, shuffled_total = sum(source_duration.values()), sum(shuffled_duration.values())
    report = {
        "meetings": len(source_duration),
        "turns": len(target_keys),
        "rendered_duration": {
            "source_seconds": source_total,
            "shuffled_seconds": shuffled_total,
            "difference_seconds": shuffled_total - source_total,
            "difference_percent": 100 * (shuffled_total - source_total) / source_total,
            "per_meeting_abs_difference_seconds": summary(meeting_delta),
            "per_meeting_abs_difference_percent": summary(meeting_delta_percent),
            "largest_per_meeting_differences": [
                {
                    "recording_id": recording_id,
                    "source_seconds": source_seconds,
                    "shuffled_seconds": shuffled_seconds,
                    "difference_seconds": shuffled_seconds - source_seconds,
                }
                for _, recording_id, source_seconds, shuffled_seconds
                in sorted(largest_meeting_deltas, reverse=True)[:10]
            ],
        },
        "speech_duration_abs_difference_seconds": {
            "per_turn": summary(turn_delta),
            "per_meeting": summary(speech_delta),
            "global_difference_seconds": shuffled_speech_total - source_speech_total,
        },
        "mapping": {
            "one_to_one": True,
            "identity_assignments": 0,
            "same_meeting_donors": same_meeting,
            "delivery_level_mismatches": level_mismatch,
        },
        "reflow_abs_error_seconds": {key: summary(value) for key, value in timing_error.items()},
        "reflow_relation_counts": dict(sorted(timing_count.items())),
        "backchannels_limited_by_donor_or_host": limited_backchannels,
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("OK: all structural, duration, mapping, and reflow checks passed.")


if __name__ == "__main__":
    main()
