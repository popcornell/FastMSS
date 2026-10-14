"""Apply FastMSS-style room acoustics and noise to an existing TTS corpus.

The staged TTS recipe saves a dry mixture and one dry stem per speaker.  FastMSS
uses one RIR position per speaker within a meeting, so convolving the existing
stems is equivalent to convolving each source utterance before summing them.  This
script keeps the source corpus read-only and writes a standalone acoustic variant
with its own audio and manifests.

The source needs per-speaker stems: tts_staged.py writes them by default, and
derive_shuffled_turns.py writes them with --write-stems (or recovers them later
with --write-stems-only). Supervisions are copied unchanged (labels stay on the
dry timeline); each meeting draws one room and one RIR position per speaker,
seeded by (--seed, recording id), so results do not depend on --jobs or order.

Extra outputs:
  rirs/ + manifests/rir-bank.json   the simulated RIR bank (reused on --resume)
  manifests/augmentation.json       global settings; --resume refuses changes
  manifests/<id>_acoustics.json     per-meeting room, RIRs, noise segments, gains
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from lhotse import AudioSource, Recording, RecordingSet, SupervisionSet
from scipy.signal import convolve

from fastmss.rirsimulator import RIRSimulator
from fastmss.tts.corpus import (combine_meeting_manifests, fmt_seconds, stable_seed,
                                write_meeting_manifests)
from fastmss.utils import concatenate_audio_with_crossfade, repeat_audio_with_crossfade


SR = 16000


# --------------------------------------------------------------------------
# RIR bank
# --------------------------------------------------------------------------

def _rir_cfg(output_dir: Path, positions: int, room_size: list[float],
             room_ceiling: list[float], rt60: list[float]) -> SimpleNamespace:
    """The same RIR configuration used by ``tts_staged.py``."""
    return SimpleNamespace(
        samplerate=SR,
        output_dir=str(output_dir),
        save_anechoic=False,
        rt60=rt60,
        room_sz=room_size,
        room_ceiling=room_ceiling,
        n_pos_rirs=positions,
        delta_dist=0.5,
        use_rand_ism=False,
        rand_disp=0.0,
        mic_type="single",
    )


def _load_rir_bank(out: Path) -> list[list[str]] | None:
    """Previously generated RIR paths per room, or None if absent/incomplete."""
    path = out / "manifests" / "rir-bank.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    rooms = data.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        return None
    resolved = []
    for room in rooms:
        paths = [str(out / p) if not Path(p).is_absolute() else p for p in room]
        if not paths or any(not Path(p).is_file() for p in paths):
            return None
        resolved.append(paths)
    return resolved


def _prepare_rirs(args, out: Path) -> list[list[str]] | None:
    """Reuse the variant's RIR bank, or simulate --rir-rooms rooms and record them."""
    if not args.reverberate:
        return None

    existing = _load_rir_bank(out)
    if existing is not None:
        print(f"RIR bank: reusing {len(existing)} room(s) from {out / 'rirs'}")
        return existing
    if args.resume and (out / "rirs").exists():
        raise RuntimeError(
            f"{out / 'rirs'} exists but has no valid manifests/rir-bank.json; "
            "refusing to silently generate a different RIR bank during resume")

    print(f"RIR bank: generating {args.rir_rooms} room(s), "
          f"{args.rir_positions} positions each")
    np.random.seed(args.seed)
    simulator = RIRSimulator(_rir_cfg(
        out, args.rir_positions, args.rir_room_size,
        args.rir_room_ceiling, args.rir_rt60))
    rooms = [simulator.gen_rirs(f"room_{i:03d}") for i in range(args.rir_rooms)]
    relative = [[str(Path(p).resolve().relative_to(out.resolve())) for p in room]
                for room in rooms]
    with open(out / "manifests" / "rir-bank.json", "w") as f:
        json.dump({"seed": args.seed, "rooms": relative,
                   "samplerate": SR, "positions_per_room": args.rir_positions}, f,
                  indent=2)
    return rooms


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------

def _load_mono(path: Path, expected_sr: int = SR) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if sr != expected_sr:
        raise ValueError(f"{path}: expected {expected_sr} Hz, found {sr} Hz")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if not wav.size:
        raise ValueError(f"{path}: empty audio")
    return wav.astype("float32", copy=False)


def _load_rir(path: str, expected_sr: int = SR) -> np.ndarray:
    """First channel of an RIR, cut at its direct-path peak so speech is not delayed."""
    rir, sr = sf.read(path, dtype="float32")
    if sr != expected_sr:
        raise ValueError(f"{path}: expected {expected_sr} Hz, found {sr} Hz")
    if rir.ndim > 1:
        rir = rir[:, 0]
    peak = int(np.argmax(np.abs(rir)))
    rir = rir[peak:]
    if not rir.size:
        raise ValueError(f"{path}: empty RIR after direct-path alignment")
    return rir.astype("float32", copy=False)


def _trim_trailing_silence(wav: np.ndarray) -> np.ndarray:
    """Remove the one-second dry-mixture pad before adding a reverberant tail."""
    nonzero = np.flatnonzero(np.abs(wav) > 0)
    return wav[:nonzero[-1] + 1] if nonzero.size else wav[:1]


def _pad_to(wav: np.ndarray, length: int) -> np.ndarray:
    return wav if len(wav) >= length else np.pad(wav, (0, length - len(wav)))


def _mix_stems(stems: dict[str, np.ndarray], dry_length: int) -> np.ndarray:
    length = max(dry_length, *(len(w) for w in stems.values()))
    mixed = np.zeros(length, dtype="float32")
    for wav in stems.values():
        mixed[:len(wav)] += wav
    return mixed


def _speech_level_db(supervisions, mix: np.ndarray) -> tuple[float, str]:
    """Quietest supervised speech level (noise reference), and where it came from."""
    levels = []
    for s in supervisions:
        custom = getattr(s, "custom", None) or {}
        value = custom.get("speech_level_db")
        if isinstance(value, (float, int)):
            levels.append(float(value))
    if levels:
        return min(levels), "source_supervision.speech_level_db"
    rms = float(np.sqrt(np.mean(mix ** 2)))
    return 20.0 * np.log10(rms + 1e-12), "mixed_rms_fallback"


def _read_noise(path: str, start: int, stop: int, expected_sr: int,
                rng: np.random.Generator) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, start=start, stop=stop, dtype="float32")
    if sr != expected_sr:
        raise ValueError(f"{path}: expected {expected_sr} Hz, found {sr} Hz")
    channel = 0
    if wav.ndim > 1:
        if wav.shape[1] > 1:
            channel = int(rng.integers(wav.shape[1]))
        wav = wav[:, channel]
    return wav.astype("float32", copy=False), channel


def _real_noise(mix: np.ndarray, files: list[str], min_speech_db: float,
                rel_gain: tuple[float, float], rng: np.random.Generator,
                multi_noise: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """FastMSS ``add_real_noise`` with deterministic RNG and provenance output."""
    target = len(mix)
    provenance: list[dict] = []
    if multi_noise:
        crossfade = int(0.5 * SR)
        segments: list[np.ndarray] = []
        sampled = 0
        while sampled < target:
            path = str(files[int(rng.integers(len(files)))])
            info = sf.SoundFile(path)
            if info.samplerate != SR:
                raise ValueError(f"{path}: expected {SR} Hz, found {info.samplerate} Hz")
            available = len(info)
            if available <= 0:
                raise ValueError(f"{path}: empty noise file")
            if not segments:
                read_length = min(available, target)
            else:
                remaining = target - sampled + len(segments) * crossfade
                read_length = min(available, max(remaining, crossfade * 2))
            start = int(rng.integers(0, available - read_length)) if available > read_length else 0
            segment, channel = _read_noise(path, start, start + read_length, SR, rng)
            rms = float(np.sqrt(np.mean(segment ** 2)))
            segment = segment / (rms + 1e-8)
            segments.append(segment)
            sampled += len(segment)
            provenance.append({"path": path, "start_sample": start,
                               "num_samples": len(segment), "channel": channel})
        noise = concatenate_audio_with_crossfade(segments, crossfade)
        noise = _pad_to(noise, target)[:target]
        noise = noise / (float(np.sqrt(np.mean(noise ** 2))) + 1e-8)
    else:
        path = str(files[int(rng.integers(len(files)))])
        info = sf.SoundFile(path)
        if info.samplerate != SR:
            raise ValueError(f"{path}: expected {SR} Hz, found {info.samplerate} Hz")
        start = int(rng.integers(0, len(info) - target)) if len(info) > target else 0
        noise, channel = _read_noise(path, start, start + target, SR, rng)
        if len(noise) < target:
            noise = repeat_audio_with_crossfade(noise, target, SR)
        noise = noise / (float(np.sqrt(np.mean(noise ** 2))) + 1e-8)
        provenance.append({"path": path, "start_sample": start,
                           "num_samples": min(target, len(info) - start), "channel": channel})

    level_db = float(min_speech_db + rng.uniform(*rel_gain))
    noise = (noise * (10.0 ** (level_db / 20.0))).astype("float32")
    return (mix + noise).astype("float32"), noise, {
        "kind": "real", "level_db": round(level_db, 3),
        "min_speech_level_db": round(min_speech_db, 3), "segments": provenance,
    }


def _gaussian_noise(mix: np.ndarray, min_speech_db: float,
                    rel_gain: tuple[float, float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Distributionally identical to FastMSS ``add_gaussian_noise``."""
    level_db = float(min_speech_db + rng.uniform(*rel_gain))
    noise = rng.normal(0.0, 10.0 ** (level_db / 20.0), size=mix.shape).astype("float32")
    return (mix + noise).astype("float32"), noise, {
        "kind": "gaussian", "level_db": round(level_db, 3),
        "min_speech_level_db": round(min_speech_db, 3),
    }


# --------------------------------------------------------------------------
# Per-meeting augmentation
# --------------------------------------------------------------------------

def _source_stem_path(source: Path, recording_id: str, speaker: str) -> Path:
    candidate = source / "audio" / f"{recording_id}-spk-{speaker}.wav"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{recording_id}: missing source stem for {speaker}: {candidate}. "
            "Acoustic augmentation requires the dry per-speaker stems.")
    return candidate


def _output_complete(out: Path, recording_id: str, speakers: list[str]) -> bool:
    """True if every output file of a meeting exists and is non-empty (for --resume)."""
    audio = out / "audio"
    manifest = out / "manifests"
    required = [audio / f"{recording_id}.wav",
                manifest / f"{recording_id}-recordings.jsonl.gz",
                manifest / f"{recording_id}-supervisions.jsonl.gz",
                manifest / f"{recording_id}-cuts.jsonl.gz",
                manifest / f"{recording_id}.rttm",
                manifest / f"{recording_id}_acoustics.json"]
    required += [audio / f"{recording_id}-spk-{speaker}.wav" for speaker in speakers]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _copy_text_metadata(source: Path, out: Path) -> None:
    """Carry plans, realizations, shuffle provenance and bank.json into the variant."""
    source_manifests, out_manifests = source / "manifests", out / "manifests"
    for path in source_manifests.glob("*_text.json"):
        shutil.copy2(path, out_manifests / path.name)
    for path in source_manifests.glob("*_realized.json"):
        shutil.copy2(path, out_manifests / path.name)
    for name in ("tts-text-plans.jsonl", "shuffled-turn.json",
                 "shuffled-turn-mapping.jsonl.gz", "shuffled-turn-stems.json",
                 "validation.json"):
        path = source_manifests / name
        if path.is_file():
            shutil.copy2(path, out_manifests / path.name)
    bank = source / "bank.json"
    if bank.is_file():
        shutil.copy2(bank, out / bank.name)


def _augment_one(recording, supervisions, args, source: Path, out: Path,
                 rir_rooms: list[list[str]] | None) -> tuple[str, float, float]:
    """Reverberate stems / add noise for one meeting; write audio, manifests, provenance.

    Order: per-speaker RIR convolution -> sum -> noise relative to the quietest
    speech level -> one shared headroom gain applied to mix, stems and noise.
    """
    started = time.perf_counter()
    recording_id = recording.id
    speakers = sorted({s.speaker for s in supervisions})
    rng = np.random.default_rng(stable_seed(args.seed, recording_id))

    source_mix = _load_mono(Path(recording.sources[0].source))
    dry_stems = {speaker: _load_mono(_source_stem_path(source, recording_id, speaker))
                 for speaker in speakers}
    if any(len(stem) != len(source_mix) for stem in dry_stems.values()):
        raise ValueError(f"{recording_id}: dry mixture and stems have inconsistent lengths")

    rirs: dict[str, str] = {}
    if rir_rooms is None:
        stems = dry_stems
        mixed = source_mix.copy()
        room_index = None
    else:
        room_index = int(rng.integers(len(rir_rooms)))
        room = rir_rooms[room_index]
        stems = {}
        for index, speaker in enumerate(speakers):
            rir_path = room[index % len(room)]
            rirs[speaker] = str(Path(rir_path).resolve().relative_to(out.resolve()))
            stems[speaker] = convolve(_trim_trailing_silence(dry_stems[speaker]),
                                      _load_rir(rir_path), mode="full").astype("float32")
        mixed = _mix_stems(stems, len(source_mix))

    min_speech_db, level_source = _speech_level_db(supervisions, mixed)
    noise = None
    noise_meta = None
    if args.add_noise:
        if args.noise_files:
            mixed, noise, noise_meta = _real_noise(
                mixed, args.noise_files, min_speech_db, tuple(args.noise_rel_gain),
                rng, args.multi_noise_sampling)
        else:
            mixed, noise, noise_meta = _gaussian_noise(
                mixed, min_speech_db, tuple(args.gaussian_noise_range), rng)

    peak = float(np.abs(mixed).max())
    headroom_gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
    if headroom_gain < 1.0:
        mixed *= headroom_gain
        stems = {speaker: stem * headroom_gain for speaker, stem in stems.items()}
        if noise is not None:
            noise *= headroom_gain

    length = len(mixed)
    stems = {speaker: _pad_to(stem, length).astype("float32")
             for speaker, stem in stems.items()}
    audio_dir, manifest_dir = out / "audio", out / "manifests"
    mix_path = audio_dir / f"{recording_id}.wav"
    sf.write(mix_path, mixed, SR)
    for speaker, stem in stems.items():
        sf.write(audio_dir / f"{recording_id}-spk-{speaker}.wav", stem, SR)
    if args.save_noise and noise is not None:
        noise_dir = out / "noise"
        noise_dir.mkdir(exist_ok=True)
        sf.write(noise_dir / f"{recording_id}.wav", _pad_to(noise, length), SR)

    augmented = Recording(
        id=recording_id,
        sources=[AudioSource(type="file", channels=[0], source=str(mix_path.resolve()))],
        sampling_rate=SR,
        num_samples=length,
        duration=length / SR,
    )
    write_meeting_manifests(manifest_dir, augmented, supervisions)

    acoustic_meta = {
        "recording_id": recording_id,
        "source_recording": str(Path(recording.sources[0].source).resolve()),
        "seed": stable_seed(args.seed, recording_id),
        "rir_room_index": room_index,
        "rir_by_speaker": rirs,
        "noise": noise_meta,
        "speech_level_reference": level_source,
        "headroom_gain": round(headroom_gain, 8),
        "rendered_duration": round(length / SR, 6),
    }
    with open(manifest_dir / f"{recording_id}_acoustics.json", "w") as f:
        json.dump(acoustic_meta, f, indent=2)
    return recording_id, length / SR, time.perf_counter() - started


def _load_input(source: Path):
    """Source recordings and their supervisions grouped by recording id."""
    manifest_dir = source / "manifests"
    recordings_path = manifest_dir / "tts-recordings.jsonl.gz"
    supervisions_path = manifest_dir / "tts-supervisions.jsonl.gz"
    if not recordings_path.is_file() or not supervisions_path.is_file():
        raise FileNotFoundError(
            "Expected manifests/tts-recordings.jsonl.gz and "
            "manifests/tts-supervisions.jsonl.gz under --source")
    recordings = list(RecordingSet.from_file(recordings_path))
    by_recording: dict[str, list] = {recording.id: [] for recording in recordings}
    for supervision in SupervisionSet.from_file(supervisions_path):
        if supervision.recording_id in by_recording:
            by_recording[supervision.recording_id].append(supervision)
    missing = [recording.id for recording in recordings if not by_recording[recording.id]]
    if missing:
        raise RuntimeError(f"No supervisions for {len(missing)} recording(s), e.g. {missing[0]}")
    return recordings, by_recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed dry TTS dataset directory.")
    parser.add_argument("--out", required=True, help="New directory for this acoustic variant.")
    parser.add_argument("--reverberate", action="store_true", help="Apply simulated FastMSS RIRs.")
    parser.add_argument("--add-noise", action="store_true", help="Apply real noise or Gaussian noise.")
    parser.add_argument("--noise-folder", default=None,
                        help="Folder recursively scanned for 16-kHz WAV/FLAC noise. Omit for Gaussian noise.")
    parser.add_argument("--noise-rel-gain", type=float, nargs=2, default=[-20.0, -3.0],
                        metavar=("LOW_DB", "HIGH_DB"),
                        help="Real-noise offset from the minimum speech level (FastMSS default: -20 -3).")
    parser.add_argument("--gaussian-noise-range", type=float, nargs=2, default=[-30.0, 3.0],
                        metavar=("LOW_DB", "HIGH_DB"),
                        help="Gaussian-noise offset range (TTS FastMSS path default: -30 3).")
    parser.add_argument("--single-noise", dest="multi_noise_sampling", action="store_false",
                        help="Use one repeated noise file instead of FastMSS multi-noise crossfading.")
    parser.set_defaults(multi_noise_sampling=True)
    parser.add_argument("--save-noise", action="store_true",
                        help="Also save the final, headroom-scaled noise waveform per meeting.")
    parser.add_argument("--rir-rooms", type=int, default=2,
                        help="Number of RIR rooms to sample (FastMSS staged default: 2).")
    parser.add_argument("--rir-positions", type=int, default=6,
                        help="RIR positions per room (FastMSS staged default: 6).")
    parser.add_argument("--rir-room-size", type=float, nargs=2, default=[5.0, 8.0],
                        metavar=("MIN_M", "MAX_M"),
                        help="Room length/width sampling bounds in metres "
                             "(FastMSS staged default: 5 8).")
    parser.add_argument("--rir-room-ceiling", type=float, nargs=2, default=[2.7, 3.5],
                        metavar=("MIN_M", "MAX_M"),
                        help="Room ceiling-height sampling bounds in metres "
                             "(FastMSS staged default: 2.7 3.5).")
    parser.add_argument("--rir-rt60", type=float, nargs=2, default=[0.2, 0.6],
                        metavar=("MIN_S", "MAX_S"),
                        help="RT60 sampling bounds in seconds "
                             "(FastMSS staged default: 0.2 0.6).")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CPU meetings.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N meetings.")
    parser.add_argument("--resume", action="store_true", help="Skip complete augmented meetings.")
    args = parser.parse_args()

    if not args.reverberate and not args.add_noise:
        parser.error("choose at least one of --reverberate or --add-noise")
    if args.rir_rooms <= 0 or args.rir_positions <= 0 or args.jobs <= 0:
        parser.error("--rir-rooms, --rir-positions, and --jobs must be positive")
    for option, values in (("--rir-room-size", args.rir_room_size),
                           ("--rir-room-ceiling", args.rir_room_ceiling),
                           ("--rir-rt60", args.rir_rt60)):
        if values[0] <= 0 or values[0] > values[1]:
            parser.error(f"{option} requires 0 < MIN <= MAX")
    if args.rir_room_size[0] <= 1.0:
        parser.error("--rir-room-size MIN_M must exceed 1 m for source placement")
    if args.rir_room_ceiling[0] <= 2.0:
        parser.error("--rir-room-ceiling MIN_M must exceed 2 m for source placement")
    if args.noise_rel_gain[0] > args.noise_rel_gain[1]:
        parser.error("--noise-rel-gain LOW_DB cannot exceed HIGH_DB")
    if args.gaussian_noise_range[0] > args.gaussian_noise_range[1]:
        parser.error("--gaussian-noise-range LOW_DB cannot exceed HIGH_DB")

    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if source == out:
        parser.error("--out must be a new variant directory, not --source")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"{out} already exists and is not empty; choose another --out or use --resume")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    args.noise_files = []
    if args.add_noise and args.noise_folder:
        folder = Path(args.noise_folder)
        args.noise_files = sorted(str(path.resolve()) for extension in ("*.wav", "*.flac")
                                  for path in folder.rglob(extension))
        if not args.noise_files:
            raise FileNotFoundError(f"No WAV/FLAC files found under {folder}")
        print(f"noise: {len(args.noise_files)} file(s) from {folder}")
    elif args.add_noise:
        print("noise: Gaussian (no --noise-folder supplied)")

    recordings, by_recording = _load_input(source)
    if args.limit is not None:
        recordings = recordings[:args.limit]
    if not recordings:
        raise RuntimeError("No recordings selected")
    _copy_text_metadata(source, out)
    rir_rooms = _prepare_rirs(args, out)

    # Everything that changes the output; compared on --resume.
    global_meta = {
        "source": str(source),
        "seed": args.seed,
        "reverberate": args.reverberate,
        "add_noise": args.add_noise,
        "noise_folder": str(Path(args.noise_folder).resolve()) if args.noise_folder else None,
        "noise_rel_gain": args.noise_rel_gain,
        "gaussian_noise_range": args.gaussian_noise_range,
        "noise_catalog_sha256": hashlib.sha256("\n".join(args.noise_files).encode()).hexdigest()
        if args.noise_files else None,
        "multi_noise_sampling": args.multi_noise_sampling,
        "rir_rooms": args.rir_rooms,
        "rir_positions": args.rir_positions,
        "rir_room_size": args.rir_room_size,
        "rir_room_ceiling": args.rir_room_ceiling,
        "rir_rt60": args.rir_rt60,
        "save_noise": args.save_noise,
    }
    metadata_path = out / "manifests" / "augmentation.json"
    if args.resume and metadata_path.is_file():
        previous = json.load(open(metadata_path))
        if previous != global_meta:
            changed = sorted(key for key in set(previous) | set(global_meta)
                             if key not in previous or key not in global_meta
                             or previous[key] != global_meta[key])
            raise RuntimeError(
                "Refusing to resume with different acoustic settings: "
                + ", ".join(changed))
    with open(metadata_path, "w") as f:
        json.dump(global_meta, f, indent=2)

    pending = []
    for recording in recordings:
        speakers = sorted({s.speaker for s in by_recording[recording.id]})
        if args.resume and _output_complete(out, recording.id, speakers):
            print(f"[{recording.id}] exists, skipping")
        else:
            pending.append(recording)
    print(f"acoustic augmentation: {len(recordings) - len(pending)} complete, {len(pending)} pending")

    times = []
    wall_start = time.perf_counter()
    if args.jobs == 1:
        for recording in pending:
            recording_id, duration, elapsed = _augment_one(
                recording, by_recording[recording.id], args, source, out, rir_rooms)
            times.append(elapsed)
            print(f"[{recording_id}] {duration:.1f}s audio, {fmt_seconds(elapsed)}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs,
                                                   thread_name_prefix="acoustics") as executor:
            futures = [executor.submit(_augment_one, recording, by_recording[recording.id],
                                       args, source, out, rir_rooms)
                       for recording in pending]
            for future in concurrent.futures.as_completed(futures):
                recording_id, duration, elapsed = future.result()
                times.append(elapsed)
                print(f"[{recording_id}] {duration:.1f}s audio, {fmt_seconds(elapsed)}")

    combine_meeting_manifests(out / "manifests", [recording.id for recording in recordings])
    wall = time.perf_counter() - wall_start
    if times:
        print(f"OK: {len(recordings)} meetings -> {out}\n"
              f"Acoustics: n={len(times)}, avg={fmt_seconds(sum(times) / len(times))}, "
              f"min={fmt_seconds(min(times))}, max={fmt_seconds(max(times))}, "
              f"wall={fmt_seconds(wall)}")
    else:
        print(f"OK: {len(recordings)} meetings -> {out} (no new meetings)")


if __name__ == "__main__":
    main()
