#!/usr/bin/env python3
"""Pseudo-label SSSD / otoSpeech / Apptek per-speaker streams with NeMo Parakeet v3.

Produces a Lhotse cutset at <output_dir>/<prefix>_<split>.jsonl.gz with single-
speaker word-level alignments, compatible with FastMSS's simulator.

Requires NeMo: `pip install -e .[pseudo_label]`
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.supervision import AlignmentItem
from tqdm import tqdm

logger = logging.getLogger("pseudo_label_parakeet")

PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SR = 16000


@dataclass(frozen=True)
class StreamSpec:
    """A single per-speaker audio stream to transcribe."""

    stream_id: str
    recording_id: str
    speaker_id: str
    audio_path: Path
    channel: Optional[int] = None  # None = mono file; 0/1 = slice from stereo


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def enumerate_sssd(input_dir: Path, args: argparse.Namespace) -> Iterable[StreamSpec]:
    """SSSD: stereo full-duplex `<datetime>_<spkA>_<spkB>_<id>_mixed.{flac,wav}`.
    Yields 2 channel-split streams per file with speaker IDs from the filename."""
    glob = args.audio_glob or "**/*.flac"
    regex = re.compile(
        args.filename_regex
        or r"(?P<session>(?P<dt>\d+_\d+)_(?P<spkA>[A-Za-z0-9]+)_(?P<spkB>[A-Za-z0-9]+)_(?P<sid>[A-Za-z0-9]+))_mixed\.(?:wav|flac)$"
    )
    parse_failures = 0
    for path in sorted(input_dir.glob(glob)):
        m = regex.search(path.name)
        if not m:
            if parse_failures < 3:
                logger.warning(
                    "SSSD regex did not match: %s (regex=%s)",
                    path.name,
                    regex.pattern,
                )
            parse_failures += 1
            continue
        session = m.group("session")
        speakers = (m.group("spkA"), m.group("spkB"))
        try:
            info = sf.info(str(path))
        except Exception as e:
            logger.warning("Failed to probe %s: %s", path, e)
            continue
        n_ch = info.channels
        for ch, spk in zip(range(min(n_ch, 2)), speakers):
            yield StreamSpec(
                stream_id=f"{session}_{spk}",
                recording_id=session,
                speaker_id=spk,
                audio_path=path,
                channel=ch if n_ch > 1 else None,
            )
        if n_ch == 1:
            logger.warning(
                "SSSD file is mono (expected stereo full-duplex): %s", path
            )
    if parse_failures:
        logger.warning("SSSD: %d filenames did not match regex.", parse_failures)


def enumerate_otospeech(
    input_dir: Path, args: argparse.Namespace
) -> Iterable[StreamSpec]:
    glob = args.audio_glob or "*.flac"
    for path in sorted(input_dir.glob(glob)):
        try:
            info = sf.info(str(path))
        except Exception as e:
            logger.warning("Failed to probe %s: %s", path, e)
            continue
        stem = path.stem
        if info.channels == 1:
            yield StreamSpec(
                stream_id=stem,
                recording_id=stem,
                speaker_id=f"{stem}_ch0",
                audio_path=path,
                channel=None,
            )
        else:
            for ch in range(min(info.channels, 2)):
                yield StreamSpec(
                    stream_id=f"{stem}_ch{ch}",
                    recording_id=stem,
                    speaker_id=f"{stem}_ch{ch}",
                    audio_path=path,
                    channel=ch,
                )


def enumerate_apptek(
    input_dir: Path, args: argparse.Namespace
) -> Iterable[StreamSpec]:
    glob = args.audio_glob or "**/audio/*_channel?.wav"
    accent_filter = args.accent
    rx = re.compile(r"^(?P<call>.+)_channel(?P<ch>[12])\.wav$")
    for path in sorted(input_dir.glob(glob)):
        if accent_filter and accent_filter not in path.parts:
            continue
        m = rx.match(path.name)
        if not m:
            continue
        call_id = m.group("call")
        ch_n = m.group("ch")
        stream_id = f"{call_id}_ch{ch_n}"
        yield StreamSpec(
            stream_id=stream_id,
            recording_id=call_id,
            speaker_id=stream_id,
            audio_path=path,
            channel=None,
        )


ADAPTERS: dict[str, Callable[[Path, argparse.Namespace], Iterable[StreamSpec]]] = {
    "sssd": enumerate_sssd,
    "otospeech": enumerate_otospeech,
    "apptek": enumerate_apptek,
}


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #


def load_audio(spec: StreamSpec, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load a per-speaker mono stream as a float32 numpy array at target_sr."""
    audio, sr = sf.read(str(spec.audio_path), always_2d=True)  # (n, channels)
    if spec.channel is not None:
        audio = audio[:, spec.channel]
    else:
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    audio = audio.astype(np.float32, copy=False)

    if sr != target_sr:
        import torch
        import torchaudio.functional as F

        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio = (
            F.resample(audio_t, orig_freq=sr, new_freq=target_sr)
            .squeeze(0)
            .numpy()
        )
        sr = target_sr
    return audio, sr


def maybe_materialize_mono(
    spec: StreamSpec, audio: np.ndarray, sr: int, cache_dir: Path
) -> Path:
    """For otoSpeech stereo splits, persist per-channel mono FLAC so the
    resulting Recording is unambiguously single-channel. For mono sources,
    return the original path."""
    if spec.channel is None:
        return spec.audio_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{spec.stream_id}.flac"
    if not out.exists():
        sf.write(str(out), audio, sr, subtype="PCM_16", format="FLAC")
    return out


# --------------------------------------------------------------------------- #
# Parakeet inference
# --------------------------------------------------------------------------- #


def load_parakeet(device: str):
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as e:
        sys.stderr.write(
            "ERROR: NeMo is required for Parakeet inference.\n"
            "Install with:  pip install -e .[pseudo_label]\n"
            f"(Underlying ImportError: {e})\n"
        )
        sys.exit(1)

    import nemo

    logger.info("NeMo version: %s", getattr(nemo, "__version__", "unknown"))
    logger.info("Loading %s ...", PARAKEET_MODEL_ID)
    model = ASRModel.from_pretrained(PARAKEET_MODEL_ID)
    model.eval()
    if device == "cuda":
        model = model.cuda()
    return model


def _extract_word_timestamps(hyp) -> Optional[List[dict]]:
    """Return a list of {'word','start','end'} dicts from a NeMo hypothesis,
    probing the few field names NeMo has used across versions."""
    ts = getattr(hyp, "timestamp", None)
    if isinstance(ts, dict) and "word" in ts and ts["word"]:
        entries = ts["word"]
        out = []
        for w in entries:
            if isinstance(w, dict):
                word = w.get("word") or w.get("char") or w.get("symbol")
                start = w.get("start") if "start" in w else w.get("start_offset")
                end = w.get("end") if "end" in w else w.get("end_offset")
                if word is None or start is None or end is None:
                    return None
                out.append({"word": str(word), "start": float(start), "end": float(end)})
        return out
    wts = getattr(hyp, "word_timestamps", None)
    if wts:
        return [
            {"word": str(w["word"]), "start": float(w["start"]), "end": float(w["end"])}
            for w in wts
        ]
    return None


def transcribe_batch(model, audios: List[np.ndarray], batch_size: int):
    """Run model.transcribe with timestamps, returning a list parallel to audios
    of [{'word','start','end'}, ...] lists (or None for failures)."""
    try:
        hyps = model.transcribe(
            audio=audios, batch_size=batch_size, timestamps=True
        )
    except TypeError:
        try:
            model.change_decoding_strategy(
                decoding_cfg={"compute_timestamps": True, "preserve_alignments": True}
            )
        except Exception:
            pass
        hyps = model.transcribe(audio=audios, batch_size=batch_size)

    # transcribe() may return (best_hyps, all_hyps) tuple in some NeMo versions
    if isinstance(hyps, tuple):
        hyps = hyps[0]

    out = []
    for h in hyps:
        if isinstance(h, str):
            out.append(None)
            continue
        out.append(_extract_word_timestamps(h))
    return out


# --------------------------------------------------------------------------- #
# Manifest assembly
# --------------------------------------------------------------------------- #


_PUNCT_ONLY = re.compile(r"^\W+$")


def _build_alignment_items(
    words: List[dict], audio_duration: float
) -> List[AlignmentItem]:
    items = []
    for w in words:
        sym = w["word"].strip()
        if not sym or _PUNCT_ONLY.match(sym):
            continue
        start = max(0.0, float(w["start"]))
        end = min(audio_duration, float(w["end"]))
        if end <= start:
            continue
        items.append(AlignmentItem(symbol=sym, start=start, duration=end - start))
    items.sort(key=lambda it: it.start)
    return items


def build_recording_and_supervision(
    spec: StreamSpec,
    audio_path: Path,
    words: List[dict],
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    rec = Recording.from_file(audio_path, recording_id=spec.stream_id)
    items = _build_alignment_items(words, rec.duration)
    if not items:
        return None
    sup = SupervisionSegment(
        id=f"{spec.stream_id}_sup",
        recording_id=spec.stream_id,
        start=0.0,
        duration=rec.duration,
        channel=0,
        speaker=spec.speaker_id,
        text=" ".join(it.symbol for it in items),
        alignment={"word": items},
    )
    return rec, sup


def write_cutset(
    recs: List[Recording],
    sups: List[SupervisionSegment],
    output_path: Path,
) -> None:
    import tempfile

    cutset = CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recs),
        supervisions=SupervisionSet.from_segments(sups),
    )
    # Use a tempfile in the same dir that preserves the full extension —
    # lhotse decides gzip vs plain JSONL from the path suffix.
    suffix = "".join(output_path.suffixes)  # e.g. ".jsonl.gz"
    fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=".partial-", suffix=suffix
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        cutset.to_file(tmp)
        tmp.replace(output_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_existing_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids = set()
    opener = gzip.open if output_path.suffix == ".gz" else open
    with opener(output_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(ADAPTERS.keys()))
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--prefix", default=None, help="Default: matches --dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_audio_workers", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--audio_glob", default=None)
    p.add_argument(
        "--filename_regex",
        default=None,
        help="SSSD only; default is <session>-<spk>-<idx>.wav",
    )
    p.add_argument("--accent", default=None, help="Apptek only; e.g. en_US")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Enumerate streams and exit without loading the model.",
    )
    return p.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return choice


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    if not args.input_dir.exists():
        logger.error("Input dir does not exist: %s", args.input_dir)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or args.dataset
    output_path = args.output_dir / f"{prefix}_{args.split}.jsonl.gz"

    logger.info("Enumerating %s streams under %s ...", args.dataset, args.input_dir)
    specs = list(ADAPTERS[args.dataset](args.input_dir, args))
    logger.info("Found %d streams.", len(specs))

    if not specs:
        logger.error("No streams found. Check --input_dir / --audio_glob.")
        return 1

    if args.limit:
        specs = specs[: args.limit]
        logger.info("Limited to %d streams.", len(specs))

    if args.dry_run:
        for spec in specs[:8]:
            logger.info("DRY: %s", spec)
        logger.info("Dry run complete (%d total).", len(specs))
        return 0

    existing_ids = set() if args.overwrite else load_existing_ids(output_path)
    if existing_ids:
        before = len(specs)
        specs = [s for s in specs if s.stream_id not in existing_ids]
        logger.info(
            "Resume: %d/%d streams already in %s; %d remaining.",
            before - len(specs),
            before,
            output_path,
            len(specs),
        )
        if not specs:
            logger.info("Nothing to do.")
            return 0

    device = resolve_device(args.device)
    if device == "cpu":
        logger.warning(
            "Running on CPU. Parakeet 0.6B inference will be very slow; "
            "expect roughly 5-10x the audio duration. Use a GPU for real workloads."
        )

    model = load_parakeet(device)

    cache_dir = args.output_dir / "audio_streams"
    recs: List[Recording] = []
    sups: List[SupervisionSegment] = []

    bs = args.batch_size
    pbar = tqdm(total=len(specs), desc="Transcribing")
    i = 0
    first_batch_logged = False
    while i < len(specs):
        batch_specs = specs[i : i + bs]
        i += bs
        audios = []
        for spec in batch_specs:
            try:
                audio, _ = load_audio(spec, target_sr=TARGET_SR)
                audios.append(audio)
            except Exception as e:
                logger.warning("Failed to load %s: %s", spec.audio_path, e)
                audios.append(None)

        valid_pairs = [
            (s, a) for s, a in zip(batch_specs, audios) if a is not None
        ]
        if not valid_pairs:
            pbar.update(len(batch_specs))
            continue
        valid_audios = [a for _, a in valid_pairs]

        word_lists = transcribe_batch(model, valid_audios, batch_size=bs)

        if not first_batch_logged:
            first_batch_logged = True
            sample = next((wl for wl in word_lists if wl), None)
            if sample is None:
                logger.error(
                    "Parakeet returned no word-level timestamps for the first batch. "
                    "Check NeMo version: ASRModel.transcribe(timestamps=True) is "
                    "required (NeMo >= 2.0). Exiting before writing alignment-less cuts."
                )
                return 2
            logger.info(
                "First-batch sample word entry: %s",
                {k: type(v).__name__ for k, v in sample[0].items()},
            )

        for (spec, audio), words in zip(valid_pairs, word_lists):
            if not words:
                continue
            audio_path = maybe_materialize_mono(
                spec, audio, TARGET_SR, cache_dir
            )
            result = build_recording_and_supervision(spec, audio_path, words)
            if result is None:
                continue
            rec, sup = result
            recs.append(rec)
            sups.append(sup)

        pbar.update(len(batch_specs))
    pbar.close()

    if existing_ids:
        # Merge with previously written cuts so resume preserves them
        prev = CutSet.from_file(output_path)
        for cut in prev:
            recs.append(cut.recording)
            for s in cut.supervisions:
                sups.append(s)

    if not recs:
        logger.error("No successful transcriptions; refusing to write empty manifest.")
        return 3

    logger.info("Writing cutset with %d cuts to %s", len(recs), output_path)
    write_cutset(recs, sups, output_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
