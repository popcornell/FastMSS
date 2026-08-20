"""LibriSpeech reference-voice bank for cloning backends.

Voice-first (INTEGRATION.md 5): draw the speakers and their sexes FIRST, then let
the LLM write personas to fit. The reverse order bakes the author model's
role-gender prior into the corpus and correlates it with the acoustic voice.

Sex comes from LibriSpeech's own SPEAKERS.TXT -- free, no inference.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_S, MAX_S = 4.0, 8.0          # inference_instruct2 wants a short, clean prompt
PEAK_LO, PEAK_HI = 0.05, 0.999   # too quiet / clipped


def speaker_sex(root: str | Path) -> dict[str, str]:
    out = {}
    for line in open(Path(root) / "SPEAKERS.TXT"):
        if line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            out[parts[0]] = parts[1]
    return out


def transcript_for(flac: Path) -> str | None:
    trans = flac.parent / f"{flac.parent.parent.name}-{flac.parent.name}.trans.txt"
    if not trans.exists():
        return None
    for line in open(trans):
        uid, _, text = line.partition(" ")
        if uid == flac.stem:
            return text.strip()
    return None


def pick_clip(spk_dir: Path) -> tuple[Path, float, str] | None:
    for flac in sorted(spk_dir.rglob("*.flac")):
        info = sf.SoundFile(str(flac))
        dur = len(info) / info.samplerate
        if not (MIN_S <= dur <= MAX_S):
            continue
        w, _ = sf.read(str(flac), dtype="float32")
        peak = float(np.abs(w).max())
        if not (PEAK_LO < peak < PEAK_HI):
            continue
        txt = transcript_for(flac)
        if txt:
            return flac, dur, txt
    return None


def build_bank(root: str | Path, splits=("train-clean-100",), limit=200,
               cache: str | Path = "voice_prompts/bank.json") -> list[dict]:
    """Scan LibriSpeech once -> [{ls_id, sex, path, duration, transcript}]."""
    cache = Path(cache)
    if cache.exists():
        return json.load(open(cache))
    root = Path(root)
    sex = speaker_sex(root)
    bank = []
    for split in splits:
        for spk_dir in sorted((root / split).iterdir()):
            if not spk_dir.is_dir():
                continue
            got = pick_clip(spk_dir)
            if got is None:
                continue
            flac, dur, txt = got
            bank.append({"ls_id": spk_dir.name, "sex": sex.get(spk_dir.name, "?"),
                         "path": str(flac), "duration": round(dur, 2),
                         "transcript": txt})
            if len(bank) >= limit:
                break
        if len(bank) >= limit:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(bank, open(cache, "w"), indent=1)
    return bank


def cast_meeting(bank: list[dict], n_speakers: int, seed: int = 0) -> dict[str, dict]:
    """Draw a sex composition, then the voices. No speaker twice in a meeting."""
    rng = random.Random(seed)
    males = [b for b in bank if b["sex"] == "M"]
    females = [b for b in bank if b["sex"] == "F"]
    n_m = rng.randint(0, n_speakers)                    # all-M and all-F are real cases
    n_m = min(n_m, len(males)); n_f = n_speakers - n_m
    if n_f > len(females):
        n_f = len(females); n_m = n_speakers - n_f
    chosen = rng.sample(males, n_m) + rng.sample(females, n_f)
    rng.shuffle(chosen)
    return {f"spk_{chr(65+i)}": dict(c) for i, c in enumerate(chosen)}
