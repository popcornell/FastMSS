"""Synthetic source data shared by the tests (no external corpora needed)."""

from pathlib import Path

import numpy as np
import soundfile as sf
from lhotse import MonoCut, Recording
from lhotse.supervision import AlignmentItem, SupervisionSegment
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
FS = 16000


def speech_like(n, rng, fs=FS):
    """Band-limited noise with a syllable-rate amplitude envelope."""
    x = rng.standard_normal(n)
    # crude 300-3400 Hz band-pass via FFT masking
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < 300) | (f > 3400)] = 0
    x = np.fft.irfft(X, n)
    t = np.arange(n) / fs
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))
    x = x * env
    return (0.1 * x / (np.sqrt(np.mean(x**2)) + 1e-8)).astype(np.float32)


def make_spk2cuts(root, n_spk=4, n_utt=6, seed=0):
    rng = np.random.default_rng(seed)
    audio_dir = Path(root) / "src_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    spk2cuts = {}
    for s in range(n_spk):
        spk = f"spk{s}"
        spk2cuts[spk] = []
        for u in range(n_utt):
            dur = float(rng.uniform(0.8, 2.5))
            n = int(dur * FS)
            path = audio_dir / f"{spk}_{u}.wav"
            sf.write(path, speech_like(n, rng), FS)
            rec = Recording.from_file(path, recording_id=f"{spk}_{u}")
            n_words = 3
            wdur = rec.duration / n_words
            words = [
                AlignmentItem(symbol=f"w{k}", start=k * wdur, duration=wdur * 0.9)
                for k in range(n_words)
            ]
            sup = SupervisionSegment(
                id=f"{spk}_{u}",
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                channel=0,
                speaker=spk,
                text=" ".join(w.symbol for w in words),
                alignment={"word": words},
            )
            spk2cuts[spk].append(
                MonoCut(
                    id=f"{spk}_{u}",
                    start=0.0,
                    duration=rec.duration,
                    channel=0,
                    recording=rec,
                    supervisions=[sup],
                )
            )
    return spk2cuts


def make_noise_files(root, n=3, dur=6.0, seed=1):
    rng = np.random.default_rng(seed)
    d = Path(root) / "noise"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        p = d / f"noise{i}.wav"
        sf.write(p, (0.05 * rng.standard_normal(int(dur * FS))).astype(np.float32), FS)
        files.append(str(p))
    return files


def base_cfg(output_dir, **overrides):
    cfg = OmegaConf.load(REPO / "recipes" / "default.yaml")
    cfg.pop("hydra", None)
    cfg.output_dir = str(output_dir)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg
