"""Stage 3 -- TTS. CosyVoice 3 only, cloning from LibriSpeech reference clips.

CosyVoice pins torch 2.3.1 / numpy 1.26, which would downgrade this environment,
so it runs as a persistent worker in its own conda env and we talk to it over a
JSON-line pipe. The model loads once per run, not once per utterance.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_VERSION = 1

# style -> (natural-language instruction, gain dB).
# `speed` is deliberately absent: it is post-hoc mel interpolation and artefacts
# audibly even at 0.92. Rate differences live in the INSTRUCTION, which changes
# the generated tokens instead of resampling them.
COSY_STYLES = {
    "neutral":        ("Speak naturally, as in a relaxed work meeting.", 0.0),
    "engaged":        ("Speak with interest and energy, leaning into the point.", 0.5),
    "hesitant":       ("Speak hesitantly, unsure, with halting delivery.", -1.5),
    "urgent":         ("Speak quickly and urgently, cutting in.", 1.5),
    "insistent":      ("Speak firmly and insistently, pressing the point.", 1.5),
    "low_arousal":    ("Murmur quietly and briefly, barely engaged.", -6.0),
    "affirmative":    ("Murmur a short quiet agreement.", -5.0),
    "mildly_annoyed": ("Speak with slight irritation, clipped.", 0.5),
    "patient":        ("Speak calmly and patiently, unhurried.", -0.5),
}
# vocal effort is a generative property, not just amplitude: it goes into the
# instruction AND into a gain sampled from the level's range.
LEVELS = {
    "whisper": (-16.0, -11.0, "almost whispering, very little voice"),
    "soft":    (-9.0,  -5.0,  "quietly, low effort"),
    "normal":  (-2.0,   2.0,  ""),
    "raised":  (2.0,    5.0,  "with raised voice, more effort"),
    "loud":    (5.0,    8.0,  "loudly, projecting across the room"),
}


def level_gain_db(level: str, key: str) -> float:
    lo, hi, _ = LEVELS[level]
    h = int(hashlib.sha1(f"{key}|{level}".encode()).hexdigest()[:8], 16)
    return lo + (hi - lo) * (h / 0xFFFFFFFF)


def trim_silence(wav: np.ndarray, sr: int, thresh=0.02, pad_s=0.02) -> np.ndarray:
    """Deliberately low threshold: a rendered breath measures ~7% of peak."""
    if wav.size == 0:
        return wav
    idx = np.where(np.abs(wav) > thresh * np.abs(wav).max())[0]
    if idx.size == 0:
        return wav
    pad = int(pad_s * sr)
    return wav[max(0, idx[0] - pad): min(len(wav), idx[-1] + pad)]


class CosyVoiceBackend:
    name = "cosyvoice"
    sr = 24000                                   # overwritten by the worker's report

    def __init__(self, cache_dir="clips_native",
                 env_python="/Users/samco/miniconda3/envs/cosyvoice/bin/python",
                 worker=None, project_root=None):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        root = Path(project_root or Path(__file__).resolve().parents[3])
        self.worker_path = str(worker or root / "cosyvoice_worker.py")
        if not os.path.exists(env_python):
            raise RuntimeError(
                f"cosyvoice env python not found at {env_python}.\n"
                f"  conda create -n cosyvoice python=3.10 -y && conda activate cosyvoice\n"
                f"  conda install -c conda-forge pynini=2.1.6 -y\n"
                f"  pip install -r third_party/CosyVoice/requirements.txt")
        if not os.path.exists(self.worker_path):
            raise RuntimeError(f"cosyvoice_worker.py not found at {self.worker_path}")
        self.proc = subprocess.Popen(
            [env_python, "-u", self.worker_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=os.path.dirname(self.worker_path))
        self.sr = int(json.loads(self._readline("model load"))["sample_rate"])

    def _readline(self, what=""):
        """Next protocol line, skipping library chatter that reaches stdout."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read()[-3000:] if self.proc.stderr else ""
                raise RuntimeError(f"cosyvoice worker died during {what}:\n{err}")
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return line

    def synth(self, text, style, level, speaker, persona) -> tuple[np.ndarray, str]:
        """persona must carry `path`: the speaker's LibriSpeech reference clip."""
        ref = persona.get("path")
        if not ref or not os.path.exists(ref):
            raise RuntimeError(f"no reference clip for {speaker}: {ref!r}")
        instruct, style_db = COSY_STYLES.get(style, COSY_STYLES["neutral"])
        effort = LEVELS.get(level, LEVELS["normal"])[2]
        if effort:
            instruct = f"{instruct.rstrip('.')}, {effort}."

        key = "|".join([str(CACHE_VERSION), self.name, text, instruct, ref])
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        path = self.cache / f"{self.name}_{h}.wav"
        if not path.exists():
            tmp = self.cache / f"_tmp_{h}.wav"
            job = {"text": text, "instruct": instruct, "speed": 1.0,
                   "prompt_wav": os.path.abspath(ref), "out": os.path.abspath(tmp)}
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
            resp = json.loads(self._readline("synthesis"))
            if not resp.get("ok"):
                raise RuntimeError(f"cosyvoice worker error: {resp.get('error')}\n"
                                   f"{resp.get('trace', '')}")
            wav, _ = sf.read(resp["path"], dtype="float32")
            os.remove(resp["path"])
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = trim_silence(wav, self.sr)
            g = 10 ** ((style_db + level_gain_db(level, h)) / 20.0)
            # float, never clipped: a loud + Lombard line may legitimately exceed 1.0
            sf.write(path, (wav * g).astype("float32"), self.sr, subtype="FLOAT")
        wav, _ = sf.read(path, dtype="float32")
        return wav, str(path)

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"stop": True}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def get_backend(name: str, **kw):
    if name == "cosyvoice":
        return CosyVoiceBackend(**kw)
    raise ValueError(f"unknown backend {name!r} (available: cosyvoice)")
