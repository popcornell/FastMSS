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

# No style table. CosyVoice 3's style channel is natural language, so the
# per-utterance instruction comes straight from the LLM. `speed` stays pinned at
# 1.0: it is post-hoc mel interpolation and artefacts audibly even at 0.92.
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


def find_env_python() -> str:
    """Interpreter for the separate CosyVoice env.

    $COSYVOICE_PYTHON if set, else conda's `cosyvoice` env under the conda root
    this process can see. Never sys.executable: the whole point of the worker is
    that its torch pin is incompatible with ours.
    """
    env = os.environ.get("COSYVOICE_PYTHON")
    if env:
        return env
    roots = []
    if os.environ.get("CONDA_EXE"):                  # <root>/bin/conda
        roots.append(Path(os.environ["CONDA_EXE"]).resolve().parents[1])
    roots += [Path.home() / "miniconda3", Path.home() / "anaconda3",
              Path.home() / "miniforge3"]
    for r in roots:
        cand = r / "envs" / "cosyvoice" / "bin" / "python"
        if cand.exists():
            return str(cand)
    return str(roots[0] / "envs" / "cosyvoice" / "bin" / "python") if roots else "python"


class CosyVoiceBackend:
    name = "cosyvoice"
    sr = 24000                                   # overwritten by the worker's report

    def __init__(self, cache_dir="clips_native", env_python=None, worker=None):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        # The worker ships with this package; the env it runs in does not.
        self.worker_path = str(worker or Path(__file__).with_name("cosyvoice_worker.py"))
        env_python = str(env_python or find_env_python())
        if not os.path.exists(env_python):
            raise RuntimeError(
                f"cosyvoice env python not found at {env_python}. Create the env, then\n"
                f"point COSYVOICE_PYTHON at its interpreter:\n"
                f"  conda create -n cosyvoice python=3.10 -y && conda activate cosyvoice\n"
                f"  conda install -c conda-forge pynini=2.1.6 -y\n"
                f"  pip install -r $COSYVOICE_REPO/requirements.txt\n"
                f"  export COSYVOICE_PYTHON=$(python -c 'import sys; print(sys.executable)')")
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

    def synth(self, text, instruct, level, speaker, persona) -> tuple[np.ndarray, str]:
        """persona must carry `path`: the speaker's LibriSpeech reference clip.

        Speaker identity is the reference clip. `zero_shot_spk_id` is NOT used:
        frontend_instruct2 routes instruct_text through the prompt_text slot, and
        the registered-speaker branch replaces the whole cached input, so an id
        would silently discard the instruction. The worker memoises the
        prompt-derived features by clip path instead, which is the same saving.
        """
        ref = persona.get("path")
        if not ref or not os.path.exists(ref):
            raise RuntimeError(f"no reference clip for {speaker}: {ref!r}")
        instruct = (instruct or "").strip() or "Speak naturally."
        effort = LEVELS.get(level, LEVELS["normal"])[2]
        if effort:
            instruct = f"{instruct.rstrip('.')}, {effort}."
        style_db = 0.0

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
