"""Stage 3 -- TTS. CosyVoice 3 only, cloning from LibriSpeech reference clips.

CosyVoice pins torch 2.3.1 / numpy 1.26, which would downgrade this environment,
so it runs as a persistent worker in its own conda env and we talk to it over a
JSON-line pipe. The model loads once per run, not once per utterance.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_VERSION = 2              # bump whenever the cache key or clip format changes
DEFAULT_SYNTH_RETRIES = 3      # total attempts per utterance, not extra retries

# One lock per cache key, so parallel TTS threads (tts_staged.py --tts-jobs)
# never render the same clip twice but still synthesise different clips at once.
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()

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

# --- short-text workarounds -------------------------------------------------
# CosyVoice's encoder convolutions fail ("Kernel size can't be greater than
# actual input size") or return zero samples when a line is only one or two
# phonemes long, which is exactly what backchannels are. The tables below give
# the synthesiser a slightly longer, still natural, string. They change only
# what is *spoken*; the transcript written to the manifests keeps the LLM text.

# Retry text for murmurs that failed on the first attempt.
SHORT_TEXT_RETRY = {
    "hm": "hmm.",
    "hmm": "hmm.",
    "mm": "mmm.",
    "mhm": "mmm.",
    "uh": "uhh.",
    "um": "umm.",
    "oh": "ohh.",
    "ah": "ahh.",
}
# Words known to fail every time; expanded up front, before the first attempt.
SHORT_TEXT_EXPANSIONS = {
    "broken": "broken, really.",
}
# Error substrings that identify the too-short-input failure mode.
SHORT_SEQUENCE_ERRORS = (
    "Kernel size can't be greater than actual input size",
    "Calculated padded input size per channel",
    "CosyVoice produced zero audio samples",
)
_SPOKEN_TEXT = re.compile(r"[A-Za-z0-9]")


def _short_text_retry(text: str) -> str | None:
    """Longer retry string for a known murmur or any <=3-letter word, else None."""
    clean = text.strip().strip(".,!?;:\"'()[]{} ").lower()
    if clean in SHORT_TEXT_RETRY:
        return SHORT_TEXT_RETRY[clean]
    if clean.isalpha() and len(clean) <= 3:
        return f"{clean}, {clean}."
    return None


def _expanded_short_sequence_retry(text: str) -> str:
    """Give CosyVoice's phoneme encoder enough context after a short-sequence error.

    A one-word English turn may expand to fewer than four internal frames even
    when it has more than three letters (for example, ``broken``). The original
    text remains in the transcript; this expansion is synthesis-only.
    """
    special = _short_text_retry(text)
    if special is not None and special != text:
        return special
    core = str(text).strip().rstrip(".,!?;:")
    return f"{core}, really." if core else "uh, really."


def _is_short_sequence_error(exc: Exception) -> bool:
    detail = str(exc)
    return any(marker in detail for marker in SHORT_SEQUENCE_ERRORS)


def tts_safe_text(text: str) -> str:
    """Return a speakable form for punctuation-only and known short turns."""
    text = str(text or "").strip()
    if not _SPOKEN_TEXT.search(text):
        return "uh."
    clean = text.strip(".,!?;:\"'()[]{} ").lower()
    return SHORT_TEXT_EXPANSIONS.get(clean, text)


def tts_safe_instruct(instruct: str) -> str:
    """Instruction with at least one spoken character; punctuation-only is invalid."""
    instruct = str(instruct or "").strip()
    return instruct if _SPOKEN_TEXT.search(instruct) else "Hesitate briefly."


def _cache_lock(key: str) -> threading.Lock:
    """Serialize identical cache misses without serializing independent TTS jobs."""
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.Lock())


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

    def __init__(self, cache_dir="clips_native", env_python=None, worker=None,
                 retries=DEFAULT_SYNTH_RETRIES):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.retries = max(1, int(retries))
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
        self.env_python = env_python
        self._stderr_tail = deque(maxlen=300)
        self._start_worker()

    def _start_worker(self) -> None:
        """Spawn the worker and block until it reports the model is loaded."""
        self.proc = subprocess.Popen(
            [self.env_python, "-u", self.worker_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=os.path.dirname(self.worker_path))
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self.proc.stderr,), daemon=True)
        self._stderr_thread.start()
        self.sr = int(json.loads(self._readline("model load"))["sample_rate"])

    def _drain_stderr(self, stream):
        # Drained on a thread: a full stderr pipe would otherwise block the
        # worker mid-synthesis. Only the tail is kept, for error messages.
        for line in stream:
            self._stderr_tail.append(line)

    def _restart_worker(self) -> None:
        """Replace a crashed worker before retrying the current synthesis job."""
        old = self.proc
        try:
            if old.poll() is None:
                old.terminate()
                old.wait(timeout=5)
        except Exception:
            try:
                old.kill()
                old.wait(timeout=5)
            except Exception:
                pass
        self._start_worker()

    def _worker_unusable(self, exc: Exception) -> bool:
        """True if the failure killed the worker (vs. a per-request error)."""
        return (self.proc.poll() is not None
                or isinstance(exc, BrokenPipeError)
                or "cosyvoice worker died during" in str(exc))

    def _stderr_text(self) -> str:
        return "".join(self._stderr_tail)[-3000:]

    def _readline(self, what=""):
        """Next protocol line, skipping library chatter that reaches stdout."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"cosyvoice worker died during {what}:\n{self._stderr_text()}")
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return line

    def _read_cached(self, path: Path) -> np.ndarray | None:
        """Cached clip, or None if missing, empty or unreadable (e.g. a killed write)."""
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            wav, _ = sf.read(path, dtype="float32")
        except Exception:
            return None
        return wav if wav.size else None

    def synth(self, text, instruct, level, speaker, persona) -> tuple[np.ndarray, str]:
        """persona must carry `path`: the speaker's LibriSpeech reference clip.

        Returns (wav, cache_path). Clips are cached under a hash of
        (text, instruction, reference clip), so resumed runs never re-render.

        Failure handling: each request is attempted up to `self.retries` times.
        A crashed worker is restarted between attempts, and after the first
        failure a too-short line is swapped for a longer synthesis-only string
        (see SHORT_TEXT_RETRY).

        Speaker identity is the reference clip. `zero_shot_spk_id` is NOT used:
        frontend_instruct2 routes instruct_text through the prompt_text slot, and
        the registered-speaker branch replaces the whole cached input, so an id
        would silently discard the instruction. The worker memoises the
        prompt-derived features by clip path instead, which is the same saving.
        """
        ref = persona.get("path")
        if not ref or not os.path.exists(ref):
            raise RuntimeError(f"no reference clip for {speaker}: {ref!r}")
        text = tts_safe_text(text)
        instruct = tts_safe_instruct(instruct)
        effort = LEVELS.get(level, LEVELS["normal"])[2]
        if effort:
            instruct = f"{instruct.rstrip('.')}, {effort}."
        style_db = 0.0

        key = "|".join([str(CACHE_VERSION), self.name, text, instruct, ref])
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        path = self.cache / f"{self.name}_{h}.wav"
        wav = self._read_cached(path)
        if wav is not None:
            return wav, str(path)
        if path.exists():                            # corrupt leftover from a crash
            path.unlink()

        with _cache_lock(h):
            # another thread may have rendered this key while we waited
            wav = self._read_cached(path)
            if wav is not None:
                return wav, str(path)

            errors = []
            synth_text = text
            retry_text = _short_text_retry(text)
            for attempt in range(1, self.retries + 1):
                # A unique output path prevents stale/partial files and makes a
                # protocol response unambiguously belong to this request.
                tmp = self.cache / (
                    f"_tmp_{h}_{os.getpid()}_{threading.get_ident()}_{attempt}.wav")
                tmp.unlink(missing_ok=True)
                job = {"text": synth_text, "instruct": instruct, "speed": 1.0,
                       "prompt_wav": os.path.abspath(ref), "out": os.path.abspath(tmp)}
                try:
                    self.proc.stdin.write(json.dumps(job) + "\n")
                    self.proc.stdin.flush()
                    resp = json.loads(self._readline("synthesis"))
                    if not resp.get("ok"):
                        raise RuntimeError(
                            f"worker error: {resp.get('error')}\n{resp.get('trace', '')}")
                    if os.path.abspath(resp.get("path", "")) != os.path.abspath(tmp):
                        raise RuntimeError(f"worker returned unexpected output path: {resp.get('path')!r}")
                    wav = self._read_cached(tmp)
                    if wav is None:
                        raise RuntimeError("worker produced an empty or unreadable wav")
                    break
                except Exception as exc:
                    errors.append(f"attempt {attempt}/{self.retries}: {exc}")
                    tmp.unlink(missing_ok=True)
                    if self._worker_unusable(exc) and attempt < self.retries:
                        print(f"cosyvoice worker restart after failed synthesis: {exc}")
                        try:
                            self._restart_worker()
                        except Exception as restart_exc:
                            errors.append(f"worker restart failed: {restart_exc}")
                    if attempt == 1:                 # switch to a longer string once
                        if retry_text:
                            synth_text = retry_text
                        elif _is_short_sequence_error(exc):
                            synth_text = _expanded_short_sequence_retry(text)
                            print(f"cosyvoice expanded short synthesis text: "
                                  f"{text!r} -> {synth_text!r}")
                    if attempt < self.retries:
                        print(f"cosyvoice retry {attempt}/{self.retries - 1} for {text!r}: {exc}")
                        time.sleep(0.1)
            else:                                    # every attempt failed
                stderr = self._stderr_text().strip()
                detail = " | ".join(errors)
                if stderr:
                    detail += f"\nworker stderr:\n{stderr}"
                raise RuntimeError(f"cosyvoice failed for {text!r} after {self.retries} attempts: {detail}")

            tmp.unlink(missing_ok=True)
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
