"""Stage 4b -- acoustics, via FastMSS.

TTSMeetingSimulator subclasses ConversationalMeetingSimulator and reuses its
helpers unchanged (normalize_to, add_real_noise, add_gaussian_noise,
create_fir_highpass). Only gen_audio is reimplemented, because the TTS path must
diverge in two places that live mid-method upstream:

  simulator.py:625  normalize_to(audio, base+rel_gain) re-levels EVERY utterance
                    independently, erasing the whisper..loud channel and Lombard.
  simulator.py:585  np.random.choice(room_rirs) picks a fresh RIR per utterance,
                    teleporting a speaker around the room between turns.

Nothing upstream is edited.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from lhotse import AudioSource, Recording
from lhotse.supervision import SupervisionSegment
from lhotse.utils import uuid4
from scipy.signal import convolve, resample_poly

from fastmss.simulator import ConversationalMeetingSimulator


class TTSMeetingSimulator(ConversationalMeetingSimulator):
    def __init__(self, cfg, output_dir, rirs=None, noise_files=None):
        if not isinstance(cfg, SimpleNamespace):
            from omegaconf import OmegaConf
            cfg = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
        # parent expects spk2cuts; the TTS path supplies its own audio
        super().__init__(cfg, output_dir, spk2cuts={}, rirs=rirs,
                         noise_files=noise_files)

    # ---- level model: per-speaker baseline + slow drift, not per-utterance ----
    def _speaker_levels(self, speakers, rng):
        base = np.random.uniform(*self.cfg.base_gain)
        return base, {s: np.random.uniform(*self.cfg.rel_gain) for s in speakers}

    def _drift(self, t, spk_seed, sigma=1.5, tau=45.0):
        """Smooth mean-reverting walk sampled at utterance onsets (hand-set)."""
        rng = np.random.default_rng(spk_seed)
        phase = rng.uniform(0, 2 * np.pi, 3)
        w = 2 * np.pi / np.array([tau, tau * 2.3, tau * 0.6])
        return float(sigma * np.mean(np.sin(w * t + phase)))

    def gen_meeting(self, records, personas, recording_id=None):
        cfg = self.cfg
        sr = cfg.samplerate
        recording_id = recording_id or str(uuid4())
        speakers = sorted({r["speaker"] for r in records})
        rng = np.random.default_rng(abs(hash(recording_id)) % (2 ** 32))

        base_gain, rel = self._speaker_levels(speakers, rng)
        level_off = {"whisper": -13.5, "soft": -7.0, "normal": 0.0,
                     "raised": 3.5, "loud": 6.5}

        do_reverb = cfg.reverberate and np.random.random() < getattr(cfg, "reverb_prob", 1.0)
        spk_rir = {}
        if do_reverb and self.rirs:
            room = self.rirs[np.random.randint(0, len(self.rirs))]
            for i, s in enumerate(speakers):          # ONE position per speaker
                spk_rir[s] = str(room[i % len(room)])

        total = max(r["onset"] + r["dur"] for r in records)
        out = np.zeros((1, int(total * sr) + sr))
        stems = {s: np.zeros_like(out) for s in speakers} if cfg.save_spk else {}
        speech_lvls = []

        for r in records:
            wav, sr0 = sf.read(r["path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr0 != sr:
                wav = resample_poly(wav, sr, sr0).astype("float32")
            wav = wav[None, :]
            wav = wav - np.mean(wav, -1, keepdims=True)
            if cfg.use_fir:
                wav = convolve(wav, self.create_fir_highpass(40, 63)[None, :], mode="full")

            lvl = (base_gain + rel[r["speaker"]]
                   + self._drift(r["onset"], hash((recording_id, r["speaker"])) % 2**31)
                   + level_off.get(r["level"], 0.0)
                   + r.get("lombard_db", 0.0))
            lvl = float(np.clip(lvl, base_gain + rel[r["speaker"]] - 8,
                                base_gain + rel[r["speaker"]] + 8))   # clamp the stack
            speech_lvls.append(lvl)
            wav, _ = self.normalize_to(wav, lvl)

            if do_reverb and spk_rir:
                rir, fs = sf.read(spk_rir[r["speaker"]])
                assert fs == sr, f"RIR sr {fs} != {sr}"
                rir = rir[np.argmax(np.abs(rir)):]
                if rir.ndim == 1:
                    rir = rir[:, None]
                wav = convolve(wav, rir.T, mode="full")

            off = int(r["onset"] * sr)
            if off + wav.shape[-1] > out.shape[-1]:
                pad = off + wav.shape[-1] - out.shape[-1]
                out = np.pad(out, ((0, 0), (0, pad)))
                for k in stems:
                    stems[k] = np.pad(stems[k], ((0, 0), (0, pad)))
            out[:, off:off + wav.shape[-1]] += wav
            if cfg.save_spk:
                stems[r["speaker"]][:, off:off + wav.shape[-1]] += wav[0][None, :]

        noise_p = getattr(cfg, "noise_probability_global", 1.0)
        if cfg.add_noise and np.random.random() < noise_p:
            mn = min(speech_lvls)
            out = (self.add_gaussian_noise(out, mn, (-30, 3)) if self.noise_files is None
                   else self.add_real_noise(out, mn, cfg.noise_rel_gain))

        peak = np.abs(out).max()
        if peak > 0.99:                                # ONE headroom decision
            g = 0.99 / peak
            out *= g
            for k in stems:
                stems[k] *= g

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        audio_path = os.path.join(self.output_dir, f"{recording_id}.wav")
        sf.write(audio_path, out.T, sr)
        for k, v in stems.items():
            sf.write(os.path.join(self.output_dir, f"{recording_id}-spk-{k}.wav"), v.T, sr)

        recording = Recording(
            id=recording_id,
            sources=[AudioSource(type="file", channels=[0], source=audio_path)],
            sampling_rate=sr, num_samples=out.shape[-1],
            duration=out.shape[-1] / sr)

        sups = []
        for i, (r, lvl) in enumerate(zip(records, speech_lvls)):
            sups.append(SupervisionSegment(
                id=f"{recording_id}_{i:04d}", recording_id=recording_id,
                start=round(r["onset"], 4), duration=round(r["dur"], 4),
                channel=0, speaker=r["speaker"], text=r["text"], language="English",
                custom={"transition_type": r["event"].upper(),
                        "speech_level_db": round(lvl, 2),
                        "instruct": r["instruct"], "level": r["level"],
                        "lombard_db": round(r.get("lombard_db", 0.0), 2),
                        "checks": r["checks"]}))
        return recording, sups
