"""Stage 4a -- placement. Turns a filled skeleton into a timed plan.

Everything here runs on MEASURED clip durations, never planned ones: every HMM
offset is relative (gap after previous end, ratio of host duration), so TTS
length drift shifts the next onset but never accumulates.

Acoustics (RIR, noise, gain, mixing, manifests) are NOT done here -- that is
FastMSS's job, in simulator.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

BC_HEADROOM_DB = 4.0      # a backchannel must sit this far under its host
INT_FLOOR_DB = 1.0        # an interrupter should not be quieter than the turn it cuts
LEVEL_INERTIA = 0.5       # per-speaker Lombard carry-over across turns
FADE_S = 0.04


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if x.size else -120.0


def _snap_to_dip(wav, sr, cut_s, search_s=0.15):
    """Word-boundary proxy: nearest low-energy frame within +/- search_s."""
    hop, win = int(0.01 * sr), int(0.02 * sr)
    lo = max(0, int((cut_s - search_s) * sr))
    hi = min(len(wav) - win, int((cut_s + search_s) * sr))
    if hi <= lo:
        return min(len(wav), max(0, int(cut_s * sr)))
    best, best_e = lo, np.inf
    for i in range(lo, hi, hop):
        e = float(np.mean(wav[i:i + win] ** 2))
        if e < best_e:
            best, best_e = i, e
    return best


def _fade_out(wav, sr):
    n = min(len(wav), int(FADE_S * sr))
    if n > 0:
        wav = wav.copy()
        wav[-n:] *= np.linspace(1.0, 0.0, n)
    return wav


def lombard_boost_db(own_db, competing_db):
    """Classic ~0.4 dB per dB slope, floored at 0 and capped at 6."""
    return float(np.clip(3.0 + 0.4 * (competing_db - own_db), 0.0, 6.0))


def build_plan(skeleton, dialogue, backend, personas, out_dir="clips_final",
               verbose=True):
    """-> list of placed utterance records, ordered by uid."""
    sr = backend.sr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    placed: dict[int, dict] = {}
    records: list[dict] = []
    t_prev_end = 0.0
    prev_main_uid = None
    level_state: dict[str, float] = {}

    for slot in skeleton["slots"]:
        uid = slot["uid"]
        e = dialogue[uid]
        wav, src = backend.synth(e["text"], e["style"], e["level"],
                                 slot["speaker"], personas.get(slot["speaker"], {}))
        dur = len(wav) / sr
        checks: list[str] = []

        if slot["event"] == "backchannel":
            host = placed.get(slot["host_uid"])
            if host is None:
                continue
            onset = host["onset"] + slot["pos_frac"] * host["dur"]
            onset = min(onset, host["onset"] + max(0.0, host["dur"] - dur))
            if onset < host["onset"] - 1e-6:
                checks.append("backchannel clamped to host onset")
            seg = host["wav"][int((onset - host["onset"]) * sr):
                              int((onset - host["onset"] + dur) * sr)]
            delta = _rms_db(wav) - _rms_db(seg)
            if seg.size and delta > -BC_HEADROOM_DB:
                sc = 10 ** ((-BC_HEADROOM_DB - delta) / 20.0)
                wav = (wav * sc).astype("float32")
                checks.append(f"backchannel rescaled {20*np.log10(sc):+.1f} dB")
            rec = dict(uid=uid, speaker=slot["speaker"], event="backchannel",
                       onset=onset, dur=dur, wav=wav, style=e["style"],
                       level=e["level"], text=e["text"], src=src, checks=checks)
            placed[uid] = rec
            records.append(rec)
            continue

        if slot["event"] == "interrupt" and prev_main_uid is not None:
            host = placed[prev_main_uid]
            onset = max(0.0, t_prev_end - slot["overlap_ratio"] * host["dur"])
        else:
            onset = max(0.0, t_prev_end + slot["offset"])

        if slot["event"] == "interrupt" and prev_main_uid is not None:
            host = placed[prev_main_uid]
            ov_start = max(0.0, onset - host["onset"])
            if ov_start < host["dur"]:
                b = lombard_boost_db(_rms_db(host["wav"]), _rms_db(wav))
                if b > 0.05:
                    host["lombard_db"] = host.get("lombard_db", 0.0) + b
                    host["checks"].append(f"lombard +{b:.1f} dB from uid{uid}")
            # post_overlap is a FRACTION of measured host duration -- absolute
            # seconds silently no-op when the host renders shorter than planned.
            cut_rel = ov_start + slot["post_overlap_ratio"] * host["dur"]
            if 0 < cut_rel < host["dur"]:
                idx = _snap_to_dip(host["wav"], sr, cut_rel)
                host["wav"] = _fade_out(host["wav"][:idx], sr)
                host["dur"] = len(host["wav"]) / sr
                host["checks"].append(
                    f"truncated by interrupt at {cut_rel:.2f}s -> {host['dur']:.2f}s")
            else:
                host["checks"].append(
                    f"NO-OP truncation (cut {cut_rel:.2f}s vs host {host['dur']:.2f}s)")
            ov = host["wav"][int((onset - host["onset"]) * sr):]
            delta = _rms_db(wav) - _rms_db(ov)
            if ov.size and delta < -INT_FLOOR_DB:
                sc = 10 ** ((-INT_FLOOR_DB - delta) / 20.0)
                wav = (wav * sc).astype("float32")
                checks.append(f"interrupter boosted {20*np.log10(sc):+.1f} dB")
            contested = max(0.0, (host["onset"] + host["dur"]) - onset)
            if contested > 0.05:
                b = lombard_boost_db(_rms_db(wav), _rms_db(host["wav"]))
                tgt = level_state.get(slot["speaker"], 0.0) * (1 - LEVEL_INERTIA) + b
                if tgt > 0.05:
                    checks.append(f"lombard +{tgt:.1f} dB into uid{prev_main_uid}")
                    level_state[slot["speaker"]] = tgt
                    rec_lomb = tgt
                else:
                    rec_lomb = 0.0
            else:
                rec_lomb = 0.0
        else:
            level_state[slot["speaker"]] = level_state.get(slot["speaker"], 0.0) * (1 - LEVEL_INERTIA)
            rec_lomb = 0.0

        if not (0.4 <= dur / slot["planned_dur"] <= 3.0):
            checks.append(f"duration {dur:.2f}s vs planned {slot['planned_dur']:.2f}s")

        rec = dict(uid=uid, speaker=slot["speaker"], event=slot["event"],
                   onset=onset, dur=dur, wav=wav, style=e["style"],
                   level=e["level"], text=e["text"], src=src, checks=checks,
                   lombard_db=rec_lomb)
        placed[uid] = rec
        records.append(rec)
        prev_main_uid = uid
        t_prev_end = onset + dur

    # write the final (possibly truncated) clip for each utterance
    for r in records:
        p = out_dir / f"utt_{r['uid']:04d}.wav"
        sf.write(p, r["wav"].astype("float32"), sr, subtype="FLOAT")
        r["path"] = str(p)
        r["dur"] = len(r["wav"]) / sr
        r.pop("wav")

    records.sort(key=lambda r: r["uid"])
    if verbose:
        end = max(r["onset"] + r["dur"] for r in records)
        print(f"  plan: {len(records)} utts, {end:.1f}s timeline")
    return records
