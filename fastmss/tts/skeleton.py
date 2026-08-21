"""Stage 1 -- turn-taking skeleton, sampled with FastMSS's own HMM parameters.

Reuses fastmss.hmm_turn_taking.TransitionParams verbatim (including .fit() on a
real SupervisionSet), so the event rates, pauses and overlap ratios are the ones
FastMSS already ships. This module only decides WHEN people speak, never what.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from fastmss.hmm_turn_taking import TransitionParams, TransitionType

MIN_TURN_S = 1.2          # below this a floor-taking event is a fragment
EPS = 0.03                # truncated-exponential clip, as in ConversationalMeetingSimulator


@dataclass
class Slot:
    uid: int
    speaker: str
    event: str                       # hold | switch | interrupt | backchannel
    planned_dur: float
    target_words: int
    offset: float                    # relative to previous main turn's end (<0 = overlap)
    gap_before: Optional[float] = None
    overlap_ratio: Optional[float] = None
    post_overlap_ratio: Optional[float] = None   # fraction of host dur, NOT seconds
    host_uid: Optional[int] = None
    pos_frac: Optional[float] = None
    interrupted_speaker: Optional[str] = None


_EVENT = {
    TransitionType.TURN_HOLD: "hold",
    TransitionType.TURN_SWITCH: "switch",
    TransitionType.INTERRUPTION: "interrupt",
    TransitionType.BACKCHANNEL: "backchannel",
}


def sample_skeleton(
    n_speakers: int,
    duration: float,
    words_per_sec: float = 2.8,
    seed: int = 0,
    hmm: TransitionParams | None = None,
    use_markov: bool = False,
    turn_dur_range: tuple[float, float] = (MIN_TURN_S, 12.0),
    bc_dur_range: tuple[float, float] = (0.25, 0.9),
) -> dict:
    """Sample slots until planned cumulative time reaches `duration`."""
    rng = np.random.default_rng(seed)
    hmm = hmm or TransitionParams()
    speakers = [f"spk_{chr(65 + i)}" for i in range(n_speakers)]

    slots: list[Slot] = []
    t = 0.0
    uid = 0
    cur = speakers[0]
    prev_tt: TransitionType | None = None
    last_main: Slot | None = None

    while t < duration:
        if use_markov and prev_tt is not None:
            probs = hmm.p_markov[list(TransitionType).index(prev_tt)]
        else:
            probs = hmm.p_ind
        tt = list(TransitionType)[int(rng.choice(len(TransitionType), p=probs))]
        event = _EVENT[tt]

        if event == "backchannel":
            if last_main is None:
                prev_tt = tt
                continue
            listener = [s for s in speakers if s != last_main.speaker]
            spk = str(rng.choice(listener))
            dur = float(np.clip(rng.lognormal(-0.85, 0.35), *bc_dur_range))
            ratio = float(np.clip(rng.exponential(hmm.beta_bc), EPS, 1 - EPS))
            slots.append(Slot(uid=uid, speaker=spk, event="backchannel",
                              planned_dur=round(dur, 3),
                              target_words=max(1, round(dur * words_per_sec * 0.57)),
                              offset=0.0, host_uid=last_main.uid,
                              pos_frac=round(1.0 - ratio, 3)))
            uid += 1
            prev_tt = tt
            continue                                  # backchannel does not advance the floor

        dur = float(np.clip(rng.lognormal(0.55, 0.75), *turn_dur_range))
        if event == "hold":
            spk = cur
            gap = float(rng.exponential(hmm.beta_th))
            s = Slot(uid=uid, speaker=spk, event="hold", planned_dur=round(dur, 3),
                     target_words=max(3, round(dur * words_per_sec)),
                     offset=round(gap, 3), gap_before=round(gap, 3))
        elif event == "switch":
            spk = str(rng.choice([s for s in speakers if s != cur]))
            gap = float(rng.exponential(hmm.beta_ts))
            s = Slot(uid=uid, speaker=spk, event="switch", planned_dur=round(dur, 3),
                     target_words=max(3, round(dur * words_per_sec)),
                     offset=round(gap, 3), gap_before=round(gap, 3))
        else:                                          # interrupt
            spk = str(rng.choice([s for s in speakers if s != cur]))
            ratio = float(np.clip(rng.exponential(hmm.beta_ir), EPS, 1 - EPS))
            host_dur = last_main.planned_dur if last_main else dur
            # post_overlap as a FRACTION of host duration (see INTEGRATION.md 9):
            # absolute seconds silently no-op the truncation when the host renders short.
            s = Slot(uid=uid, speaker=spk, event="interrupt", planned_dur=round(dur, 3),
                     target_words=max(3, round(dur * words_per_sec)),
                     offset=round(-ratio * host_dur, 3),
                     overlap_ratio=round(ratio, 3),
                     post_overlap_ratio=round(float(np.clip(rng.exponential(0.15),
                                                            0.02, 0.5)), 3),
                     interrupted_speaker=cur)

        slots.append(s)
        cur = spk
        last_main = s
        prev_tt = tt
        uid += 1
        t += max(0.0, s.offset) + dur

    # every sampled speaker must actually appear
    seen = {s.speaker for s in slots}
    for miss in [s for s in speakers if s not in seen]:
        slots[-1].speaker = miss

    return {"words_per_sec": words_per_sec, "speakers": speakers,
            "planned_duration": round(t, 3),
            "slots": [asdict(s) for s in slots]}


if __name__ == "__main__":
    import sys
    sk = sample_skeleton(4, 30.0, seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(json.dumps(sk, indent=1)[:800])
    ev = {}
    for s in sk["slots"]:
        ev[s["event"]] = ev.get(s["event"], 0) + 1
    print(f"\n{len(sk['slots'])} slots, {sk['planned_duration']}s planned, {ev}")


def timing_profile(skeleton: dict) -> dict[str, dict]:
    """Per-speaker turn-taking profile, derived from the sampled skeleton.

    The HMM fixes who speaks, for how long, and how often BEFORE anyone has a
    personality. Handing this to the cast step lets the personas be written to
    fit the timing, instead of contradicting it -- otherwise you get a persona
    who "over-explains" holding the shortest turns in the meeting.
    """
    prof: dict[str, dict] = {}
    for s in skeleton["slots"]:
        p = prof.setdefault(s["speaker"], {"turns": 0, "backchannels": 0,
                                           "interrupts": 0, "holds": 0,
                                           "words": [], "talk_s": 0.0})
        if s["event"] == "backchannel":
            p["backchannels"] += 1
            continue
        p["turns"] += 1
        p["words"].append(s["target_words"])
        p["talk_s"] += s["planned_dur"]
        if s["event"] == "interrupt":
            p["interrupts"] += 1
        elif s["event"] == "hold":
            p["holds"] += 1
    total = sum(p["talk_s"] for p in prof.values()) or 1.0
    for p in prof.values():
        w = sorted(p["words"]) or [0]
        p["median_words"] = w[len(w) // 2]
        p["longest_words"] = w[-1]
        p["share"] = round(p["talk_s"] / total, 2)
        del p["words"], p["talk_s"]
    return prof


def describe_timing(prof: dict[str, dict]) -> str:
    """Render the profile as plain English for the cast prompt."""
    longest = max((p["longest_words"] for p in prof.values()), default=0)
    lines = []
    for spk, p in sorted(prof.items()):
        bits = [f"{p['turns']} turns", f"typically {p['median_words']} words"]
        if p["longest_words"] == longest and longest:
            bits.append(f"has the longest turn in the conversation ({longest} words)")
        if p["interrupts"] >= max(2, p["turns"] // 2):
            bits.append(f"cuts in a lot ({p['interrupts']} interruptions)")
        elif p["interrupts"] == 0:
            bits.append("never cuts anyone off")
        if p["holds"] >= 2:
            bits.append("often carries on after their own pause")
        if p["backchannels"] >= 2:
            bits.append(f"mostly listens, {p['backchannels']} short reactions")
        bits.append(f"{int(p['share'] * 100)}% of the talking")
        lines.append(f"  {spk}: " + ", ".join(bits))
    return "\n".join(lines)
