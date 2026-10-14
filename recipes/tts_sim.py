"""End-to-end TTS meeting simulation: HMM skeleton -> LLM fill -> TTS -> FastMSS.

Single-pass version: the LLM and CosyVoice run interleaved, meeting by meeting,
so both must fit in memory at once. For large corpora, one shared GPU, resume
and NOTSOFAR-like specs, use recipes/tts_staged.py instead.

  python recipes/tts_sim.py --n 3 --duration 30 --reverberate
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastmss.hmm_turn_taking import TransitionParams
from fastmss.tts.backends import get_backend
from fastmss.tts.corpus import (default_cfg, write_combined_manifests,
                                write_meeting_manifests)
from fastmss.tts.fill import cast, fill
from fastmss.tts.plan import build_plan
from fastmss.tts.simulator import TTSMeetingSimulator
from fastmss.tts.refbank import build_bank, cast_meeting
from fastmss.tts.scenarios import pick_type, sample_scenario
from fastmss.tts.skeleton import (describe_timing, sample_skeleton,
                                  timing_profile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="meetings to generate")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--speakers", type=int, default=3)
    ap.add_argument("--backend", default="cosyvoice")
    ap.add_argument("--librispeech", default=os.environ.get("LIBRISPEECH_ROOT"),
                    help="LibriSpeech root (or set $LIBRISPEECH_ROOT); the voice "
                         "bank is cloned from its reference clips")
    ap.add_argument("--bank-limit", type=int, default=60)
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--no-llm", action="store_true", help="stub fill, no LLM")
    ap.add_argument("--convo-type", default=None,
                    help="force a ConvoType (default: weighted draw)")
    ap.add_argument("--reverberate", action="store_true")
    ap.add_argument("--add-noise", action="store_true")
    ap.add_argument("--noise-folder", default=None)
    ap.add_argument("--out", default="exp/tts_sim")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not a.librispeech:
        ap.error("--librispeech (or $LIBRISPEECH_ROOT) is required: the voice "
                 "bank clones from LibriSpeech reference clips")

    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    noise_files = None
    if a.add_noise and a.noise_folder:
        noise_files = [str(p) for ext in ("*.wav", "*.flac")
                       for p in Path(a.noise_folder).rglob(ext)]
        print(f"noise: {len(noise_files)} files")

    cfg = default_cfg(reverberate=a.reverberate, add_noise=a.add_noise,
                      duration=a.duration)

    rirs = None
    if a.reverberate:
        from fastmss.rirsimulator import RIRSimulator
        rcfg = default_cfg(reverberate=True)
        rcfg.output_dir = str(out / "rirs")
        Path(rcfg.output_dir).mkdir(parents=True, exist_ok=True)
        rcfg.samplerate = cfg.samplerate
        print("simulating RIRs (pyroomacoustics) ...")
        sim = RIRSimulator(rcfg)
        rirs = [sim.gen_rirs(f"room_{i}") for i in range(2)]
        print(f"  {len(rirs)} rooms, {len(rirs[0])} positions each")

    bank = build_bank(a.librispeech, limit=a.bank_limit,
                      cache=str(out / "bank.json"))
    print(f"reference bank: {len(bank)} LibriSpeech speakers "
          f"(M={sum(b['sex']=='M' for b in bank)} F={sum(b['sex']=='F' for b in bank)})")
    backend = get_backend(a.backend, cache_dir=str(out / "clips_native"))
    tts_sim = TTSMeetingSimulator(cfg, str(out / "audio"), rirs=rirs,
                                  noise_files=noise_files)

    recs, all_sups = [], []
    for i in range(a.n):
        mid = f"ttsmeeting_{i:03d}"
        print(f"\n[{mid}]")
        skel = sample_skeleton(a.speakers, a.duration, seed=a.seed + i,
                               hmm=TransitionParams(), use_markov=cfg.use_markov)
        ev = {}
        for s in skel["slots"]:
            ev[s["event"]] = ev.get(s["event"], 0) + 1
        print(f"  skeleton: {len(skel['slots'])} slots, {ev}")

        # voice-first: voices and sexes are drawn before anyone writes a persona
        voices = cast_meeting(bank, a.speakers, seed=a.seed + i)
        print("  voices: " + ", ".join(
            f"{k}=LS{v['ls_id']}({v['sex']})" for k, v in voices.items()))
        ctype = a.convo_type or pick_type(a.seed + i)
        scen = sample_scenario(ctype, a.seed + i)
        print(f"  scenario [{ctype}]: " + " | ".join(scen.values()))
        if a.no_llm:
            premise = f"a {ctype.replace('_', ' ')}"
            people = {k: {"name": k, "manner": "flat", "about": "test"}
                      for k in voices}
            dialogue = {s["uid"]: {"text": "mm-hm" if s["event"] == "backchannel"
                                  else f"this is slot {s['uid']}",
                                  "instruct": "Speak naturally.",
                                  "level": "soft" if s["event"] == "backchannel" else "normal"}
                        for s in skel["slots"]}
        else:
            sexes = {k: v["sex"] for k, v in voices.items()}
            timing = describe_timing(timing_profile(skel))
            print("  timing:"); print(timing)
            c = cast(ctype, scen, sexes, timing=timing,
                     base_url=a.base_url, model=a.model)
            premise, people = c["premise"], c["personas"]
            print(f"  premise: {premise}")
            for k, v in people.items():
                print(f"    {k} {v['name']}: {v['about']} [{v['manner']}]")
            dialogue = fill(skel, people, premise=premise,
                            base_url=a.base_url, model=a.model)

        personas = {k: {**voices[k], **people.get(k, {})} for k in voices}

        records = build_plan(skel, dialogue, backend, personas,
                             out_dir=str(out / "clips_final" / mid))
        rec, sups = tts_sim.gen_meeting(records, personas, recording_id=mid)
        recs.append(rec)
        all_sups.extend(sups)

        json.dump({"convo_type": ctype, "scenario": scen,
                   "premise": premise, "personas": personas,
                   "skeleton": skel, "dialogue": dialogue,
                   "records": [{k: v for k, v in r.items()} for r in records]},
                  open(out / "manifests" / f"{mid}_realized.json", "w"), indent=1)
        # per-meeting manifests, written now rather than after the whole loop:
        # a run killed part-way otherwise leaves finished audio with no cuts,
        # no supervisions and no RTTM.
        write_meeting_manifests(out / "manifests", rec, sups)
        print(f"  audio: {rec.duration:.1f}s -> {out/'audio'/(mid+'.wav')}")

    write_combined_manifests(out / "manifests", recs, all_sups)
    if hasattr(backend, "close"):
        backend.close()
    print(f"\nOK: {len(recs)} meetings, {len(all_sups)} supervisions -> {out}/manifests")


if __name__ == "__main__":
    main()
