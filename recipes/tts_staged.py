"""Two-phase (staged) TTS meeting simulation.

    --phase text   HMM skeleton -> scenario -> LLM cast + dialogue fill, for every
                   meeting; plans go to manifests/tts-text-plans.jsonl and
                   manifests/<id>_text.json.
    --phase tts    unload the Ollama model, start CosyVoice worker(s), synthesise
                   every saved plan, mix, and write Lhotse/RTTM manifests.
    --phase all    both, in that order (default).

Splitting the phases lets the LLM and CosyVoice take turns on one GPU, and makes
every stage resumable: existing text plans are skipped, --resume-tts skips
meetings whose audio and manifests are complete, and the CosyVoice clip cache
(clips_native/) means a re-run never re-synthesises a line.

Meetings are indexed globally (ttsmeeting_<index>) and every per-meeting seed is
derived from --seed + index, so a corpus can be extended with --append without
changing or duplicating what already exists. --notsofar-like samples per-meeting
durations (optionally normalised to --total-duration) and 3-8 speaker counts.

See README.md for worked examples and docs/tts.md for the external setup.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import lhotse
from lhotse import RecordingSet, SupervisionSet

from fastmss.hmm_turn_taking import TransitionParams
from fastmss.rirsimulator import RIRSimulator
from fastmss.tts.backends import get_backend, tts_safe_instruct, tts_safe_text
from fastmss.tts.corpus import (default_cfg, fmt_seconds, write_combined_manifests,
                                write_json_atomic, write_meeting_manifests)
from fastmss.tts.fill import cast, fill
from fastmss.tts.plan import build_plan
from fastmss.tts.refbank import build_bank, cast_meeting
from fastmss.tts.scenarios import (pick_type, sample_scenario, sample_scenario_permuted,
                                   scenario_key, space_size)
from fastmss.tts.simulator import TTSMeetingSimulator
from fastmss.tts.skeleton import describe_timing, sample_skeleton, timing_profile


# Each TTS thread lazily owns one CosyVoice worker + simulator (thread-local);
# _TTS_BACKENDS remembers them all so they can be shut down at the end.
_TTS_LOCAL = threading.local()
_TTS_BACKENDS = []
_TTS_BACKENDS_LOCK = threading.Lock()
# Per-meeting and wall-clock timings for the final summary.
_TEXT_TIMES: list[float] = []
_TEXT_WALL: float | None = None
_TTS_TIMES: list[float] = []
_TTS_WALL: float | None = None


# --------------------------------------------------------------------------
# Text plans and generation state
# --------------------------------------------------------------------------

def manifest_path(out: Path) -> Path:
    return out / "manifests" / "tts-text-plans.jsonl"


def write_plan(out: Path, plan: dict) -> None:
    """Save one meeting's plan as <id>_text.json and append it to the JSONL index."""
    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    with open(mdir / f"{plan['id']}_text.json", "w") as f:
        json.dump(plan, f, indent=1)
    with open(manifest_path(out), "a") as f:
        f.write(json.dumps(plan) + "\n")


_MEETING_ID_RE = re.compile(r"^ttsmeeting_(\d+)$")


def _plan_index(plan: dict) -> int:
    match = _MEETING_ID_RE.fullmatch(str(plan.get("id", "")))
    if not match:
        raise ValueError(f"invalid FastMSS meeting id in text plan: {plan.get('id')!r}")
    return int(match.group(1))


def load_plans(out: Path) -> list[dict]:
    """All text plans in the corpus, sorted by global index; duplicate ids are an error."""
    path = manifest_path(out)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run --phase text first")
    plans_by_id: dict[str, dict] = {}
    for line_no, line in enumerate(open(path), start=1):
        if not line.strip():
            continue
        plan = json.loads(line)
        mid = plan.get("id")
        if mid in plans_by_id:
            raise RuntimeError(f"duplicate text plan id {mid!r} in {path} (line {line_no})")
        _plan_index(plan)
        plans_by_id[mid] = plan
    return sorted(plans_by_id.values(), key=_plan_index)


def generation_state_path(out: Path) -> Path:
    return out / "manifests" / "generation-state.json"


def _load_generation_state(out: Path) -> dict | None:
    path = generation_state_path(out)
    return json.load(open(path)) if path.exists() else None


def _existing_scenario_keys(out: Path) -> dict[str, set[str]]:
    """Scenario tuples already used in the corpus, per conversation type."""
    used: dict[str, set[str]] = {}
    for plan in load_plans(out):
        ctype, scenario = plan.get("convo_type"), plan.get("scenario")
        if isinstance(ctype, str) and isinstance(scenario, dict):
            used.setdefault(ctype, set()).add(scenario_key(ctype, scenario))
    return used


class ScenarioAllocator:
    """Allocate complete scenario tuples without reuse, including legacy plans.

    Walks the seeded permutation from sample_scenario_permuted() and skips any
    tuple already in `used`, so a meeting never repeats another's situation.
    """

    def __init__(self, seed: int, start_index: int, used: dict[str, set[str]]):
        self.seed = seed
        self.start_index = start_index
        self.used = used
        self.cursors: dict[str, int] = {}

    def next(self, convo_type: str) -> tuple[dict[str, str], int]:
        used = self.used.setdefault(convo_type, set())
        size = space_size(convo_type)
        if len(used) >= size:
            raise RuntimeError(
                f"scenario space exhausted for {convo_type}: {len(used)} of {size} used")
        cursor = self.cursors.get(convo_type, self.start_index)
        for _ in range(size):
            scenario = sample_scenario_permuted(convo_type, cursor, self.seed)
            ordinal = cursor
            cursor += 1
            key = scenario_key(convo_type, scenario)
            if key not in used:
                used.add(key)
                self.cursors[convo_type] = cursor
                return scenario, ordinal
        raise RuntimeError(f"could not allocate a fresh scenario for {convo_type}")


def _append_start_index(out: Path) -> int:
    plans = load_plans(out)
    if not plans:
        raise RuntimeError(f"--append requires at least one plan in {manifest_path(out)}")
    return max(_plan_index(plan) for plan in plans) + 1


def _save_generation_state(out: Path, state: dict) -> None:
    write_json_atomic(generation_state_path(out), state)


def _batch_metadata(args) -> dict:
    return {
        "start_index": args.start_index,
        "n": args.n,
        "total_duration": args.total_duration,
        "notsofar_like": args.notsofar_like,
        "speakers_min": args.speakers_min,
        "speakers_max": args.speakers_max,
        "speaker_lambda": args.speaker_lambda,
    }


def _configure_generation_range(args, out: Path, ap: argparse.ArgumentParser) -> None:
    """Resolve args.start_index (and --append bookkeeping) before anything runs.

    --append picks the next free global index, records the batch as
    `active_batch` in generation-state.json *before* generating, and turns on
    unique scenarios. Re-running the identical command while that batch is
    active resumes it (and enables --resume-tts) instead of starting another.
    """
    if args.append and args.start_index is not None:
        ap.error("--append and --start-index cannot be used together")
    if args.append and args.phase == "tts":
        ap.error("--append is for --phase text or all; use --start-index with --phase tts")
    if args.append and args.overwrite_text:
        ap.error("--append cannot be combined with --overwrite-text")
    if args.start_index is not None and args.start_index < 0:
        ap.error("--start-index must be non-negative")

    if args.append:
        state = _load_generation_state(out)
        if state is not None and state.get("base_seed") != args.seed:
            ap.error(
                f"--seed={args.seed} conflicts with existing generation state "
                f"(base seed {state.get('base_seed')})")
        if state is None:
            print("append: no generation state found; treating this as a legacy corpus. "
                  "Use the original base --seed (0 for the documented first batch).")
            state = {"schema_version": 1, "base_seed": args.seed}

        active = state.get("active_batch")
        if active is not None:
            if active.get("n") != args.n or active.get("total_duration") != args.total_duration:
                ap.error(
                    "an incomplete append batch exists; rerun it with the same --n and "
                    "--total-duration, or finish it with --phase tts --start-index")
            args.start_index = int(active["start_index"])
            args.resume_tts = True
            print(f"append: resuming incomplete batch at global meeting index "
                  f"{args.start_index} ({args.n} meetings); --resume-tts enabled")
        else:
            args.start_index = _append_start_index(out)
            state["active_batch"] = _batch_metadata(args)
            _save_generation_state(out, state)
            print(f"append: starting at global meeting index {args.start_index}")
        args.unique_scenarios = True
        print("append: full scenario tuples will not be reused")
    elif args.phase in {"all", "text"} and args.start_index is None:
        args.start_index = 0


def _write_generation_state(out: Path, args) -> None:
    """Record the corpus seed, next free index and last batch after text generation."""
    plans = load_plans(out)
    next_index = max((_plan_index(plan) for plan in plans), default=-1) + 1
    state = _load_generation_state(out) or {}
    state.update({
        "schema_version": 1,
        "base_seed": args.seed,
        "next_index": next_index,
        "scenario_policy": "unique_full_axis_tuple" if args.unique_scenarios else "seeded_hash",
        "last_batch": _batch_metadata(args),
    })
    _save_generation_state(out, state)


def _complete_active_batch(out: Path) -> None:
    """Clear `active_batch` once every meeting in it has a text plan and finished audio."""
    state = _load_generation_state(out)
    if not state or not state.get("active_batch"):
        return
    active = state["active_batch"]
    start = int(active["start_index"])
    expected = set(range(start, start + int(active["n"])))
    plans = load_plans(out)
    by_index = {_plan_index(plan): plan for plan in plans}
    if not expected.issubset(by_index):
        return
    if not all(_meeting_complete(out, by_index[index]["id"]) for index in expected):
        return
    state["last_completed_append_batch"] = active
    state.pop("active_batch")
    state["next_index"] = max((_plan_index(plan) for plan in plans), default=-1) + 1
    _save_generation_state(out, state)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _timing_summary(label: str, times: list[float], wall_s: float | None = None) -> str:
    if not times:
        return f"{label}: no new items"
    avg = sum(times) / len(times)
    msg = (f"{label}: n={len(times)}, avg={fmt_seconds(avg)}, "
           f"min={fmt_seconds(min(times))}, max={fmt_seconds(max(times))}")
    if wall_s is not None:
        msg += f", wall={fmt_seconds(wall_s)}"
    return msg


def _print_timing_report(include_text: bool, include_tts: bool) -> None:
    if not include_text and not include_tts:
        return
    print("\nTiming summary")
    if include_text:
        print("  " + _timing_summary("Text generation", _TEXT_TIMES, _TEXT_WALL))
    if include_tts:
        print("  " + _timing_summary("TTS + assembly", _TTS_TIMES, _TTS_WALL))


# --------------------------------------------------------------------------
# Meeting specs: per-meeting duration and speaker count
# --------------------------------------------------------------------------

def _bounded_durations(n: int, total: float, seed: int, cv: float,
                       min_s: float | None, max_s: float | None) -> list[float]:
    """n durations that sum to exactly `total`, each within [min_s, max_s].

    Draws log-normal weights (spread set by `cv`), scales them to the total,
    then repeatedly clamps out-of-range values and rescales the rest; any
    remaining rounding error is absorbed by the values closest to the mean.
    Defaults: min = 0.45 x mean, max = 1.8 x mean.
    """
    if n <= 0:
        return []
    avg = total / n
    lo = float(min_s) if min_s is not None else max(1.0, 0.45 * avg)
    hi = float(max_s) if max_s is not None else max(lo, 1.8 * avg)
    if lo * n > total or hi * n < total:
        raise ValueError(
            f"duration bounds infeasible for total={total:.1f}s, n={n}: "
            f"min*n={lo*n:.1f}s, max*n={hi*n:.1f}s")

    rng = random.Random(seed)
    weights = [rng.lognormvariate(0.0, cv) for _ in range(n)]
    vals = [total * w / sum(weights) for w in weights]

    fixed = [False] * n
    vals = [min(hi, max(lo, v)) for v in vals]
    for _ in range(n + 1):
        free = [i for i in range(n) if not fixed[i]]
        if not free:
            break
        remaining = total - sum(vals[i] for i in range(n) if fixed[i])
        scale = remaining / sum(vals[i] for i in free)
        changed = False
        for i in free:
            vals[i] *= scale
            if vals[i] < lo:
                vals[i] = lo
                fixed[i] = True
                changed = True
            elif vals[i] > hi:
                vals[i] = hi
                fixed[i] = True
                changed = True
        if not changed:
            break

    delta = total - sum(vals)
    for i in sorted(range(n), key=lambda j: abs(vals[j] - avg)):
        room = (hi - vals[i]) if delta > 0 else (vals[i] - lo)
        step = max(-room, min(room, delta))
        vals[i] += step
        delta -= step
        if abs(delta) < 1e-9:
            break
    vals[-1] += total - sum(vals)
    rounded = [round(v, 3) for v in vals]
    rounded[-1] = round(rounded[-1] + round(total, 3) - sum(rounded), 3)
    return rounded


def _poisson_weight(k: int, lam: float) -> float:
    """Unnormalised Poisson pmf, computed in log space."""
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _speaker_counts(n: int, seed: int, default: int, randomize: bool,
                    min_n: int, max_n: int, lam: float) -> list[int]:
    """Fixed `default` speakers, or Poisson(lam) truncated to [min_n, max_n]."""
    if not randomize:
        return [default] * n
    rng = random.Random(seed)
    choices = list(range(min_n, max_n + 1))
    if not choices:
        raise ValueError("speaker range is empty")
    weights = [_poisson_weight(k, lam) for k in choices]
    return rng.choices(choices, weights=weights, k=n)


def meeting_specs(args, start_index: int) -> list[dict]:
    """[{duration, speakers}] for meetings start_index .. start_index + n - 1.

    Seeded from --seed + start_index, so a batch's specs are reproducible.
    """
    randomize_dur = args.randomize_duration or args.notsofar_like
    randomize_spk = args.randomize_speakers or args.notsofar_like
    total = float(args.total_duration) if args.total_duration is not None else args.n * args.duration
    durations = (_bounded_durations(args.n, total, args.seed + start_index + 1009,
                                    args.duration_cv, args.duration_min, args.duration_max)
                 if randomize_dur else [float(args.duration)] * args.n)
    speakers = _speaker_counts(args.n, args.seed + start_index + 2003, args.speakers,
                               randomize_spk, args.speakers_min, args.speakers_max,
                               args.speaker_lambda)
    specs = [{"duration": durations[i], "speakers": speakers[i]} for i in range(args.n)]
    if randomize_dur or randomize_spk:
        print("meeting specs: " + ", ".join(
            f"{start_index + i:03d}={specs[i]['duration']:.1f}s/{specs[i]['speakers']}spk"
            for i in range(args.n)))
        print(f"meeting specs total duration: {sum(durations):.1f}s")
    return specs


# --------------------------------------------------------------------------
# Ollama hand-off between the phases
# --------------------------------------------------------------------------

def _ollama_resident_models() -> list[str]:
    """Model names listed by `ollama ps` (empty if ollama is unavailable)."""
    ps = subprocess.run(["ollama", "ps"], text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, check=False)
    if ps.returncode != 0:
        return []
    names = []
    for line in ps.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def unload_ollama_model(model: str, wait_s: float = 20.0, require_empty: bool = True) -> None:
    """Ask Ollama to unload the model and wait until no Ollama model is resident."""
    print(f"unloading Ollama model: {model}")
    stop = subprocess.run(["ollama", "stop", model], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False)
    if stop.returncode != 0 and not _ollama_resident_models():
        print("Ollama is already unavailable; continuing to CosyVoice")
        return
    deadline = time.time() + wait_s
    while time.time() < deadline:
        resident = _ollama_resident_models()
        if model not in resident and (not require_empty or not resident):
            print("Ollama has no resident model using GPU" if not resident
                  else f"Ollama resident models: {resident}")
            return
        time.sleep(1.0)
    resident = _ollama_resident_models()
    raise RuntimeError(
        f"Ollama still has resident model(s) after {wait_s}s: {resident}. "
        f"Stop them manually with `ollama stop <model>` or use --skip-ollama-stop.")


# --------------------------------------------------------------------------
# Phase 1: text
# --------------------------------------------------------------------------

def generate_text_plans(args) -> list[dict]:
    """Write a text plan for every meeting in the selected range; existing plans are kept.

    Per meeting: HMM skeleton -> voices from the LibriSpeech bank -> scenario
    -> LLM premise/personas (shaped by the skeleton's timing) -> dialogue fill.
    """
    global _TEXT_TIMES, _TEXT_WALL
    out = Path(args.out)
    if args.overwrite_text and manifest_path(out).exists():
        manifest_path(out).unlink()

    bank = build_bank(args.librispeech, limit=args.bank_limit,
                      cache=str(out / "bank.json"))
    print(f"reference bank: {len(bank)} LibriSpeech speakers "
          f"(M={sum(b['sex']=='M' for b in bank)} F={sum(b['sex']=='F' for b in bank)})")

    specs = meeting_specs(args, args.start_index)
    max_speakers = max(spec["speakers"] for spec in specs) if specs else 0
    if len(bank) < max_speakers:
        raise RuntimeError(
            f"voice bank has {len(bank)} speakers but requested up to {max_speakers}")

    used_scenarios = (_existing_scenario_keys(out)
                      if args.unique_scenarios and manifest_path(out).exists() else {})
    scenario_allocator = (ScenarioAllocator(args.seed, args.start_index, used_scenarios)
                          if args.unique_scenarios else None)
    plans = []
    text_times = []
    text_wall_start = time.perf_counter()
    for i in range(args.n):
        spec_i = specs[i]
        n_speakers = int(spec_i["speakers"])
        duration = float(spec_i["duration"])
        global_index = args.start_index + i
        meeting_seed = args.seed + global_index
        mid = f"ttsmeeting_{global_index:03d}"
        existing = out / "manifests" / f"{mid}_text.json"
        if existing.exists() and not args.overwrite_text:
            plan = json.load(open(existing))
            plans.append(plan)
            print(f"[{mid}] text exists, skipping")
            continue

        print(f"\n[{mid}] text")
        text_start = time.perf_counter()
        print(f"  target: {duration:.1f}s, {n_speakers} speaker(s); seed={meeting_seed}")
        skel = sample_skeleton(n_speakers, duration, seed=meeting_seed,
                               hmm=TransitionParams())
        voices = cast_meeting(bank, n_speakers, seed=meeting_seed)
        ctype = args.convo_type or pick_type(meeting_seed)
        if scenario_allocator is None:
            scen = sample_scenario(ctype, meeting_seed)
            scenario_ordinal = None
        else:
            scen, scenario_ordinal = scenario_allocator.next(ctype)
        print(f"  scenario [{ctype}]: " + " | ".join(scen.values()))

        if args.no_llm:                              # stub text, same timing
            premise = f"a {ctype.replace('_', ' ')}"
            people = {k: {"name": k, "manner": "flat", "about": "test"}
                      for k in voices}
            dialogue = {s["uid"]: {"text": "mm" if s["event"] == "backchannel"
                                   else f"this is slot {s['uid']}",
                                   "instruct": "Speak naturally.",
                                   "level": "soft" if s["event"] == "backchannel"
                                   else "normal"}
                        for s in skel["slots"]}
        else:
            sexes = {k: v["sex"] for k, v in voices.items()}
            timing = describe_timing(timing_profile(skel))
            print("  timing:"); print(timing)
            c = cast(ctype, scen, sexes, timing=timing,
                     base_url=args.base_url, model=args.model)
            premise, people = c["premise"], c["personas"]
            print(f"  premise: {premise}")
            dialogue = fill(skel, people, premise=premise,
                            base_url=args.base_url, model=args.model,
                            chunk=max(1, int(args.fill_chunk)))

        plan = {"id": mid, "global_index": global_index, "meeting_seed": meeting_seed,
                "convo_type": ctype, "scenario": scen,
                "target_duration": duration, "n_speakers": n_speakers,
                "premise": premise, "personas": {k: {**voices[k], **people.get(k, {})}
                                                 for k in voices},
                "skeleton": skel, "dialogue": dialogue}
        if scenario_ordinal is not None:
            plan["scenario_ordinal"] = scenario_ordinal
        write_plan(out, plan)
        plans.append(plan)
        elapsed = time.perf_counter() - text_start
        text_times.append(elapsed)
        print(f"  text time: {fmt_seconds(elapsed)}")
    _TEXT_TIMES = text_times
    _TEXT_WALL = time.perf_counter() - text_wall_start
    _write_generation_state(out, args)
    return plans


# --------------------------------------------------------------------------
# Phase 2: TTS + mixing
# --------------------------------------------------------------------------

def _get_tts_state(args, out: Path, cfg, rirs, noise_files):
    """This thread's (CosyVoice backend, mixer), created on first use."""
    state = getattr(_TTS_LOCAL, "state", None)
    if state is not None:
        return state

    backend = get_backend(args.backend, cache_dir=str(out / "clips_native"),
                          retries=args.cosyvoice_retries)
    tts_sim = TTSMeetingSimulator(cfg, str(out / "audio"), rirs=rirs,
                                  noise_files=noise_files)
    state = (backend, tts_sim)
    _TTS_LOCAL.state = state
    with _TTS_BACKENDS_LOCK:
        _TTS_BACKENDS.append(backend)
    return state


def _synthesize_one(plan: dict, args, out: Path, cfg, rirs, noise_files):
    """Render, place and mix one meeting; write its audio and per-meeting manifests."""
    mid = plan["id"]
    tts_start = time.perf_counter()
    print(f"\n[{mid}] TTS + mix")
    backend, tts_sim = _get_tts_state(args, out, cfg, rirs, noise_files)
    dialogue = {}
    repaired = []
    for key, entry in plan["dialogue"].items():
        safe = dict(entry)
        safe_text = tts_safe_text(safe.get("text", ""))
        safe_instruct = tts_safe_instruct(safe.get("instruct", ""))
        if safe_text != safe.get("text") or safe_instruct != safe.get("instruct"):
            repaired.append(int(key))
        safe["text"] = safe_text
        safe["instruct"] = safe_instruct
        dialogue[int(key)] = safe
    if repaired:
        print(f"  TTS normalized punctuation-only dialogue slot(s): {repaired}")
    records = build_plan(plan["skeleton"], dialogue, backend, plan["personas"],
                         out_dir=str(out / "clips_final" / mid))
    rec, sups = tts_sim.gen_meeting(records, plan["personas"], recording_id=mid)

    realized = {**plan, "dialogue": {str(k): v for k, v in dialogue.items()},
                "records": [{k: v for k, v in r.items()}
                                    for r in records]}
    json.dump(realized, open(out / "manifests" / f"{mid}_realized.json", "w"),
              indent=1)
    write_meeting_manifests(out / "manifests", rec, sups)
    elapsed = time.perf_counter() - tts_start
    print(f"  audio: {rec.duration:.1f}s -> {out/'audio'/(mid+'.wav')}")
    print(f"  TTS + mix time: {fmt_seconds(elapsed)}")
    return rec, sups, elapsed


def _close_tts_backends() -> None:
    with _TTS_BACKENDS_LOCK:
        backends = list(_TTS_BACKENDS)
        _TTS_BACKENDS.clear()
    for backend in backends:
        if hasattr(backend, "close"):
            backend.close()


def _meeting_outputs(out: Path, mid: str) -> dict[str, Path]:
    """Files that must all exist (non-empty) for a meeting to count as done."""
    mdir = out / "manifests"
    return {
        "audio": out / "audio" / f"{mid}.wav",
        "realized": mdir / f"{mid}_realized.json",
        "recordings": mdir / f"{mid}-recordings.jsonl.gz",
        "supervisions": mdir / f"{mid}-supervisions.jsonl.gz",
        "cuts": mdir / f"{mid}-cuts.jsonl.gz",
        "rttm": mdir / f"{mid}.rttm",
    }


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_completed_meeting(out: Path, mid: str):
    """Load and sanity-check a finished meeting's manifests; raises if incomplete."""
    paths = _meeting_outputs(out, mid)
    missing = [name for name, path in paths.items() if not _nonempty(path)]
    if missing:
        raise FileNotFoundError(f"{mid} missing/incomplete output(s): {', '.join(missing)}")

    recordings = RecordingSet.from_file(paths["recordings"])
    supervisions = SupervisionSet.from_file(paths["supervisions"])
    lhotse.validate_recordings_and_supervisions(recordings, supervisions)

    recs = list(recordings)
    sups = list(supervisions)
    if len(recs) != 1:
        raise RuntimeError(f"{mid} recording manifest has {len(recs)} recordings, expected 1")
    if recs[0].id != mid:
        raise RuntimeError(f"{mid} recording manifest contains recording {recs[0].id}")
    for source in recs[0].sources:
        if source.type == "file" and not _nonempty(Path(source.source)):
            raise FileNotFoundError(f"{mid} recording source missing: {source.source}")
    if any(s.recording_id != mid for s in sups):
        raise RuntimeError(f"{mid} supervision manifest contains another recording id")
    return recs[0], sups


def _meeting_complete(out: Path, mid: str) -> bool:
    try:
        _load_completed_meeting(out, mid)
    except Exception:
        return False
    return True


def synthesize_and_mix(args, plans: list[dict], all_plans: list[dict] | None = None) -> None:
    """Synthesise `plans`, then rebuild the aggregate manifests over `all_plans`.

    With --tts-jobs N, N threads each drive their own CosyVoice worker; at most
    N meetings are in flight and a new one is submitted as each finishes. The
    aggregate manifests cover the whole corpus and fail loudly if any meeting
    in it is still incomplete.
    """
    global _TTS_TIMES, _TTS_WALL
    out = Path(args.out)
    aggregate_plans = plans if all_plans is None else all_plans
    noise_files = None
    if args.add_noise and args.noise_folder:
        noise_files = [str(p) for ext in ("*.wav", "*.flac")
                       for p in Path(args.noise_folder).rglob(ext)]
        print(f"noise: {len(noise_files)} files")

    cfg = default_cfg(reverberate=args.reverberate, add_noise=args.add_noise,
                      duration=args.duration)
    rirs = None
    if args.reverberate:
        rcfg = default_cfg(reverberate=True)
        rcfg.output_dir = str(out / "rirs")
        Path(rcfg.output_dir).mkdir(parents=True, exist_ok=True)
        print("simulating RIRs (pyroomacoustics) ...")
        sim = RIRSimulator(rcfg)
        rirs = [sim.gen_rirs(f"room_{i}") for i in range(2)]
        print(f"  {len(rirs)} rooms, {len(rirs[0])} positions each")

    # The CosyVoice workers inherit this process's environment.
    if args.tts_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tts_gpu)
    if args.cosyvoice_fp16:
        os.environ["COSYVOICE_FP16"] = "1"
    if args.cosyvoice_load_trt:
        os.environ["COSYVOICE_LOAD_TRT"] = "1"
    if args.cosyvoice_load_vllm:
        os.environ["COSYVOICE_LOAD_VLLM"] = "1"
    if args.cosyvoice_no_text_frontend:
        os.environ["COSYVOICE_TEXT_FRONTEND"] = "0"
    os.environ["COSYVOICE_TRT_CONCURRENT"] = str(max(1, int(args.cosyvoice_trt_concurrent)))

    jobs = max(1, int(args.tts_jobs))
    print(f"TTS workers: {jobs}")
    pending = plans
    if args.resume_tts:
        completed = [p for p in plans if _meeting_complete(out, p["id"])]
        completed_ids = {p["id"] for p in completed}
        pending = [p for p in plans if p["id"] not in completed_ids]
        print(f"resume TTS: {len(completed)} complete, {len(pending)} pending")
        if completed:
            print(f"  first pending: {pending[0]['id']}" if pending else "  no pending meetings")

    tts_wall_start = time.perf_counter()
    tts_times = []
    try:
        if not pending:
            results = []
        elif jobs == 1:
            results = [_synthesize_one(p, args, out, cfg, rirs, noise_files) for p in pending]
        else:
            results = []
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=jobs, thread_name_prefix="tts")
            futures = {}
            next_i = 0
            try:
                while next_i < min(jobs, len(pending)):
                    plan = pending[next_i]
                    futures[ex.submit(_synthesize_one, plan, args, out, cfg, rirs, noise_files)] = plan
                    next_i += 1
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        futures.pop(fut)
                        results.append(fut.result())
                        if next_i < len(pending):
                            plan = pending[next_i]
                            futures[ex.submit(_synthesize_one, plan, args, out, cfg, rirs, noise_files)] = plan
                            next_i += 1
            except Exception:
                for fut in futures:
                    fut.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                ex.shutdown(wait=True)
        for _, _, elapsed in results:
            tts_times.append(elapsed)
    finally:
        _close_tts_backends()
    tts_wall = time.perf_counter() - tts_wall_start

    recs, all_sups, incomplete = [], [], []
    for plan in aggregate_plans:
        try:
            rec, sups = _load_completed_meeting(out, plan["id"])
        except Exception as e:
            incomplete.append((plan["id"], str(e)))
            continue
        recs.append(rec)
        all_sups.extend(sups)
    if incomplete:
        preview = "; ".join(f"{mid}: {why}" for mid, why in incomplete[:8])
        more = "" if len(incomplete) <= 8 else f"; ... {len(incomplete) - 8} more"
        raise RuntimeError(
            f"{len(incomplete)} meeting(s) still incomplete after TTS phase: {preview}{more}")
    write_combined_manifests(out / "manifests", recs, all_sups)
    _complete_active_batch(out)
    _TTS_TIMES = tts_times
    _TTS_WALL = tts_wall
    new_msg = f", {len(tts_times)} newly synthesized" if args.resume_tts else ""
    print(f"\nOK: {len(recs)} meetings, {len(all_sups)} supervisions -> {out}{new_msg}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Staged TTS meeting simulation: LLM text plans first, then CosyVoice + mixing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--phase", choices=["all", "text", "tts"], default="all",
                    help="text: write plans only; tts: synthesise saved plans; all: both.")
    ap.add_argument("--out", default="exp/tts_staged", help="Corpus directory.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Corpus base seed; meeting i uses seed + i. Keep it fixed across --append batches.")

    g = ap.add_argument_group("meeting specs")
    g.add_argument("--n", type=int, default=2, help="Meetings to generate (or, with --start-index, to synthesise).")
    g.add_argument("--duration", type=float, default=30.0,
                   help="Fixed per-meeting duration, or average duration when --total-duration is omitted.")
    g.add_argument("--total-duration", type=float, default=None,
                   help="Fixed total planned seconds across all generated meetings.")
    g.add_argument("--randomize-duration", action="store_true",
                   help="Sample per-meeting durations while preserving the fixed total duration.")
    g.add_argument("--duration-cv", type=float, default=0.35,
                   help="Duration variability for randomized meetings; higher means more spread.")
    g.add_argument("--duration-min", type=float, default=None,
                   help="Minimum randomized meeting duration in seconds; None means 0.45 x mean.")
    g.add_argument("--duration-max", type=float, default=None,
                   help="Maximum randomized meeting duration in seconds; None means 1.8 x mean.")
    g.add_argument("--speakers", type=int, default=3, help="Speakers per meeting when not randomized.")
    g.add_argument("--randomize-speakers", action="store_true",
                   help="Sample the number of speakers per meeting.")
    g.add_argument("--speakers-min", type=int, default=3,
                   help="Minimum speaker count for randomized speaker sampling.")
    g.add_argument("--speakers-max", type=int, default=8,
                   help="Maximum speaker count for randomized speaker sampling.")
    g.add_argument("--speaker-lambda", type=float, default=4.0,
                   help="Poisson lambda before truncation for randomized speaker counts.")
    g.add_argument("--notsofar-like", action="store_true",
                   help="Shorthand for --randomize-duration --randomize-speakers (NOTSOFAR-style meetings).")

    g = ap.add_argument_group("text generation")
    g.add_argument("--librispeech", default=os.environ.get("LIBRISPEECH_ROOT"),
                   help="LibriSpeech root for the reference-voice bank (default: $LIBRISPEECH_ROOT).")
    g.add_argument("--bank-limit", type=int, default=60,
                   help="Max LibriSpeech speakers in the voice bank (cached as <out>/bank.json).")
    g.add_argument("--model", default="qwen3:30b-a3b", help="LLM name on the endpoint.")
    g.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="LLM endpoint (Ollama or OpenAI-compatible).")
    g.add_argument("--fill-chunk", type=int, default=12,
                   help="Slots per LLM fill call; lower this if long conversations truncate JSON.")
    g.add_argument("--llm-max-tokens", type=int, default=None,
                   help="Default output-token cap for LLM calls (FASTMSS_LLM_MAX_TOKENS).")
    g.add_argument("--llm-cast-max-tokens", type=int, default=None,
                   help="Output-token cap for persona/premise generation (FASTMSS_LLM_CAST_MAX_TOKENS).")
    g.add_argument("--llm-fill-max-tokens", type=int, default=None,
                   help="Output-token cap for each dialogue fill chunk (FASTMSS_LLM_FILL_MAX_TOKENS).")
    g.add_argument("--no-llm", action="store_true",
                   help="Stub the dialogue instead of calling the LLM (pipeline smoke test).")
    g.add_argument("--convo-type", default=None,
                   help="Force a conversation type from fastmss/tts/scenarios.py (default: weighted draw).")
    g.add_argument("--overwrite-text", action="store_true",
                   help="Regenerate text plans instead of reusing existing ones.")

    g = ap.add_argument_group("corpus extension and resume")
    g.add_argument("--append", action="store_true",
                   help="Append a new globally indexed batch after existing text plans.")
    g.add_argument("--start-index", type=int, default=None,
                   help="Explicit global meeting index; with --phase tts, synthesize this n-meeting range.")
    g.add_argument("--unique-scenarios", action="store_true",
                   help="Do not reuse complete scenario tuples already present in the output corpus.")
    g.add_argument("--resume-tts", action="store_true",
                   help="Skip meetings with complete per-meeting audio/manifests and rebuild combined manifests.")

    g = ap.add_argument_group("acoustics (applied during mixing)")
    g.add_argument("--reverberate", action="store_true", help="Convolve with simulated RIRs (2 rooms).")
    g.add_argument("--add-noise", action="store_true",
                   help="Add noise: real noise from --noise-folder, else Gaussian.")
    g.add_argument("--noise-folder", default=None, help="Folder scanned recursively for WAV/FLAC noise.")

    g = ap.add_argument_group("Ollama hand-off")
    g.add_argument("--skip-ollama-stop", action="store_true",
                   help="Do not unload the LLM before TTS (you manage Ollama yourself).")
    g.add_argument("--allow-other-ollama-models", action="store_true",
                   help="Only require the configured LLM model to unload, not all Ollama models.")
    g.add_argument("--ollama-stop-timeout", type=float, default=20.0,
                   help="Seconds to wait for Ollama to release the GPU.")

    g = ap.add_argument_group("CosyVoice")
    g.add_argument("--backend", default="cosyvoice", help="TTS backend (only cosyvoice is implemented).")
    g.add_argument("--tts-gpu", default=None,
                   help="GPU id exposed to CosyVoice, e.g. 1. Sets CUDA_VISIBLE_DEVICES.")
    g.add_argument("--tts-jobs", type=int, default=1,
                   help="Parallel CosyVoice workers for the TTS phase. Each worker loads a model copy.")
    g.add_argument("--cosyvoice-fp16", action="store_true",
                   help="Pass fp16=True to CosyVoice3; requires CUDA.")
    g.add_argument("--cosyvoice-load-trt", action="store_true",
                   help="Enable CosyVoice TensorRT flow decoder. Builds/loads a .plan engine.")
    g.add_argument("--cosyvoice-load-vllm", action="store_true",
                   help="Enable CosyVoice vLLM backend; requires a vLLM-capable CosyVoice env/model dir.")
    g.add_argument("--cosyvoice-trt-concurrent", type=int, default=1,
                   help="TensorRT context count passed to CosyVoice when --cosyvoice-load-trt is used.")
    g.add_argument("--cosyvoice-retries", type=int, default=3,
                   help="Total attempts for a malformed or failed CosyVoice response.")
    g.add_argument("--cosyvoice-no-text-frontend", action="store_true",
                   help="Skip CosyVoice text normalization for already-normalized generated text.")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    # fill.py reads its token budgets from the environment.
    if args.llm_max_tokens is not None:
        os.environ["FASTMSS_LLM_MAX_TOKENS"] = str(args.llm_max_tokens)
    if args.llm_cast_max_tokens is not None:
        os.environ["FASTMSS_LLM_CAST_MAX_TOKENS"] = str(args.llm_cast_max_tokens)
    if args.llm_fill_max_tokens is not None:
        os.environ["FASTMSS_LLM_FILL_MAX_TOKENS"] = str(args.llm_fill_max_tokens)

    if args.n <= 0:
        ap.error("--n must be positive")
    if args.duration <= 0:
        ap.error("--duration must be positive")
    if args.total_duration is not None and args.total_duration <= 0:
        ap.error("--total-duration must be positive")
    if args.duration_min is not None and args.duration_min <= 0:
        ap.error("--duration-min must be positive")
    if args.duration_max is not None and args.duration_max <= 0:
        ap.error("--duration-max must be positive")
    if (args.duration_min is not None and args.duration_max is not None
            and args.duration_min > args.duration_max):
        ap.error("--duration-min cannot exceed --duration-max")
    if args.speakers <= 0 or args.speakers_min <= 0 or args.speakers_max <= 0:
        ap.error("speaker counts must be positive")
    if args.speaker_lambda <= 0:
        ap.error("--speaker-lambda must be positive")
    if args.speakers_min > args.speakers_max:
        ap.error("--speakers-min cannot exceed --speakers-max")

    if args.phase in {"all", "text"} and not args.librispeech:
        ap.error("--librispeech (or $LIBRISPEECH_ROOT) is required for text generation")

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    _configure_generation_range(args, out, ap)

    # Select the plans to work on: freshly generated ones, or (tts phase) all
    # saved plans / the [--start-index, +n) range of them.

    if args.phase in {"all", "text"}:
        plans = generate_text_plans(args)
        all_plans = load_plans(out)
    else:
        all_plans = load_plans(out)
        if args.start_index is None:
            plans = all_plans
        else:
            stop_index = args.start_index + args.n
            plans = [plan for plan in all_plans
                     if args.start_index <= _plan_index(plan) < stop_index]
            found = {_plan_index(plan) for plan in plans}
            missing = [index for index in range(args.start_index, stop_index)
                       if index not in found]
            if missing:
                preview = ", ".join(str(index) for index in missing[:8])
                more = "" if len(missing) <= 8 else f", ... {len(missing) - 8} more"
                raise RuntimeError(f"no text plan for requested global index/indices: {preview}{more}")
    if args.phase == "text":
        print(f"\nOK: wrote {len(plans)} text plan(s) -> {manifest_path(out)}")
        _print_timing_report(include_text=True, include_tts=False)
        return

    if not args.skip_ollama_stop and not args.no_llm:
        unload_ollama_model(args.model, wait_s=args.ollama_stop_timeout,
                            require_empty=not args.allow_other_ollama_models)
    synthesize_and_mix(args, plans, all_plans)
    _print_timing_report(include_text=args.phase == "all", include_tts=True)


if __name__ == "__main__":
    main()
