"""Guard that the legacy (non-spatial) simulation path stays bit-identical.

Usage:
    python tests/legacy_regression.py --save tests/data/legacy_baseline.json   # on the reference code
    python tests/legacy_regression.py --check tests/data/legacy_baseline.json  # after changes

Hashes every written audio file, the supervisions (minus random ids) and the
numpy RNG state after every meeting, so a change in RNG call order is caught too.
The baseline depends on library versions (pyroomacoustics, scipy, numpy), so
regenerate it on the reference commit when changing environment.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_cfg, make_noise_files, make_spk2cuts  # noqa: E402

from fastmss.rirsimulator import RIRSimulator  # noqa: E402
from fastmss.simulator import ConversationalMeetingSimulator  # noqa: E402

_SCRATCH = None  # system temp dir

SCENARIOS = {
    "clean": dict(),
    "reverb_noise_dumps": dict(
        reverberate=True,
        reverb_prob=1.0,
        add_noise=True,
        noise_probability_global=1.0,
        save_spk=True,
        save_anechoic=True,
    ),
    "single_noise_fir": dict(
        reverberate=True,
        reverb_prob=1.0,
        add_noise=True,
        noise_probability_global=1.0,
        multi_noise_sampling=False,
        use_fir=True,
        save_spk=True,
    ),
    "gaussian_noise_markov": dict(
        add_noise=True,
        noise_probability_global=1.0,
        use_markov=True,
        max_concurrent_speakers=2,
        hmm_params={"p_ind": [0.1, 0.4, 0.3, 0.2]},
        boost_overlap_factor=2.0,
    ),
}


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def run(work):
    work = Path(work)
    spk2cuts = make_spk2cuts(work)
    noise_files = make_noise_files(work)
    np.random.seed(123)
    rir_cfg = base_cfg(work / "rir", n_pos_rirs=6, rt60=[0.2, 0.4])
    rooms = [RIRSimulator(rir_cfg).gen_rirs(f"room_{i}") for i in range(2)]

    out = {}
    for name, ov in SCENARIOS.items():
        cfg = base_cfg(work / name, duration=8, min_max_spk=[2, 3], **ov)
        (work / name).mkdir(parents=True, exist_ok=True)
        noise = noise_files if (cfg.add_noise and name != "gaussian_noise_markov") else None
        sim = ConversationalMeetingSimulator(
            cfg,
            work / name / "audio",
            spk2cuts,
            rirs=rooms if cfg.reverberate else None,
            noise_files=noise,
        )
        np.random.seed(7)
        res = []
        for i in range(3):
            rec, sups = sim.gen_audio(i)
            audio_dir = Path(rec.sources[0].source).parent
            files = {}
            for f in sorted(audio_dir.glob(f"{rec.id}*.wav")):
                data, _ = sf.read(f, dtype="int16", always_2d=True)
                files[f.name.replace(rec.id, "ID")] = _sha(data)
            sup_d = []
            for s in sups:
                d = s.to_dict()
                d.pop("id"), d.pop("recording_id")
                sup_d.append(d)
            res.append(
                {
                    "files": files,
                    "num_samples": rec.num_samples,
                    "sups": hashlib.sha256(
                        json.dumps(sup_d, sort_keys=True, default=str).encode()
                    ).hexdigest()[:16],
                    "rng": _sha(np.append(np.random.get_state()[1], np.random.get_state()[2])),
                }
            )
        out[name] = res
    return out


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--check")
    args = p.parse_args()
    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        res = run(tmp)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(res, indent=1, sort_keys=True))
        print(f"saved baseline to {args.save}")
        return
    ref = json.loads(Path(args.check).read_text())
    if ref == res:
        print("LEGACY REGRESSION OK: outputs are bit-identical")
        return
    for name in ref:
        if ref[name] != res.get(name):
            print(f"MISMATCH in scenario {name}")
            print(" ref:", json.dumps(ref[name], indent=1))
            print(" new:", json.dumps(res.get(name), indent=1))
    sys.exit(1)


if __name__ == "__main__":
    main()
