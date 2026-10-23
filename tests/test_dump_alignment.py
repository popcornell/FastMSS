"""End-to-end: the dumped speaker images (reverberant, anechoic/direct, early)
must align with the reference channel of the mixture, read back from disk."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from fastmss.spatial import SpatialMeetingSimulator
from fastmss.spatial.metrics import gcc_phat
from fastmss.spatial.scene import direct_delay

FS = 16000
# the synthetic test "speech" is band-limited to 300-3400 Hz
BAND = (300, 3400)
DEVICES = ["ami", "kinect", "glasses", "closetalk", "single"]


def _run(scene_all, data_dir, out, dump, motion, add_noise, seed=11):
    cfg, path = scene_all
    _, spk2cuts, noise_files = data_dir
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.save_spk = True
    cfg.save_anechoic = True
    cfg.add_noise = add_noise
    cfg.spatial.reference = {"dump": dump, "type": "both"}
    cfg.spatial.motion = motion
    out.mkdir(parents=True, exist_ok=True)
    sim = SpatialMeetingSimulator(cfg, out, spk2cuts, scenes=[path], noise_files=noise_files)
    np.random.seed(seed)
    recs, sups = sim.gen_audio(0)
    return sim, recs, sups, spk2cuts


def _read(p):
    x, _ = sf.read(p, always_2d=True)
    return x.T


def _solo_regions(sups, spk):
    """(start, end) samples where spk is the only active talker."""
    mine = [s for s in sups if s.speaker == spk]
    others = [s for s in sups if s.speaker != spk]
    out = []
    for s in mine:
        a, b = s.start, s.end
        if all(o.end <= a or o.start >= b for o in others):
            out.append((int(a * FS), int(b * FS)))
    return out


def _by_rec(sups):
    d = defaultdict(list)
    for s in sups:
        d[s.recording_id].append(s)
    return d


STATIC = {"inter": "static", "intra": "none", "move_prob": 0.0, "head_turn_prob": 0.0}
MOVING = {"inter": "random_walk", "intra": "walk_rotate", "move_prob": 1.0, "head_turn_prob": 1.0}


@pytest.mark.parametrize("dump", ["ref", "all"])
def test_dumps_align_with_mixture_and_dry(scene_all, data_dir, tmp_path, dump):
    sim, recs, sups, spk2cuts = _run(scene_all, data_dir, tmp_path / "audio", dump, STATIC, add_noise=True)
    cuts = {c.id: c for cl in spk2cuts.values() for c in cl}
    scene = sim._load_scene(sim.scenes[0])[0]
    rec_sups = _by_rec(sups)
    checked = 0
    for rec in recs:
        dev = rec.id.split("_", 1)[1]
        meta = scene["devices"][dev]
        mix = _read(rec.sources[0].source)
        audio_dir = Path(rec.sources[0].source).parent
        rs = rec_sups[rec.id]
        for spk in sorted({s.speaker for s in rs}):
            ss = [s for s in rs if s.speaker == spk]
            ref = ss[0].custom["spatial"].get("closetalk_channel", meta["ref_channel"])
            imgs = {k: _read(audio_dir / f"{rec.id}-spk-{spk}{suf}.wav") for k, suf in (("rev", ""), ("direct", "-anechoic"), ("early", "-early"))}
            n_ch = mix.shape[0]
            for k, img in imgs.items():
                assert img.shape[0] == (n_ch if dump == "all" else 1), (dev, k, img.shape)
            ref_in_dump = ref if dump == "all" else 0

            # (a) every image vs the mixture's reference channel on solo regions: lag 0
            for a, b in _solo_regions(rs, spk):
                for k, img in imgs.items():
                    lag = gcc_phat(mix[ref, a:b], img[ref_in_dump, a:b], max_lag=40, band=BAND)
                    assert abs(lag) <= 0.5, (dev, spk, k, lag)
                    checked += 1

            # (d) dry source vs images at the supervision start: lag 0
            for s in ss:
                dry = cuts[s.custom["original_cut_id"]].load_audio()[0]
                dry = dry - dry.mean()
                a = int(s.start * FS)
                # only the fractional part of the (rounded) shift may remain
                for k, tol in (("direct", 0.6), ("early", 0.6), ("rev", 0.6)):
                    seg = imgs[k][ref_in_dump, a : a + len(dry)]
                    lag = gcc_phat(dry[: len(seg)], seg, max_lag=40, band=BAND)
                    assert abs(lag) <= tol, (dev, spk, k, lag)
                # word alignments still in meeting time
                w0 = cuts[s.custom["original_cut_id"]].supervisions[0].alignment["word"][0]
                assert np.isclose(s.alignment["word"][0].start, s.start + w0.start)

            # (b) dump=all: inter-channel delays of the anechoic image follow geometry
            if dump == "all" and meta["placement"] in ("table", "wall"):
                mp = np.asarray(meta["poses"][0]["mic_pos"])
                s = ss[0]
                geo = direct_delay(s.custom["spatial"]["position_start"], mp, FS)
                a = int(s.start * FS)
                b = a + int(s.duration * FS)
                max_lag = np.max(np.linalg.norm(mp - mp[ref], axis=1)) / 343 * FS + 3
                for m in range(n_ch):
                    est = gcc_phat(imgs["direct"][ref, a:b], imgs["direct"][m, a:b], max_lag=max_lag, band=BAND)
                    assert abs(est - (geo[m] - geo[ref])) <= 1.0, (dev, m, est, geo[m] - geo[ref])
    assert checked > 20


@pytest.mark.parametrize("dump", ["ref", "all"])
def test_reverberant_images_sum_to_mixture(scene_all, data_dir, tmp_path, dump):
    sim, recs, sups, _ = _run(scene_all, data_dir, tmp_path / "audio", dump, MOVING, add_noise=False)
    for rec in recs:
        mix = _read(rec.sources[0].source)
        audio_dir = Path(rec.sources[0].source).parent
        spks = sorted({s.speaker for s in sups if s.recording_id == rec.id})
        total = np.zeros_like(mix)
        for spk in spks:
            img = _read(audio_dir / f"{rec.id}-spk-{spk}.wav")
            if dump == "all":
                total += img
            else:
                continue
        if dump == "all":
            # 16-bit wav quantization of each file only
            assert np.max(np.abs(total - mix)) < 1e-3 * (len(spks) + 1), rec.id


def test_moving_dumps_stay_aligned(scene_all, data_dir, tmp_path):
    """With walking/rotating talkers and turning heads, images still line up
    with the mixture (lag 0 on solo regions) and with the dry source at onset."""
    sim, recs, sups, spk2cuts = _run(scene_all, data_dir, tmp_path / "audio", "ref", MOVING, add_noise=False, seed=5)
    moving = [s for s in sups if s.custom["spatial"]["moving"]]
    assert moving, "expected at least one walking utterance"
    cuts = {c.id: c for cl in spk2cuts.values() for c in cl}
    scene = sim._load_scene(sim.scenes[0])[0]
    rec_sups = _by_rec(sups)
    for rec in recs:
        dev = rec.id.split("_", 1)[1]
        meta = scene["devices"][dev]
        mix = _read(rec.sources[0].source)
        audio_dir = Path(rec.sources[0].source).parent
        rs = rec_sups[rec.id]
        for spk in sorted({s.speaker for s in rs}):
            ss = [s for s in rs if s.speaker == spk]
            ref = ss[0].custom["spatial"].get("closetalk_channel", meta["ref_channel"])
            rev = _read(audio_dir / f"{rec.id}-spk-{spk}.wav")[0]
            dire = _read(audio_dir / f"{rec.id}-spk-{spk}-anechoic.wav")[0]
            for a, b in _solo_regions(rs, spk):
                assert abs(gcc_phat(mix[ref, a:b], rev[a:b], max_lag=40, band=BAND)) <= 0.5
            for s in ss:
                dry = cuts[s.custom["original_cut_id"]].load_audio()[0]
                a = int(s.start * FS)
                n = min(int(0.4 * FS), len(dry))  # onset: before the talker moved far
                lag = gcc_phat(dry[:n] - dry.mean(), dire[a : a + n], max_lag=40, band=BAND)
                assert abs(lag) <= 1.0, (dev, spk, lag)
