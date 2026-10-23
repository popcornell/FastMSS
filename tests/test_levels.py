"""Levels end to end: closetalk crosstalk, noise level at the reference
channel, speech levels, and the silent-listener wearer."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from fastmss.spatial import SpatialMeetingSimulator

FS = 16000
STATIC = {"inter": "static", "intra": "none", "move_prob": 0.0, "head_turn_prob": 0.0}
MOVING = {"inter": "random_walk", "intra": "walk_rotate", "move_prob": 1.0, "head_turn_prob": 1.0}


def _run(scene, data_dir, out, motion=STATIC, add_noise=False, seed=11, capture=None, **spatial):
    cfg, path = scene
    _, spk2cuts, noise_files = data_dir
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.save_spk = True
    cfg.save_anechoic = True
    cfg.add_noise = add_noise
    cfg.spatial.reference = {"dump": "all", "type": "direct"}
    cfg.spatial.motion = motion
    for k, v in spatial.items():
        OmegaConf.update(cfg, f"spatial.{k}", v, force_add=True)
    out.mkdir(parents=True, exist_ok=True)
    sim = SpatialMeetingSimulator(cfg, out, spk2cuts, scenes=[path], noise_files=noise_files)
    if capture is not None:
        orig = sim._render_noise

        def spy(noise, meta, *a):
            y = orig(noise, meta, *a)
            capture.append((meta, noise, a[-1], y))
            return y

        sim._render_noise = spy
    np.random.seed(seed)
    recs, sups = sim.gen_audio(0)
    recs = recs if isinstance(recs, list) else [recs]
    return sim, recs, sups, spk2cuts


def _read(p):
    x, _ = sf.read(p, always_2d=True)
    return x.T


def _db(e):
    return 10 * np.log10(e + 1e-20)


def _images(rec, sups):
    d = Path(rec.sources[0].source).parent
    spks = sorted({s.speaker for s in sups if s.recording_id == rec.id})
    return {s: _read(d / f"{rec.id}-spk-{s}.wav") for s in spks}


@pytest.mark.parametrize("scene_name", ["scene_all", "scene_closetalk_directional"])
@pytest.mark.parametrize(
    "motion,seed", [(STATIC, 11), (MOVING, 11), (MOVING, 2)], ids=["static", "moving-11", "moving-2"]
)
def test_closetalk_crosstalk(request, data_dir, tmp_path, scene_name, motion, seed):
    """Every talker (incl. a glasses wearer turning their head, walking and
    rotating directional talkers) is much louder on their own headset than on
    the others', both in their image and in the mixture on solo regions."""
    scene = request.getfixturevalue(scene_name)
    sim, recs, sups, _ = _run(scene, data_dir, tmp_path, motion=motion, seed=seed)
    rec = next(r for r in recs if r.id.endswith("_closetalk"))
    rs = [s for s in sups if s.recording_id == rec.id]
    mix = _read(rec.sources[0].source)
    imgs = _images(rec, rs)
    assert mix.shape[0] == len(imgs)
    worst_img, worst_mix = np.inf, np.inf
    for spk, img in imgs.items():
        own = next(s for s in rs if s.speaker == spk).custom["spatial"]["closetalk_channel"]
        e = np.sum(img**2, axis=1)
        worst_img = min(worst_img, _db(e[own]) - _db(np.max(np.delete(e, own))))
        for s in (s for s in rs if s.speaker == spk):
            a, b = int(s.start * FS), int(s.end * FS)
            if any(o.speaker != spk and o.start < s.end and o.end > s.start for o in rs):
                continue
            em = np.sum(mix[:, a:b] ** 2, axis=1)
            worst_mix = min(worst_mix, _db(em[own]) - _db(np.max(np.delete(em, own))))
    print(f"crosstalk margin: images {worst_img:.1f} dB, mixture {worst_mix:.1f} dB")
    assert worst_img >= 20.0, worst_img
    assert np.isfinite(worst_mix) and worst_mix >= 15.0, worst_mix


@pytest.mark.parametrize("ntype,signal", [("diffuse", "same"), ("diffuse", "shifted"), ("mono", "same")])
def test_noise_and_speech_levels(scene_all, data_dir, tmp_path, ntype, signal):
    """mix = sum of speaker images + noise; the noise RMS at the reference
    channel is min(speech level) + U(noise_rel_gain) dB and each utterance's
    reverberant reference image has its speech_level_db, up to the common
    peak-normalization gain."""
    cap = []
    sim, recs, sups, spk2cuts = _run(
        scene_all, data_dir, tmp_path, add_noise=True, capture=cap, noise={"type": ntype, "signal": signal}
    )
    cuts = {c.id: c for cl in spk2cuts.values() for c in cl}
    scene = sim._load_scene(sim.scenes[0])[0]
    lo, hi = sim.cfg.noise_rel_gain
    assert len(cap) == len(recs)
    n_utt_checked = 0
    for rec, (meta, noise, ref, y) in zip(recs, cap):
        dname = rec.id.split("_", 1)[1]
        assert scene["devices"][dname] is meta
        rs = [s for s in sups if s.recording_id == rec.id]
        r = 0 if ref is None else ref
        # configured level, set at the reference channel
        min_lvl = min(s.custom["speech_level_db"] for s in rs)
        assert lo - 1e-6 <= noise["level_db"] - min_lvl <= hi + 1e-6
        assert abs(_db(np.mean(y[r] ** 2)) - noise["level_db"]) < 0.01
        if ntype == "mono":
            assert np.allclose(y, y[r][None])
        # the written mixture holds exactly g * noise besides the speech images
        mix = _read(rec.sources[0].source)
        imgs = _images(rec, rs)
        resid = mix - sum(imgs.values())
        g = np.sum(resid[r] * y[r]) / np.sum(y[r] ** 2)
        assert 0 < g <= 1.0 + 1e-3
        err = np.linalg.norm(resid - g * y[:, : resid.shape[1]]) / np.linalg.norm(g * y)
        assert err < 0.05, (dname, err)
        # speech: energy of each isolated utterance image = N * 10^(lvl/10) * g^2
        L = np.load(Path(sim.scenes[0]).parent / meta["bank"], mmap_mode="r").shape[-1]
        spans = []
        for s in rs:
            n = cuts[s.custom["original_cut_id"]].load_audio().shape[-1] + L - 1
            a = int(s.start * FS) - 1000
            spans.append((s, a, a + n + 2000, n))
        for s, a, b, n in spans:
            if any(o is not s and o.speaker == s.speaker and oa < b and ob > a for o, oa, ob, _ in spans):
                continue
            ch = s.custom["spatial"].get("closetalk_channel", meta["ref_channel"])
            e = np.sum(imgs[s.speaker][ch, max(a, 0) : b] ** 2) / g**2
            assert abs(_db(e / n) - s.custom["speech_level_db"]) < 0.3, (dname, s.id)
            n_utt_checked += 1
    assert n_utt_checked > 0


def test_listener_wearer(scene_listener, data_dir, tmp_path):
    """wearer=listener: no mouth sources in the bank, no talker ever sits on or
    walks through the wearer's seat, and the worn device still records."""
    cfg, path = scene_listener
    scene = json.loads(Path(path).read_text())
    assert scene["mouth_entries"] == []
    assert len(scene["seats"]) >= 1
    kinds = set(scene["sources"]["kind"]) if "kind" in scene["sources"] else None
    if kinds is not None:
        assert "mouth" not in kinds
    seats = np.asarray(scene["seats"])
    sim, recs, sups, _ = _run(scene_listener, data_dir, tmp_path, motion=MOVING, seed=3)
    src = np.asarray(scene["sources"]["pos"])
    mouth = {e for m in scene["mouth_entries"] for e in m}
    assert not mouth
    for s in sups:
        for k in ("position_start", "position_end"):
            p = np.asarray(s.custom["spatial"][k])
            assert np.min(np.linalg.norm(seats[:, :2] - p[None, :2], axis=1)) >= 0.5, (s.id, k)
    # every talker source the planner can use is >= 0.5 m from the seat
    talk = [e for n in scene["nodes"] for e in n["entries"]]
    talk += [e for t in scene["trajectories"] for e in t["entries"] + t["entries_rev"]]
    d = np.linalg.norm(src[talk][:, None, :2] - seats[None, :, :2], axis=-1)
    assert d.min() >= 0.5
    # the listener is not a talker: closetalk has one channel per speaker only
    spks = sorted({s.speaker for s in sups})
    ct = next(r for r in recs if r.id.endswith("_closetalk"))
    assert ct.num_channels == len(spks)
    gl = next(r for r in recs if r.id.endswith("_glasses"))
    assert np.max(np.abs(_read(gl.sources[0].source))) > 0


def test_listener_bank_rejects_speaker_wearer(scene_listener, data_dir, tmp_path):
    cfg, path = scene_listener
    with pytest.raises(ValueError, match="listener"):
        _run(scene_listener, data_dir, tmp_path, wearer="speaker")
