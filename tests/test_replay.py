"""Replay mode: real conversations (one talker per channel) spatialized with
their original timing. Synthetic two-sided "calls" stand in for Fisher."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import spatial_cfg
from fixtures import FS, speech_like
from lhotse import MonoCut, Recording
from lhotse.supervision import AlignmentItem, SupervisionSegment
from omegaconf import OmegaConf

from fastmss.spatial import ReplayMeetingSimulator, SceneBankSimulator, build_conversations, make_items
from fastmss.spatial.metrics import gcc_phat
from fastmss.spatial.replay import piece_bounds

BAND = (300, 3400)
DUR = 12.0
# speech segments of the two sides (seconds); A has a 0.4 s pause at 3.0-3.4
SEGS = {"A": [(0.5, 3.0), (3.4, 4.0), (6.0, 8.0)], "B": [(4.2, 5.8), (8.3, 11.0)]}
SPK = {"A": "spkA", "B": "spkB"}


def make_call(root, words_abs=False, seed=0):
    """Two per-side recordings conv1-A / conv1-B and their cuts (one cut per
    segment, as after stage 1). words_abs: word times relative to the
    recording instead of the cut."""
    rng = np.random.default_rng(seed)
    d = Path(root) / "call"
    d.mkdir(parents=True, exist_ok=True)
    cuts, chans = [], {}
    for side, segs in SEGS.items():
        n = int(DUR * FS)
        x = 1e-3 * rng.standard_normal(n)  # channel background (breaths, line noise)
        for a, b in segs:
            x[int(a * FS) : int(b * FS)] += speech_like(int(b * FS) - int(a * FS), rng)
        x = x.astype(np.float32)
        p = d / f"conv1-{side}.wav"
        sf.write(p, x, FS)
        chans[side] = sf.read(p, dtype="float32")[0]  # as stored (16-bit)
        rec = Recording.from_file(p, recording_id=f"conv1-{side}")
        for k, (a, b) in enumerate(segs):
            nw = 3
            wd = (b - a) / nw
            words = [AlignmentItem(f"w{j}", (a if words_abs else 0.0) + j * wd, 0.9 * wd) for j in range(nw)]
            cid = f"conv1-{side}-{k}"
            sup = SupervisionSegment(cid, rec.id, 0.0, b - a, channel=0, speaker=SPK[side], text="w0 w1 w2", alignment={"word": words})
            cuts.append(MonoCut(cid, a, b - a, 0, recording=rec, supervisions=[sup]))
    return cuts, chans


@pytest.fixture(scope="module")
def call(tmp_path_factory):
    return make_call(tmp_path_factory.mktemp("replay"))


def _cfg(work, devices, **sp):
    rp = {"enabled": True, **sp.pop("replay", {})}
    return spatial_cfg(
        work,
        dict(devices=devices, replay=rp, **sp),
        min_max_spk=[2, 2],
        add_noise=False,
        save_spk=True,
        save_anechoic=True,
    )


def _sim(work, cuts, devices, chunk=None, seed=3, **sp):
    cfg = _cfg(work, devices, replay={"chunk": chunk, **sp.pop("replay", {})}, **sp)
    np.random.seed(seed)
    path = SceneBankSimulator(cfg).gen_scene("room_0")
    rp = OmegaConf.to_container(cfg.spatial.replay)
    convs = build_conversations(cuts)
    items = make_items(convs, rp["chunk"])
    return ReplayMeetingSimulator(cfg, work / "audio", convs, items, scenes=[path]), items


def _read(p):
    x, _ = sf.read(p, always_2d=True)
    return x.T


def test_conversation_and_word_times(tmp_path, call):
    cuts, _ = call
    convs = build_conversations(cuts)
    assert len(convs) == 1 and convs[0]["id"] == "conv1"
    sides = {s["spk"]: s for s in convs[0]["sides"]}
    assert set(sides) == {"spkA", "spkB"}
    assert [t for s in sides["spkA"]["sups"] for t in (s["start"], s["end"])] == pytest.approx(np.ravel(SEGS["A"]).tolist())
    # cut-relative and recording-relative word times give the same absolute words
    cuts_abs, _ = make_call(tmp_path, words_abs=True)
    w_rel = [w for s in sides["spkA"]["sups"] for w in s["words"]]
    w_abs = [w for s in build_conversations(cuts_abs)[0]["sides"][0]["sups"] for w in s["words"]]
    assert [w[1] for w in w_rel] == pytest.approx([w[1] for w in w_abs])
    assert w_rel[0][1] == pytest.approx(0.5)


def test_piece_bounds():
    sups = [{"start": a, "end": b} for a, b in SEGS["A"]]
    bounds, merged = piece_bounds(sups, 0.0, DUR, min_gap=0.3)
    assert bounds[0][0] == 0.0 and bounds[-1][1] == DUR
    assert all(bounds[k][1] == bounds[k + 1][0] for k in range(len(bounds) - 1))  # contiguous
    assert [b for _, b in bounds[:-1]] == pytest.approx([3.2, 5.0])  # middle of the pauses
    # a pause shorter than min_gap is not cut
    bounds, _ = piece_bounds(sups, 0.0, DUR, min_gap=0.5)
    assert [b for _, b in bounds[:-1]] == pytest.approx([5.0])
    assert make_items([{"duration": 24.0}], chunk=10.0) == [(0, 0.0, 10.0), (0, 10.0, 20.0)]
    assert make_items([{"duration": 26.0}], chunk=10.0)[-1] == (0, 20.0, 26.0)


def test_dry_pieces_rebuild_the_channel(tmp_path, call):
    """Overlap-added dry pieces (with their cross-fades) = the original channel."""
    cuts, chans = call
    sim, _ = _sim(tmp_path, cuts, [{"preset": "single"}])
    sim._item = sim.items[0]
    tl = sim._sample_timeline()
    out = {s: np.zeros(int(DUR * FS)) for s in SPK.values()}
    for cut, off in zip(tl.utterances, tl.offsets):
        x = sim._load_dry(cut, None)
        i = int(round(off * FS))
        out[cut.supervisions[0].speaker][i : i + len(x)] += x
    for side, spk in SPK.items():
        assert np.max(np.abs(out[spk] - chans[side])) < 1e-5
    assert sum(c.supervisions[0].speaker == "spkA" for c in tl.utterances) == 3


STATIC = {"inter": "static", "intra": "none", "move_prob": 0.0}
MOVING = {"inter": "random_walk", "intra": "walk_rotate", "move_prob": 1.0}
SPEED_MAX, TRAJ_MAX = 1.2, 2.0  # speed_range[1] (config default), traj_len[1] (conftest)


@pytest.mark.parametrize("chunk", [None, 7.0])
@pytest.mark.parametrize("motion", ["static", "moving"])
def test_render_keeps_original_timing(tmp_path, call, chunk, motion):
    """The direct-path dump of each talker lines up with its original channel
    at the supervision times (static: lag 0; moving: within the propagation
    delay of the distance walked during a piece, the alignment is set at the
    piece start); supervisions are the original ones shifted to the item; the
    mixture is the sum of the talkers."""
    cuts, chans = call
    sim, items = _sim(tmp_path, cuts, [{"preset": "ami"}], chunk=chunk, motion=STATIC if motion == "static" else MOVING)
    assert len(items) == (1 if chunk is None else 2)
    tol = 0.5 if motion == "static" else SPEED_MAX * TRAJ_MAX / 343.0 * FS + 1
    for idx, (_, w0, w1) in enumerate(items):
        rec, sups = sim.gen_audio(idx)
        assert rec.duration == pytest.approx(w1 - w0, abs=0.6)  # + reverb tail
        mix = _read(rec.sources[0].source)
        d = Path(rec.sources[0].source).parent
        total = 0
        for side, spk in SPK.items():
            exp = [(max(a, w0) - w0, min(b, w1) - w0) for a, b in SEGS[side] if b > w0 and a < w1]
            got = sorted((s.start, s.end) for s in sups if s.speaker == spk)
            assert np.ravel(got).tolist() == pytest.approx(np.ravel(exp).tolist(), abs=1e-3)
            for s in sups:
                if s.speaker == spk:
                    assert all(s.start - 1e-3 <= w.start and w.end <= s.end + 1e-3 for w in s.alignment["word"])
            dire = _read(d / f"{rec.id}-spk-{spk}-anechoic.wav")[0]
            total = total + _read(d / f"{rec.id}-spk-{spk}.wav")
            ref = chans[side][int(w0 * FS) : int(w1 * FS)]
            for a, b in exp:
                ia, ib = int(a * FS), int(b * FS)
                lag = gcc_phat(ref[ia:ib], dire[ia:ib], max_lag=int(tol) + 20, band=BAND, fs=FS)
                assert abs(lag) <= tol, (spk, a, lag)
        n = min(total.shape[-1], mix.shape[-1])
        assert np.allclose(mix[0, :n], total[0, :n], atol=2e-4)


def test_one_gain_per_talker(tmp_path, call):
    """mic level: every piece of a talker gets the same gain, and the talker's
    reverberant image at the reference channel is at its level."""
    cuts, _ = call
    sim, _ = _sim(tmp_path, cuts, [{"preset": "ami"}], motion=MOVING)
    seen = {}
    orig = sim._utt_gains

    def spy(tl, placed):
        g = orig(tl, placed)
        for p, gi in zip(placed, g):
            seen.setdefault(p[0], set()).add(round(gi, 12))
        return g

    sim._utt_gains = spy
    rec, sups = sim.gen_audio(0)
    assert all(len(v) == 1 for v in seen.values()), seen
    d = Path(rec.sources[0].source).parent
    for spk in SPK.values():
        x = _read(d / f"{rec.id}-spk-{spk}.wav")[0]
        ss = [s for s in sups if s.speaker == spk]
        act = np.concatenate([x[int(s.start * FS) : int(s.end * FS)] for s in ss])
        assert 20 * np.log10(np.sqrt(np.mean(act**2))) == pytest.approx(ss[0].custom["speech_level_db"], abs=0.3)


def test_source_level_keeps_self_voice_louder(tmp_path, call):
    """Glasses worn by one talker: with level=source the wearer's own voice is
    much louder than the partner (mouth a few cm from the mics); with
    level=mic both are normalized to their sampled levels."""
    cuts, _ = call
    lv = {}
    for mode in ("source", "mic"):
        sim, _ = _sim(tmp_path / mode, cuts, [{"preset": "glasses"}], replay={"level": mode}, wearer="speaker", n_wearer_seats=1)
        np.random.seed(5)
        rec, sups = sim.gen_audio(0)
        w = {s.speaker: s.custom["is_wearer"] for s in sups}
        assert sum(w.values()) == 1
        lvl = {s.speaker: s.custom["speech_level_db"] for s in sups}
        wearer = next(s for s, v in w.items() if v)
        partner = next(s for s, v in w.items() if not v)
        lv[mode] = lvl[wearer] - lvl[partner]
    assert lv["source"] > 10.0, lv
    assert lv["source"] - lv["mic"] > 5.0, lv
