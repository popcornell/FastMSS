"""Room-acoustics sanity checks on the precomputed banks and diffuse noise."""

import numpy as np
from conftest import load_scene

from fastmss.spatial.metrics import diffuse_coherence, msc, schroeder_rt60
from fastmss.spatial.render import render
from fastmss.spatial.scene import direct_delay

FS = 16000


def test_direct_index_matches_geometry(scene_all):
    cfg, path = scene_all
    scene, banks = load_scene(path)
    src_pos = np.asarray(scene["sources"]["pos"])
    for name, meta in scene["devices"].items():
        _, dbank = banks[name]
        for pi, pose in enumerate(meta["poses"][:2]):
            mp = np.asarray(pose["mic_pos"])
            for node in scene["nodes"]:
                s = node["entries"][0]
                geo = direct_delay(src_pos[s], mp, FS)
                for c in range(mp.shape[0]):
                    h = dbank[pi, s, c]
                    assert abs(np.argmax(np.abs(h)) - geo[c]) <= 1.0, (name, c)


def test_rt60_close_to_target(scene_all):
    cfg, path = scene_all
    scene, banks = load_scene(path)
    bank, _ = banks["single"]
    est = [schroeder_rt60(bank[0, n["entries"][0], 0], FS) for n in scene["nodes"]]
    # ISM vs Sabine and the truncated (rir_max_len) tail: loose tolerance
    assert 0.5 * scene["rt60"] <= np.nanmedian(est) <= 1.6 * scene["rt60"], (est, scene["rt60"])


def test_diffuse_noise_coherence(scene_all, data_dir, tmp_path):
    """Coherence of the corner-source noise across a 20 cm mic pair.

    noise.signal=same: all sources play the same clip, so the field is one
    signal through a fixed filter -> as coherent as a single point source.
    noise.signal=shifted: time-shifted copies decorrelate the sources -> much
    lower coherence, closer to the diffuse-field sinc^2."""
    from fastmss.spatial import SpatialMeetingSimulator

    cfg, path = scene_all
    _, spk2cuts, _ = data_dir
    scene, banks = load_scene(path)
    bank, _ = banks["ami"]
    meta = scene["devices"]["ami"]
    mp = np.asarray(meta["poses"][0]["mic_pos"])
    m0, m1 = 0, 4  # opposite mics, 20 cm apart
    d = np.linalg.norm(mp[m0] - mp[m1])
    T = 10 * FS
    x = np.random.default_rng(0).standard_normal(T)
    noise = {"signal": x, "level_db": -20.0, "shifts": [k * T // 4 + 1234 * k for k in range(4)]}

    def coh(y):
        f, c = msc(y[m0], y[m1], FS, nperseg=2048)
        band = (f > 300) & (f < 4000)
        return c[band], diffuse_coherence(f[band], d)

    res = {}
    for mode in ("same", "shifted"):
        sim = SpatialMeetingSimulator(cfg, tmp_path, spk2cuts, scenes=[path])
        sim._resolve_cfg()
        sim.sp["noise"]["signal"] = mode
        y = sim._render_noise(noise, meta, bank, scene, None, None, [], T, 0)
        res[mode], theo = coh(y)
    point = render(x, [0], bank[0, scene["noise_entries"][0]][None].astype(np.float64))
    res["point"], _ = coh(point)

    assert abs(np.mean(res["same"]) - np.mean(res["point"])) < 0.1
    assert np.mean(res["shifted"]) < 0.6 * np.mean(res["same"]), {k: v.mean() for k, v in res.items()}
    assert np.mean(np.abs(res["shifted"] - theo)) < np.mean(np.abs(res["same"] - theo))
