import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import base_cfg, make_noise_files, make_spk2cuts  # noqa: E402

from fastmss.spatial import SceneBankSimulator  # noqa: E402

ALL_DEVICES = [
    {"preset": "ami"},
    {"preset": "kinect"},
    {"preset": "notsofar1"},
    {"preset": "glasses"},
    {"preset": "binaural_worn"},
    {"preset": "single"},
    {"preset": "closetalk"},
]


def spatial_cfg(work, spatial, **overrides):
    kw = dict(
        n_pos_rirs=6,
        rt60=[0.25, 0.35],
        room_sz=[5, 7],
        room_ceiling=[2.8, 3.2],
        reverberate=True,
        reverb_prob=1.0,
        add_noise=True,
        noise_probability_global=1.0,
        duration=6,
        min_max_spk=[3, 3],
    )
    kw.update(overrides)
    sp = dict(enabled=True, n_trajectories=2, traj_len=[1.0, 2.0], rir_max_len=0.3, rir_dtype="float32")
    sp.update(spatial)
    return base_cfg(work, spatial=sp, **kw)


def load_scene(path):
    with open(path) as f:
        scene = json.load(f)
    d = Path(path).parent
    banks = {
        n: (np.load(d / m["bank"]), np.load(d / m["direct_bank"]))
        for n, m in scene["devices"].items()
    }
    return scene, banks


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return d, make_spk2cuts(d), make_noise_files(d)


@pytest.fixture(scope="session")
def scene_all(tmp_path_factory):
    """Omni talkers, every device preset."""
    work = tmp_path_factory.mktemp("scene_all")
    np.random.seed(0)
    cfg = spatial_cfg(work, dict(devices=ALL_DEVICES, n_wearer_seats=1, n_yaw=8))
    path = SceneBankSimulator(cfg).gen_scene("room_0")
    return cfg, path


@pytest.fixture(scope="session")
def scene_directional(tmp_path_factory):
    """Cardioid talkers with 8 yaw bins, tabletop array + single mic."""
    work = tmp_path_factory.mktemp("scene_dir")
    np.random.seed(1)
    cfg = spatial_cfg(
        work,
        dict(
            devices=[{"preset": "ami"}, {"preset": "single"}],
            source_directivity={"pattern": "cardioid", "n_orient": 8},
        ),
    )
    path = SceneBankSimulator(cfg).gen_scene("room_0")
    return cfg, path
