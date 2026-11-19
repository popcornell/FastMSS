"""Rooms differ across room ids and do not depend on the caller's global RNG state
(forked workers inherit one state; before room_seed every worker drew the same rooms)."""
import json

import numpy as np

from conftest import spatial_cfg
from fastmss.spatial import SceneBankSimulator
from fastmss.spatial.scene import room_seed


def test_room_seed_distinct_and_stable():
    seeds = {room_seed(0, f"room_{i}") for i in range(1000)}
    assert len(seeds) == 1000
    assert room_seed(0, "room_5") == room_seed(0, "room_5") != room_seed(1, "room_5")


def test_gen_scene_rooms_differ(tmp_path):
    cfg = spatial_cfg(tmp_path, dict(devices=[{"preset": "single"}]), seed=3)
    sim = SceneBankSimulator(cfg)
    dims = []
    for rid in ("room_0", "room_1", "room_0"):
        np.random.seed(123)                     # identical state before each call
        dims.append(json.load(open(sim.gen_scene_seeded(rid)))["room_dim"])
    assert dims[0] != dims[1]
    assert dims[0] == dims[2]
