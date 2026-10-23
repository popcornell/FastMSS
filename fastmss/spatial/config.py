"""Defaults for the ``spatial`` config block (multi-channel simulation)."""

from copy import deepcopy

SPATIAL_DEFAULTS = {
    "enabled": False,
    "devices": [{"preset": "single"}],
    # directional talkers; n_orient yaw bins are precomputed per position
    "source_directivity": {"pattern": "omni", "n_orient": 4},
    # where talkers face when not moving: group (centroid of the other talkers),
    # device (first fixed device) or random
    "face": "group",
    "wearer": "speaker",  # speaker | listener (worn devices)
    "n_wearer_seats": 1,
    "n_yaw": 8,  # head-yaw bins precomputed for worn devices
    "mouth_offset": [0.09, 0.0, -0.07],  # mouth w.r.t. head center (head frame)
    "closetalk_offset": [0.03, 0.0, -0.02],  # headset mic w.r.t. mouth (facing frame)
    "n_trajectories": 10,
    "traj_len": [1.0, 4.0],
    "traj_step": 0.05,
    "node_height": [1.0, 1.8],
    "head_height": [1.1, 1.3],
    "min_dist_device": 1.0,
    "rir_max_len": 1.0,
    "rir_dtype": "float16",
    "motion": {
        "inter": "random",  # random | static | random_walk
        "max_position_change": 0.5,
        "intra": "none",  # none | walk | rotate | walk_rotate
        "move_prob": 0.0,
        "speed_range": [0.3, 1.2],
        "rotate_range": [30.0, 120.0],  # degrees per utterance
        "head_turn_prob": 0.5,  # wearer turns toward the active speaker
        "head_jitter": 20.0,  # degrees, random head-yaw jitter per turn
    },
    "noise": {
        "type": "diffuse",  # diffuse | mono (same signal on every channel)
        # diffuse: every corner source plays the same noise clip ("same") or a
        # time-shifted copy of it ("shifted"). Identical signals add up to a
        # fully coherent field; shifted copies decorrelate it (more diffuse).
        "signal": "same",
        "n_sources": 4,
        "pattern": "hypercardioid",
        "corner_dist": 1.0,
    },
    "reference": {
        "channel": None,  # null = device ref_channel
        "dump": "ref",  # ref | all
        "type": "direct",  # direct | early | both
        "early_ms": 50.0,
    },
}


def _merge(base, over):
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def get_spatial_cfg(cfg):
    """Return the spatial block merged with defaults as a plain dict.

    Works with DictConfig, SimpleNamespace or dict top-level configs, and returns
    {"enabled": False, ...} when the block is missing (legacy behaviour)."""
    if isinstance(cfg, dict):
        sp = cfg.get("spatial")
    else:
        sp = getattr(cfg, "spatial", None)
    if sp is None:
        return deepcopy(SPATIAL_DEFAULTS)
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(sp, DictConfig):
            sp = OmegaConf.to_container(sp, resolve=True)
    except ImportError:
        pass
    return _merge(SPATIAL_DEFAULTS, sp)


def spatial_enabled(cfg):
    return bool(get_spatial_cfg(cfg)["enabled"])
