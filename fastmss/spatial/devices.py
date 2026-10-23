"""Recording device presets (microphone array topologies) and their geometry.

Device frame convention: x forward, y left, z up (metres). A device pose is a
center position plus a yaw (rotation around z, radians). Mic world positions
are ``center + Rz(yaw) @ offsets``.

Placement rules:
    table  -- tabletop array (AMI, NOTSOFAR-1, generic circular), fixed per room
    wall   -- mounted near a wall facing into the room (Kinect), fixed per room
    free   -- anywhere in the room (legacy-like single mic), fixed per room
    head   -- worn by a person (binaural, glasses); moves with the wearer
    mouth  -- close-talk headset, one channel per speaker, moves with the speaker

Geometries marked APPROXIMATE are not from an official spec; override them in
the config (``offsets: [[x, y, z], ...]``) if exact positions are needed.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

PLACEMENTS = ("table", "wall", "free", "head", "mouth")


@dataclass
class DeviceSpec:
    name: str
    offsets: np.ndarray  # (n_mics, 3) in the device frame
    placement: str
    ref_channel: int = 0
    # mic directivity: pattern name (see directivity.py) and how each capsule
    # is oriented: "forward" (device +x), "outward" (radially, in the xy plane),
    # "up" (+z) or explicit per-mic azimuths in degrees (device frame).
    mic_pattern: str = "omni"
    mic_orientation: object = "outward"
    height: List[float] = field(default_factory=lambda: [0.7, 1.0])
    # wall placement: distance of the array from the wall (metres)
    wall_dist: List[float] = field(default_factory=lambda: [0.1, 0.5])

    @property
    def n_mics(self):
        return self.offsets.shape[0]

    @property
    def worn(self):
        return self.placement == "head"

    @property
    def per_speaker(self):
        return self.placement == "mouth"

    def mic_positions(self, center, yaw):
        """World positions (n_mics, 3) for a device pose."""
        return np.asarray(center)[None, :] + self.offsets @ rot_z(yaw).T

    def mic_azimuths(self, yaw):
        """World azimuth (radians) each capsule points to, plus colatitudes."""
        n = self.n_mics
        col = np.full(n, np.pi / 2)
        orient = self.mic_orientation
        if isinstance(orient, (list, tuple, np.ndarray)):
            az = np.deg2rad(np.asarray(orient, dtype=float))
        elif orient == "forward":
            az = np.zeros(n)
        elif orient == "up":
            az = np.zeros(n)
            col = np.zeros(n)
        elif orient == "outward":
            xy = self.offsets[:, :2]
            az = np.arctan2(xy[:, 1], xy[:, 0])
            # capsules at the device center point forward
            az[np.linalg.norm(xy, axis=1) < 1e-6] = 0.0
        else:
            raise ValueError(f"Unknown mic_orientation {orient}")
        return az + yaw, col


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def circular_offsets(radius, n_mics, center_mic=False, z=0.0):
    ang = 2 * np.pi * np.arange(n_mics) / n_mics
    pts = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n_mics, z)], 1)
    if center_mic:
        pts = np.concatenate([np.zeros((1, 3)), pts], 0)
    return pts


def linear_offsets(xs):
    """Linear array along the device y axis (broadside = +x)."""
    xs = np.asarray(xs, dtype=float)
    return np.stack([np.zeros_like(xs), xs, np.zeros_like(xs)], 1)


# --------------------------------------------------------------------------- #
# Presets. Each returns the DeviceSpec kwargs; geometry params can be overridden
# through the config (e.g. {preset: circular, radius: 0.05, n_mics: 4}).
# --------------------------------------------------------------------------- #


def _single(p):
    return dict(offsets=np.zeros((1, 3)), placement=p.get("placement", "free"))


def _circular(p):
    radius = p.get("radius", 0.0425)
    if "diameter" in p:  # legacy config key
        d = p["diameter"]
        radius = (np.random.uniform(*d) if isinstance(d, (list, tuple)) else d) / 2
    return dict(
        offsets=circular_offsets(radius, p.get("n_mics", 8), p.get("center_mic", False)),
        placement="table",
    )


def _ami(p):
    # AMI Mic Array: 8 omni capsules on a 10 cm radius circle.
    return dict(
        offsets=circular_offsets(p.get("radius", 0.1), p.get("n_mics", 8)),
        placement="table",
    )


def _notsofar1(p):
    # 7 mics: 6 on a circle + 1 at the center (channel 0 = center).
    # APPROXIMATE radius (4.25 cm, common 6+1 reference design); verify.
    return dict(
        offsets=circular_offsets(p.get("radius", 0.0425), 6, center_mic=True),
        placement="table",
    )


def _kinect(p):
    # 4-mic linear array, Kinect v1 capsule positions (mm): -113, 36, 76, 113.
    # APPROXIMATE for Kinect v2 (CHiME-5/6 devices); verify.
    xs = p.get("positions", [-0.113, 0.036, 0.076, 0.113])
    return dict(
        offsets=linear_offsets(xs),
        placement="wall",
        mic_orientation="forward",
        height=p.get("height", [0.8, 1.8]),
    )


def _binaural_worn(p):
    # CHiME-6 style binaural in-ear mics. Head frame, origin = head center.
    half = p.get("ear_distance", 0.15) / 2
    return dict(
        offsets=np.array([[0.0, half, 0.0], [0.0, -half, 0.0]]),
        placement="head",
        mic_orientation=[90.0, -90.0],
    )


def _glasses(p):
    # EasyCom-like AR glasses: 4 mics on the frame + 2 in-ear (channels 4, 5).
    # APPROXIMATE positions read off the dataset illustration; verify.
    offs = np.array(
        [
            [0.085, 0.070, 0.035],  # front-left frame corner
            [0.085, -0.070, 0.035],  # front-right frame corner
            [0.010, 0.075, 0.030],  # left temple arm
            [0.010, -0.075, 0.030],  # right temple arm
            [0.000, 0.075, 0.000],  # left in-ear (binaural)
            [0.000, -0.075, 0.000],  # right in-ear (binaural)
        ]
    )
    return dict(
        offsets=offs,
        placement="head",
        mic_orientation=[30.0, -30.0, 90.0, -90.0, 90.0, -90.0],
    )


def _closetalk(p):
    # One headset mic per speaker; offset handled per speaker by the scene.
    return dict(offsets=np.zeros((1, 3)), placement="mouth")


PRESETS = {
    "single": _single,
    "circular": _circular,
    "ami": _ami,
    "notsofar1": _notsofar1,
    "kinect": _kinect,
    "binaural_worn": _binaural_worn,
    "glasses": _glasses,
    "closetalk": _closetalk,
}

_SPEC_KEYS = ("placement", "ref_channel", "mic_pattern", "mic_orientation", "height", "wall_dist")


def build_device(conf, default_name=None) -> DeviceSpec:
    """Build a DeviceSpec from a config dict like {preset: ami, name: ..., ...}."""
    conf = dict(deepcopy(conf))
    preset = conf.get("preset", "single")
    if preset not in PRESETS:
        raise ValueError(f"Unknown device preset {preset}; choose from {list(PRESETS)}")
    kw = PRESETS[preset](conf)
    if "offsets" in conf:
        kw["offsets"] = np.asarray(conf["offsets"], dtype=float).reshape(-1, 3)
    for k in _SPEC_KEYS:
        if k in conf:
            kw[k] = conf[k]
    spec = DeviceSpec(name=conf.get("name", default_name or preset), **kw)
    if spec.placement not in PLACEMENTS:
        raise ValueError(f"Unknown placement {spec.placement}")
    if not 0 <= spec.ref_channel < spec.n_mics:
        raise ValueError(f"ref_channel {spec.ref_channel} out of range for {spec.name}")
    return spec


def build_devices(confs) -> List[DeviceSpec]:
    devices = []
    names = set()
    for i, c in enumerate(confs):
        c = {"preset": c} if isinstance(c, str) else c
        d = build_device(c)
        if d.name in names:
            d.name = f"{d.name}{i}"
        names.add(d.name)
        devices.append(d)
    if sum(d.per_speaker for d in devices) > 1:
        raise ValueError("At most one closetalk device is supported.")
    return devices
