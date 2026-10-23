"""Analytic directivity patterns (pyroomacoustics) addressed by name."""

import numpy as np
from pyroomacoustics.directivities import (
    Cardioid,
    DirectionVector,
    FigureEight,
    HyperCardioid,
    SubCardioid,
)

# gain(theta) = p + (1 - p) * cos(theta)
PATTERN_P = {
    "omni": 1.0,
    "subcardioid": 0.75,
    "cardioid": 0.5,
    "hypercardioid": 0.25,
    "figure8": 0.0,
}

_CLS = {
    "subcardioid": SubCardioid,
    "cardioid": Cardioid,
    "hypercardioid": HyperCardioid,
    "figure8": FigureEight,
}


def is_omni(pattern):
    return pattern in (None, "omni")


def make_directivity(pattern, azimuth, colatitude=np.pi / 2):
    """pyroomacoustics directivity pointing at (azimuth, colatitude) in radians,
    or None for omni."""
    if is_omni(pattern):
        return None
    if pattern not in _CLS:
        raise ValueError(f"Unknown directivity pattern {pattern}; choose from {list(PATTERN_P)}")
    return _CLS[pattern](
        DirectionVector(azimuth=float(azimuth), colatitude=float(colatitude), degrees=False)
    )


def pattern_gain(pattern, theta):
    """Analytic amplitude response for the angle theta off the main axis."""
    p = PATTERN_P[pattern or "omni"]
    return p + (1 - p) * np.cos(theta)
