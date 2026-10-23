"""Multi-channel spatial simulation: devices, precomputed scene banks, motion
and time-varying rendering. Enabled with ``spatial.enabled: true``."""

from .config import SPATIAL_DEFAULTS, get_spatial_cfg, spatial_enabled
from .devices import PRESETS, DeviceSpec, build_device, build_devices
from .meeting import SpatialMeetingSimulator
from .motion import INTER, INTRA, MotionPlanner, register_inter, register_intra
from .scene import SceneBankSimulator

__all__ = [
    "SPATIAL_DEFAULTS",
    "get_spatial_cfg",
    "spatial_enabled",
    "PRESETS",
    "DeviceSpec",
    "build_device",
    "build_devices",
    "SpatialMeetingSimulator",
    "INTER",
    "INTRA",
    "MotionPlanner",
    "register_inter",
    "register_intra",
    "SceneBankSimulator",
]
