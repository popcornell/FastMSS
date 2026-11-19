"""Measured array transfer functions (ATFs) for worn devices.

Loads a far-field ATF set measured on a head-and-torso simulator wearing the
device (format of the EasyCom ``Device_ATFs.h5``: datasets ``IR`` [sample,
direction, channel], ``Theta`` (polar angle, 0 = up), ``Phi`` (azimuth, 0 =
forward, counter-clockwise = towards the left) and ``SamplingFreq_Hz``).

Every channel is simulated at the head center with a pyroomacoustics
``MeasuredDirectivity``: each image source is filtered with the measured
impulse response of the nearest measured direction. The ATFs already contain
the inter-channel delays, head shadowing and diffraction, so the effective mic
positions (a plane-wave fit of the measured delays) are only metadata for
geometry-based tools.

Preprocessing:
  * the common measurement delay (loudspeaker distance) is removed so that
    sample ``pre`` of each IR is the arrival at the head center;
  * IRs are cut to ``pre + tail`` samples with a half-Hann fade-out;
  * resampled to the simulation rate;
  * one scalar diffuse-field normalisation (mean energy over directions and
    channels = 1); inter-channel and directional level differences are kept.

The far-field ATFs are also used for the wearer's own mouth (direction from
the head center), which ignores near-field effects, unless the set carries a
near-field mouth ATF (Aria, ``load_near``; scene.py then uses it for the
direct path of the wearer's own voice).

Formats (by file suffix):
  * ``.h5``: EasyCom ``Device_ATFs.h5`` (above).
  * ``.pkl``: CHiME-8 MMCSG / MCAS ``aria_atfs.pkl`` (Aria glasses, 7 mics in
    the MMCSG channel order): ``far_field`` (sample, ch, direction),
    ``directions`` (3, direction) = azimuth (deg, 0 = forward, 90 = left),
    elevation (deg, 90 = up), distance (1.53 m); ``near_field`` (sample, ch)
    from the mouth; 48 kHz. The device frame is the Aria one (x front, y left,
    z up), as in FastMSS.
"""

from functools import lru_cache
from math import gcd

import numpy as np

SPEED_OF_SOUND = 343.0


def _peak(ir, up=8):
    """Sub-sample position of the absolute peak of each IR (..., n)."""
    n = ir.shape[-1]
    y = np.fft.irfft(np.fft.rfft(ir, axis=-1), n * up, axis=-1)
    return np.argmax(np.abs(y), axis=-1) / up


def directions(theta, phi):
    """Unit vectors (n, 3) in the device frame (x fwd, y left, z up)."""
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1)


ARIA_FS = 48000.0


def _read_h5(path):
    import h5py

    with h5py.File(path, "r") as f:
        ir = np.asarray(f["IR"][()], dtype=np.float64)  # (sample, dir, ch)
        theta = np.asarray(f["Theta"][()], dtype=np.float64).ravel()
        phi = np.asarray(f["Phi"][()], dtype=np.float64).ravel()
        fs = float(np.asarray(f["SamplingFreq_Hz"][()]).ravel()[0])
    return np.ascontiguousarray(ir.transpose(2, 1, 0)), directions(theta, phi), fs, None  # ir (ch, dir, sample)


def _read_aria_pkl(path):
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)
    ir = np.ascontiguousarray(np.asarray(d["far_field"], dtype=np.float64).transpose(1, 2, 0))  # (ch, dir, sample)
    az, el = np.deg2rad(np.asarray(d["directions"], dtype=np.float64)[:2])
    u = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    near = d.get("near_field")
    near = None if near is None else np.ascontiguousarray(np.asarray(near, dtype=np.float64).T)  # (ch, sample)
    return ir, u, ARIA_FS, near


@lru_cache(maxsize=4)
def load_raw(path):
    """Raw ATF set: dict(ir [n_ch, n_dir, n], u [n_dir, 3], fs, t0, offsets [n_ch, 3], fit_rms,
    near [n_ch, n] or None)."""
    ir, u, fs, near = (_read_aria_pkl if str(path).endswith(".pkl") else _read_h5)(path)
    n_ch, n_dir, _ = ir.shape
    # plane-wave fit of the peak delays: tau[d, m] = t0 - u_d . r_m / c * fs
    tau = _peak(ir).T  # (dir, ch)
    A = np.zeros((n_dir * n_ch, 1 + 3 * n_ch))
    A[:, 0] = 1.0
    for m in range(n_ch):
        A[m::n_ch, 1 + 3 * m : 4 + 3 * m] = -u / SPEED_OF_SOUND * fs
    x, *_ = np.linalg.lstsq(A, tau.ravel(), rcond=None)
    res = A @ x - tau.ravel()
    return {
        "ir": ir,
        "u": u,
        "fs": fs,
        "t0": float(x[0]),
        "offsets": x[1:].reshape(n_ch, 3),
        "fit_rms": float(np.sqrt(np.mean(res**2))),
        "near": near,
    }


def atf_offsets(path):
    """Effective mic positions (n_ch, 3) in metres, head frame."""
    return load_raw(path)["offsets"].copy()


@lru_cache(maxsize=8)
def load_atf(path, fs, pre_ms=0.5, tail_ms=3.0):
    """Preprocessed ATFs at ``fs``: dict(ir [n_ch, n_dir, n], u [n_dir, 3], pre (samples))."""
    from scipy.signal import resample_poly

    raw = load_raw(path)
    fs0 = raw["fs"]
    pre0 = int(round(pre_ms * 1e-3 * fs0))
    n0 = pre0 + int(round(tail_ms * 1e-3 * fs0))
    start = int(np.floor(raw["t0"])) - pre0
    if start < 0:
        raise ValueError(f"ATF {path}: not enough samples before the direct sound")
    ir = raw["ir"][..., start : start + n0].copy()
    # residual fractional part of t0 stays in the IRs (< 1 sample at fs0)
    fade = int(round(0.5e-3 * fs0))
    ir[..., -fade:] *= np.hanning(2 * fade)[fade:]
    if fs != fs0:
        g = gcd(int(fs), int(fs0))
        ir = resample_poly(ir, int(fs) // g, int(fs0) // g, axis=-1)
    ir = ir / np.sqrt(np.mean(np.sum(ir**2, axis=-1)))
    pre = (pre0 + raw["t0"] - np.floor(raw["t0"])) * fs / fs0
    return {"ir": ir.astype(np.float64), "u": raw["u"], "pre": float(pre)}


@lru_cache(maxsize=8)
def load_near(path, fs):
    """Near-field mouth ATF at ``fs`` (n_ch, n), or None if the set has none.
    Whole measured response, resampled; level and timing are set by the scene
    (it is aligned and level-matched to an omni point source at the mouth)."""
    from scipy.signal import resample_poly

    raw = load_raw(path)
    if raw["near"] is None:
        return None
    h = raw["near"].copy()
    fs0 = raw["fs"]
    if fs != fs0:
        g = gcd(int(fs), int(fs0))
        h = resample_poly(h, int(fs) // g, int(fs0) // g, axis=-1)
    return h


def make_atf_directivities(path, fs, yaw):
    """One pyroomacoustics MeasuredDirectivity per channel for head yaw (radians)."""
    from pyroomacoustics.directivities import MeasuredDirectivity, Rotation3D
    from pyroomacoustics.doa import GridSphere

    atf = load_atf(path, fs)
    grid = GridSphere(cartesian_points=atf["u"].T)
    rot = Rotation3D([float(yaw)], rot_order="z", degrees=False)
    return [MeasuredDirectivity(rot, grid, atf["ir"][m], fs) for m in range(atf["ir"].shape[0])]


@lru_cache(maxsize=8)
def diffuse_filters(path, fs, n_taps=127):
    """Zero-phase FIRs (n_ch, n_taps), centered at n_taps // 2: per-channel
    diffuse-field magnitude response (RMS over all measured directions)."""
    ir = load_atf(path, fs)["ir"]
    nfft = 1 << int(np.ceil(np.log2(max(ir.shape[-1], n_taps) * 2)))
    mag = np.sqrt(np.mean(np.abs(np.fft.rfft(ir, nfft, axis=-1)) ** 2, axis=1))  # (ch, f)
    h = np.fft.fftshift(np.fft.irfft(mag, nfft, axis=-1), axes=-1)
    mid = nfft // 2
    h = h[:, mid - n_taps // 2 : mid + n_taps // 2 + 1] * np.hanning(n_taps + 2)[1:-1]
    return h


def phat_delay(h, fs, band=(200.0, 8000.0), up=16):
    """Delay (fractional samples) of impulse responses h (..., n): peak of the
    band-limited PHAT-whitened response."""
    shape, n = h.shape[:-1], h.shape[-1]
    h = h.reshape(-1, n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    w = ((f >= band[0]) & (f <= min(band[1], 0.45 * fs))).astype(float)
    out = np.zeros(len(h))
    for i in range(0, len(h), 256):
        H = np.fft.rfft(h[i : i + 256], nfft, axis=-1)
        G = H / np.maximum(np.abs(H), 1e-12) * w
        y = np.fft.irfft(G, nfft * up, axis=-1)
        out[i : i + 256] = np.argmax(y[:, : n * up], axis=-1) / up
    return out.reshape(shape)
