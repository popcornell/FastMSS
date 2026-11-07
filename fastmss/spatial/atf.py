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
the head center), which ignores near-field effects.
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


@lru_cache(maxsize=4)
def load_raw(path):
    """Raw ATF set: dict(ir [n_ch, n_dir, n], u [n_dir, 3], fs, t0, offsets [n_ch, 3], fit_rms)."""
    import h5py

    with h5py.File(path, "r") as f:
        ir = np.asarray(f["IR"][()], dtype=np.float64)  # (sample, dir, ch)
        theta = np.asarray(f["Theta"][()], dtype=np.float64).ravel()
        phi = np.asarray(f["Phi"][()], dtype=np.float64).ravel()
        fs = float(np.asarray(f["SamplingFreq_Hz"][()]).ravel()[0])
    ir = np.ascontiguousarray(ir.transpose(2, 1, 0))  # (ch, dir, sample)
    n_ch, n_dir, _ = ir.shape
    u = directions(theta, phi)
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
