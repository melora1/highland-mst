#!/usr/bin/env python3
"""Acceptance-aware Moliere calibration from the manuscript's RADIAL density.

The defining quantity is

    eps_M(p, path, cut)
        = theta_RMS(cut; p, path) / theta_space_highland(p, x/X0) - 1,

where theta_RMS^2=M_2 is evaluated from the non-factorized two-dimensional
Moliere density P_M(Theta).  The projected-marginal product
F(theta_x)F(theta_y) is intentionally not used here.

All functions in this module implement the manuscript's constant-momentum
limit.  They do not silently substitute for the p(X) energy-loss-aware
calibration required when momentum loss is appreciable.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

import moliere as ml
from config import MATERIALS, P_CACHE_STEP, THETA_CUT, X_CACHE_STEP
from kinematics import theta_space_highland

# Compatibility name only.  The corrected implementation has no universal
# optimum; use optimal_k()/optimal_cut() for the chosen momentum and path.
K_OPT = None
_CUT_CACHE_STEP = 0.002  # rad, cache bucket for arbitrary per-event cuts


def _x_over_x0(X_al, X_cu, X_pb):
    return (X_al / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
            + X_cu / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"]
            + X_pb / MATERIALS["Pb"]["rho"] / MATERIALS["Pb"]["X0"])


def _theta_rms_radial(chi_c2, B, cut=THETA_CUT):
    """Conditional RMS from the manuscript's radial P_M(Theta)."""
    _, M2, _ = ml.radial_moments(chi_c2, B, float(cut), nmax=2)
    return math.sqrt(max(M2, 0.0))


def _theta_rms_disc(chi_c2, chi_a2, B, cut=THETA_CUT, n=None):
    """Backward-compatible alias for old plotting code.

    Despite the historical name, this now performs the correct radial
    one-dimensional moment integral.  chi_a2 and n are unused because B and
    chi_c2 completely specify the n<=2 radial series at fixed path/momentum.
    """
    del chi_a2, n
    return _theta_rms_radial(chi_c2, B, cut=cut)


@lru_cache(maxsize=16384)
def _theta_rms_bucketed(p_key, X_al_key, X_cu_key, X_pb_key, cut_key):
    if X_al_key == 0 and X_cu_key == 0 and X_pb_key == 0:
        return 0.0
    p = p_key * P_CACHE_STEP
    X_al = X_al_key * X_CACHE_STEP
    X_cu = X_cu_key * X_CACHE_STEP
    X_pb = X_pb_key * X_CACHE_STEP
    cut = cut_key * _CUT_CACHE_STEP
    chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, X_pb, p)
    B = ml.solve_B(chi_c2, chi_a2)
    return _theta_rms_radial(chi_c2, B, cut=cut)


def _broadcast_inputs(*xs):
    arrs = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs])
    shape = arrs[0].shape
    return [a.ravel() for a in arrs], shape


def _compat_shape(values, shape):
    """Preserve the repository's historical scalar->length-1-array API."""
    values = np.asarray(values, dtype=float)
    return values.reshape(1) if shape == () else values.reshape(shape)


def theta_RMS_at_cut(p, X_al, X_cu, X_pb, theta_cut):
    """Radial acceptance-truncated RMS at arbitrary cut(s).

    Inputs broadcast to a common shape.  A NumPy array of that shape is
    returned (a scalar input therefore returns a 0-D array).
    """
    (flat, shape) = _broadcast_inputs(p, X_al, X_cu, X_pb, theta_cut)
    p_f, al_f, cu_f, pb_f, cut_f = flat
    keys = np.column_stack([
        np.rint(p_f / P_CACHE_STEP).astype(np.int64),
        np.rint(al_f / X_CACHE_STEP).astype(np.int64),
        np.rint(cu_f / X_CACHE_STEP).astype(np.int64),
        np.rint(pb_f / X_CACHE_STEP).astype(np.int64),
        np.rint(cut_f / _CUT_CACHE_STEP).astype(np.int64),
    ])
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    vals = np.empty(unique.shape[0], dtype=float)
    for i, k in enumerate(unique):
        vals[i] = _theta_rms_bucketed(*(int(v) for v in k))
    return _compat_shape(vals[inv], shape)


def theta_RMS(p, X_al, X_cu, X_pb):
    """Radial RMS for the repository's fixed THETA_CUT acceptance."""
    return theta_RMS_at_cut(p, X_al, X_cu, X_pb, THETA_CUT)


def eps_M(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    """Fractional radial RMS/Highland mismatch for the same finite acceptance."""
    (flat, shape) = _broadcast_inputs(p, X_al, X_cu, X_pb, theta_cut)
    p_f, al_f, cu_f, pb_f, cut_f = flat
    trms = theta_RMS_at_cut(p_f, al_f, cu_f, pb_f, cut_f).ravel()
    xX0 = _x_over_x0(al_f, cu_f, pb_f)
    tspace = np.asarray(theta_space_highland(p_f, xX0), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(tspace > 0.0, trms / tspace - 1.0, 0.0)
    return _compat_shape(out, shape)


def theta_space_corrected(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    """Alias for the acceptance-matched radial RMS, not a fitted correction."""
    return theta_RMS_at_cut(p, X_al, X_cu, X_pb, theta_cut)


def efficiency(p, X_al, X_cu, X_pb, k):
    """Manuscript efficiency eta=sqrt(F_c)*M2/sigma(theta^2).

    k is defined by theta_cut=k*theta0, where theta0 is the projected Highland
    RMS.  The result is model/path/momentum specific; no universality is
    assumed.
    """
    chi_c2, chi_a2 = ml.combine_path(float(X_al), float(X_cu), float(X_pb),
                                     float(p))
    if chi_c2 <= 0.0:
        return 0.0
    B = ml.solve_B(chi_c2, chi_a2)
    xX0 = float(_x_over_x0(X_al, X_cu, X_pb))
    theta0 = float(theta_space_highland(p, xX0)) / math.sqrt(2.0)
    cut = float(k) * theta0
    Fc, M2, M4 = ml.radial_moments(chi_c2, B, cut, nmax=2)
    var = max(M4 - M2 * M2, 0.0)
    if Fc <= 0.0 or var <= 0.0:
        return 0.0
    return math.sqrt(Fc) * M2 / math.sqrt(var)


@lru_cache(maxsize=8192)
def _optimal_k_bucketed(p_key, X_al_key, X_cu_key, X_pb_key,
                        k_lo_milli=500, k_hi_milli=8000):
    if X_al_key == 0 and X_cu_key == 0 and X_pb_key == 0:
        return 0.0
    p = p_key * P_CACHE_STEP
    X_al = X_al_key * X_CACHE_STEP
    X_cu = X_cu_key * X_CACHE_STEP
    X_pb = X_pb_key * X_CACHE_STEP
    lo = k_lo_milli / 1000.0
    hi = k_hi_milli / 1000.0

    res = minimize_scalar(
        lambda kval: -efficiency(p, X_al, X_cu, X_pb, kval),
        bounds=(lo, hi), method="bounded",
        options={"xatol": 2e-4},
    )
    if not res.success:
        raise RuntimeError(f"acceptance optimization failed: {res.message}")
    return float(res.x)


def optimal_k(p, X_al, X_cu, X_pb, bounds=(0.5, 8.0)):
    """Numerically maximize eta for each supplied momentum/path hypothesis."""
    lo, hi = map(float, bounds)
    if not (0.0 < lo < hi):
        raise ValueError("bounds must satisfy 0 < low < high")
    (flat, shape) = _broadcast_inputs(p, X_al, X_cu, X_pb)
    p_f, al_f, cu_f, pb_f = flat
    keys = np.column_stack([
        np.rint(p_f / P_CACHE_STEP).astype(np.int64),
        np.rint(al_f / X_CACHE_STEP).astype(np.int64),
        np.rint(cu_f / X_CACHE_STEP).astype(np.int64),
        np.rint(pb_f / X_CACHE_STEP).astype(np.int64),
    ])
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    vals = np.empty(unique.shape[0], dtype=float)
    lo_m = int(round(lo * 1000))
    hi_m = int(round(hi * 1000))
    for i, k in enumerate(unique):
        vals[i] = _optimal_k_bucketed(*(int(v) for v in k), lo_m, hi_m)
    return _compat_shape(vals[inv], shape)


def optimal_cut(p, X_al, X_cu, X_pb, k_opt=None, bounds=(0.5, 8.0)):
    """Return theta_cut for a chosen model/path.

    If k_opt is None, eta is maximized separately for the supplied
    momentum/path bucket(s).  Passing an explicit k_opt remains available for
    controlled comparisons, but it is not treated as a manuscript constant.
    """
    (flat, shape) = _broadcast_inputs(p, X_al, X_cu, X_pb)
    p_f, al_f, cu_f, pb_f = flat
    xX0 = _x_over_x0(al_f, cu_f, pb_f)
    theta0 = np.asarray(theta_space_highland(p_f, xX0), dtype=float) / math.sqrt(2.0)
    if k_opt is None:
        k = optimal_k(p_f, al_f, cu_f, pb_f, bounds=bounds).ravel()
    else:
        k = np.broadcast_to(np.asarray(k_opt, dtype=float), shape).ravel()
    return _compat_shape(k * theta0, shape)


def eps_M_marginal(p):
    """Momentum-only axial-reference diagnostic (Al 10 cm + Cu 15 cm).

    This is retained only as a diagnostic reduction of the full path-dependent
    eps_M.  It is not the defining calibration for general event paths.
    """
    p_arr = np.asarray(p, dtype=float)
    X_al = MATERIALS["Al"]["rho"] * 10.0
    X_cu = MATERIALS["Cu"]["rho"] * 15.0
    return eps_M(p_arr, X_al, X_cu, 0.0)


def verify_radial():
    """Print radial calibration and model-specific eta optimum for axial path."""
    X_al = MATERIALS["Al"]["rho"] * 10.0
    X_cu = MATERIALS["Cu"]["rho"] * 15.0
    print(f"{'p':>5} {'eps_M/%':>10} {'k_opt':>10} {'eta_max':>10}")
    for p in (1.0, 2.0, 3.5, 6.0):
        e = float(eps_M(p, X_al, X_cu, 0.0)[0]) * 100.0
        k = float(optimal_k(p, X_al, X_cu, 0.0)[0])
        eta = efficiency(p, X_al, X_cu, 0.0, k)
        print(f"{p:5.1f} {e:10.3f} {k:10.4f} {eta:10.4f}")


if __name__ == "__main__":
    verify_radial()