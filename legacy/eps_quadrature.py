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
calibration required when momentum loss is appreciable (see energy_loss.py
and eps_quadrature_pofx.py).

--------------------------------------------------------------------------
TWO CALIBRATION PATHS -- EXACT vs BUCKETED.  READ BEFORE ADDING A CALL SITE.
--------------------------------------------------------------------------
The Moliere quadrature is too expensive to evaluate per event, so the
*_bucketed helpers quantize (p, X_Al, X_Cu, X_Pb, cut) onto the
P_CACHE_STEP / X_CACHE_STEP / _CUT_CACHE_STEP grids and cache the result.
The Highland denominator is closed form and is NOT cached, so it is
evaluated at the unrounded arguments.  Rounding one side of a ratio and not
the other leaves an offset.

For PER-EVENT weights that offset is a wash: X_Al_ref and X_Cu_ref vary
continuously across the raster, so the rounding is unbiased event to event,
and the cache is what makes a 2e6-event run tractable.  The bucketed
functions -- theta_RMS, theta_RMS_at_cut, eps_M -- are for that case.

For a FIXED reference path it is a systematic, because the same areal
density is rounded the same way every time.  The axial reference path has
X_Cu = 8.96 * 15 = 134.4 g/cm^2, which rounds UP to 134.5 on the 0.25 g/cm^2
grid -- 0.28 mm of extra copper in the numerator only.  Measured offset on
that path, at 200 mrad:

    p (GeV/c)      1.0      2.0      3.5      6.0
    exact       4.0350   9.7342  13.1075  16.1104   %
    bucketed    4.0677   9.7710  13.1456  16.1496   %
    offset      +0.033   +0.037   +0.038   +0.039   pp

Small in absolute terms, but ~27% of the 6 GeV/c energy-loss shift it would
be differenced against.  Table I of the manuscript was produced with the
bucketed values and must be reissued from the exact ones.

    Fixed path, one or few evaluations   ->  eps_M_exact / theta_RMS_exact
    Fixed path, one evaluation per event ->  eps_M_marginal (exact interpolant)
    Per-event varying path               ->  eps_M / theta_RMS (bucketed)

efficiency() and optimal_k() are unaffected: efficiency() calls
combine_path on the raw X, and optimal_k's bucketing shifts the cut and the
width together.  test_pofx.test_optimal_k_insensitive_to_bucketing verifies
that rather than assuming it.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

from . import moliere as ml
from config import MATERIALS, P_CACHE_STEP, THETA_CUT, X_CACHE_STEP
from .kinematics import theta_space_highland

# Compatibility name only.  The corrected implementation has no universal
# optimum; use optimal_k()/optimal_cut() for the chosen momentum and path.
K_OPT = None
_CUT_CACHE_STEP = 0.002  # rad, cache bucket for arbitrary per-event cuts

# Axial reference path used by eps_M_marginal (manuscript Sec. IV A).
_AXIAL_T_AL = 10.0  # cm
_AXIAL_T_CU = 15.0  # cm


def _x_over_x0(X_al, X_cu, X_pb):
    return (
        X_al / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
        + X_cu / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"]
        + X_pb / MATERIALS["Pb"]["rho"] / MATERIALS["Pb"]["X0"]
    )


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


# ==========================================================================
# EXACT (unbucketed) calibration -- for fixed reference paths
# ==========================================================================


def theta_RMS_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT, nmax=2):
    """Unbucketed radial acceptance-truncated RMS [rad].  Scalar, uncached.

    Use for tables, figures, and any quantity evaluated on a FIXED reference
    path.  See the module docstring for why theta_RMS() is not appropriate
    there.
    """
    chi_c2, chi_a2 = ml.combine_path(float(X_al), float(X_cu), float(X_pb), float(p))
    if chi_c2 <= 0.0:
        return 0.0
    B = ml.solve_B(chi_c2, chi_a2)
    _, M2, _ = ml.radial_moments(chi_c2, B, float(theta_cut), nmax=nmax)
    return math.sqrt(max(M2, 0.0))


def eps_M_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT, nmax=2):
    """Unbucketed fractional radial-RMS / Highland mismatch.  Scalar.

    This is the Table I quantity.
    """
    xX0 = _x_over_x0(float(X_al), float(X_cu), float(X_pb))
    tspace = float(theta_space_highland(float(p), xX0))
    if not (tspace > 0.0):
        return 0.0
    return theta_RMS_exact(p, X_al, X_cu, X_pb, theta_cut, nmax) / tspace - 1.0


def Fc_M4_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT, nmax=2):
    """Unbucketed (F_c, M_2, M_4) for a fixed path, Eqs. (12)-(14)."""
    chi_c2, chi_a2 = ml.combine_path(float(X_al), float(X_cu), float(X_pb), float(p))
    if chi_c2 <= 0.0:
        return 0.0, 0.0, 0.0
    B = ml.solve_B(chi_c2, chi_a2)
    return ml.radial_moments(chi_c2, B, float(theta_cut), nmax=nmax)


# ==========================================================================
# BUCKETED calibration -- for per-event, varying-path use
# ==========================================================================


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
    """Radial acceptance-truncated RMS at arbitrary cut(s).  BUCKETED.

    Inputs broadcast to a common shape.  A NumPy array of that shape is
    returned (a scalar input therefore returns a 0-D array).

    For a fixed reference path use theta_RMS_exact; see the module docstring.
    """
    (flat, shape) = _broadcast_inputs(p, X_al, X_cu, X_pb, theta_cut)
    p_f, al_f, cu_f, pb_f, cut_f = flat
    keys = np.column_stack(
        [
            np.rint(p_f / P_CACHE_STEP).astype(np.int64),
            np.rint(al_f / X_CACHE_STEP).astype(np.int64),
            np.rint(cu_f / X_CACHE_STEP).astype(np.int64),
            np.rint(pb_f / X_CACHE_STEP).astype(np.int64),
            np.rint(cut_f / _CUT_CACHE_STEP).astype(np.int64),
        ]
    )
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    vals = np.empty(unique.shape[0], dtype=float)
    for i, k in enumerate(unique):
        vals[i] = _theta_rms_bucketed(*(int(v) for v in k))
    return _compat_shape(vals[inv], shape)


def theta_RMS(p, X_al, X_cu, X_pb):
    """Radial RMS for the repository's fixed THETA_CUT acceptance.  BUCKETED."""
    return theta_RMS_at_cut(p, X_al, X_cu, X_pb, THETA_CUT)


def eps_M(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    """Fractional radial RMS/Highland mismatch for the same finite acceptance.

    BUCKETED numerator, unbucketed denominator -- intended for per-event
    calls with a varying path.  For a fixed reference path use eps_M_exact.
    """
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


# ==========================================================================
# Acceptance efficiency and optimum (already unbucketed in X)
# ==========================================================================


def efficiency(p, X_al, X_cu, X_pb, k):
    """Manuscript efficiency eta=sqrt(F_c)*M2/sigma(theta^2).

    k is defined by theta_cut=k*theta0, where theta0 is the projected Highland
    RMS.  The result is model/path/momentum specific; no universality is
    assumed.  chi_c2/chi_a2 come from combine_path on the raw X, so this
    function carries no areal-density quantization.
    """
    chi_c2, chi_a2 = ml.combine_path(float(X_al), float(X_cu), float(X_pb), float(p))
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
def _optimal_k_bucketed(
    p_key, X_al_key, X_cu_key, X_pb_key, k_lo_milli=500, k_hi_milli=8000
):
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
        bounds=(lo, hi),
        method="bounded",
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
    keys = np.column_stack(
        [
            np.rint(p_f / P_CACHE_STEP).astype(np.int64),
            np.rint(al_f / X_CACHE_STEP).astype(np.int64),
            np.rint(cu_f / X_CACHE_STEP).astype(np.int64),
            np.rint(pb_f / X_CACHE_STEP).astype(np.int64),
        ]
    )
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    vals = np.empty(unique.shape[0], dtype=float)
    lo_m = int(round(lo * 1000))
    hi_m = int(round(hi * 1000))
    for i, k in enumerate(unique):
        vals[i] = _optimal_k_bucketed(*(int(v) for v in k), lo_m, hi_m)
    return _compat_shape(vals[inv], shape)


def optimal_k_exact(p, X_al, X_cu, X_pb, bounds=(0.5, 8.0)):
    """Unbucketed optimal k for a fixed path.  Scalar, uncached."""
    lo, hi = map(float, bounds)
    res = minimize_scalar(
        lambda kval: -efficiency(p, X_al, X_cu, X_pb, kval),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 2e-4},
    )
    if not res.success:
        raise RuntimeError(f"acceptance optimization failed: {res.message}")
    return float(res.x)


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


# ==========================================================================
# Fixed-path, per-event: exact interpolant in momentum
# ==========================================================================
#
# eps_M_marginal evaluates ONE path (the axial reference) at up to 2e6
# momenta.  Bucketing X there is a systematic (fixed path), and bucketing p
# at P_CACHE_STEP = 0.010 GeV/c costs up to +/-0.028 pp near 1 GeV/c, where
# d eps_M / dp ~ 5.7 pp per GeV/c.  Both are avoided by tabulating the exact
# eps_M on a dense grid in ln p and interpolating.
#
# Grid resolution is set so linear interpolation error is <1e-4 pp; see
# _marginal_interp_error() and test_pofx.test_marginal_interpolant_accurate.

_MARGINAL_P_LO = 0.15  # GeV/c
_MARGINAL_P_HI = 200.0  # GeV/c
_MARGINAL_N = 1500
_MARGINAL_CACHE = {}


def _marginal_grid(theta_cut):
    """(ln p grid, eps_M grid) for the axial reference path.  Built once."""
    key = round(float(theta_cut) / 1e-9)
    if key in _MARGINAL_CACHE:
        return _MARGINAL_CACHE[key]
    X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
    X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
    lnp = np.linspace(math.log(_MARGINAL_P_LO), math.log(_MARGINAL_P_HI), _MARGINAL_N)
    eps = np.array([eps_M_exact(math.exp(v), X_al, X_cu, 0.0, theta_cut) for v in lnp])
    _MARGINAL_CACHE[key] = (lnp, eps)
    return lnp, eps


def eps_M_marginal(p, theta_cut=THETA_CUT):
    """Momentum-only axial-reference diagnostic (Al 10 cm + Cu 15 cm).

    EXACT in the areal density and in momentum: evaluated from a dense
    tabulation of eps_M_exact and interpolated in ln p.  Momenta outside
    [_MARGINAL_P_LO, _MARGINAL_P_HI] fall back to a direct exact evaluation.

    This is retained only as a diagnostic reduction of the full path-dependent
    eps_M.  It is not the defining calibration for general event paths, and it
    is the CONSTANT-MOMENTUM value; see eps_quadrature_pofx.eps_M_marginal_pofx
    when energy loss is appreciable.
    """
    p_arr = np.asarray(p, dtype=float)
    shape = p_arr.shape
    flat = np.atleast_1d(p_arr).ravel()
    out = np.zeros(flat.size, dtype=float)

    good = np.isfinite(flat) & (flat > 0.0)
    inside = good & (flat >= _MARGINAL_P_LO) & (flat <= _MARGINAL_P_HI)
    lnp, eps = _marginal_grid(theta_cut)
    out[inside] = np.interp(np.log(flat[inside]), lnp, eps)

    outside = good & ~inside
    if outside.any():
        X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
        X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
        for i in np.flatnonzero(outside):
            try:
                out[i] = eps_M_exact(float(flat[i]), X_al, X_cu, 0.0, theta_cut)
            except (ValueError, RuntimeError):
                out[i] = 0.0
    return _compat_shape(out, shape)


def _marginal_interp_error(theta_cut=THETA_CUT, n_probe=400):
    """Max |interpolated - exact| over random momenta, in percentage points.

    Diagnostic for the grid resolution above; not used in production.
    """
    X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
    X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
    rng = np.random.default_rng(0)
    ps = np.exp(rng.uniform(math.log(0.3), math.log(20.0), n_probe))
    got = eps_M_marginal(ps, theta_cut)
    want = np.array([eps_M_exact(float(v), X_al, X_cu, 0.0, theta_cut) for v in ps])
    return float(np.max(np.abs(got - want)) * 100.0)


# ==========================================================================


def verify_radial():
    """Axial reference-path calibration.  The EXACT column is Table I."""
    X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
    X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
    print(
        f"{'p':>5} {'eps_M/% (exact)':>16} {'eps_M/% (bucketed)':>19} "
        f"{'offset/pp':>10} {'k_opt':>9} {'eta_max':>9}"
    )
    for p in (1.0, 2.0, 3.5, 6.0):
        e_ex = eps_M_exact(p, X_al, X_cu, 0.0) * 100.0
        e_bk = float(eps_M(p, X_al, X_cu, 0.0)[0]) * 100.0
        k = optimal_k_exact(p, X_al, X_cu, 0.0)
        print(
            f"{p:5.1f} {e_ex:16.3f} {e_bk:19.3f} {e_bk - e_ex:+10.4f} "
            f"{k:9.4f} {efficiency(p, X_al, X_cu, 0.0, k):9.4f}"
        )
    print()
    print("EXACT is the manuscript quantity.  The bucketed column carries the")
    print("X_CACHE_STEP quantization of the fixed reference path (X_Cu 134.4 -> 134.5")
    print("g/cm^2) and is correct only for per-event, varying-path use.")
    print(
        f"eps_M_marginal interpolation error: "
        f"{_marginal_interp_error():.2e} pp (max over 0.3-20 GeV/c)"
    )


if __name__ == "__main__":
    verify_radial()
