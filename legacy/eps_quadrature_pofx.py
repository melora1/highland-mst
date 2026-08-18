"""p(X)-aware drop-in replacements for eps_quadrature's calibration API.

Signatures and bucketing mirror eps_quadrature.py exactly (same
P_CACHE_STEP / X_CACHE_STEP, same broadcast/compat-shape behaviour), so
results_pipeline.py can switch calibrations by changing only its imports.

--------------------------------------------------------------------------
THREE QUANTITIES, NOT INTERCHANGEABLE
--------------------------------------------------------------------------
theta_RMS_pofx  sqrt(M_2) from the varying-p accumulated (chi_c^2, B).  The
                acceptance-matched denominator for w_Q once energy loss is
                propagated.

eps_M_pofx      theta_RMS[p(X)] / theta_space[p(X)] - 1.  BOTH sides on the
                same p(X) profile, as Sec. II F requires.  Replaces Table I.

eps_M_mixed     theta_RMS[p(X)] / theta_space_highland(p_in) - 1.  What
                Eqs. (18)-(19) actually produce in the detector, because
                p_meas is the UPSTREAM tagged momentum while the muon
                scatters on the degraded profile.

Reporting one where another is meant is the main way to get Step 1 wrong;
the names are deliberately unshareable.

--------------------------------------------------------------------------
EXACT vs BUCKETED, AND A STRUCTURAL ASYMMETRY WORTH KNOWING
--------------------------------------------------------------------------
As in eps_quadrature, the bucketed entry points quantize X to X_CACHE_STEP
for caching.  The two epsilons respond differently, and the difference is
not a bug in either:

  eps_M_pofx  takes numerator AND denominator from the same calibrate() call
              at the same bucketed X, so the sqrt(X) scaling largely cancels
              and only the truncation residual survives.  Measured on the
              axial reference path: about -0.009 pp.

  eps_M_mixed divides a bucketed numerator by theta_space_highland at the
              raw x/X0.  One-sided, exactly like eps_quadrature.eps_M, so it
              carries the same offset: about +0.033 pp on that path.

Neither matters per event with a varying path, where the rounding is
unbiased.  Both matter on a FIXED path, where the same X is rounded the same
way every time.  Use the *_exact functions, or eps_M_marginal_pofx, there.
"""

import math
from functools import lru_cache

import numpy as np

from . import energy_loss as el
from config import MATERIALS, P_CACHE_STEP, THETA_CUT, X_CACHE_STEP
from .kinematics import theta_space_highland

_CUT_CACHE_STEP = 0.002  # rad, matches eps_quadrature
_TOL = 0.01  # Step 1.1 slicing criterion

# Axial reference path (manuscript Sec. IV A), matching eps_quadrature.
_AXIAL_T_AL = 10.0  # cm
_AXIAL_T_CU = 15.0  # cm


def _thicknesses(X_al, X_cu, X_pb):
    return (
        X_al / MATERIALS["Al"]["rho"],
        X_cu / MATERIALS["Cu"]["rho"],
        X_pb / MATERIALS["Pb"]["rho"],
    )


def _x_over_x0(X_al, X_cu, X_pb):
    return (
        X_al / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
        + X_cu / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"]
        + X_pb / MATERIALS["Pb"]["rho"] / MATERIALS["Pb"]["X0"]
    )


def _broadcast_inputs(*xs):
    arrs = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs])
    shape = arrs[0].shape
    return [a.ravel() for a in arrs], shape


def _compat_shape(values, shape):
    values = np.asarray(values, dtype=float)
    return values.reshape(1) if shape == () else values.reshape(shape)


# ==========================================================================
# EXACT (unbucketed) -- for fixed reference paths
# ==========================================================================


def calibrate_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT, tol=_TOL):
    """Unbucketed p(X) calibration for one momentum and one path.

    Thin wrapper on energy_loss.calibrate taking areal densities rather than
    thicknesses, so it is signature-compatible with the rest of this module.
    """
    t_al, t_cu, t_pb = _thicknesses(float(X_al), float(X_cu), float(X_pb))
    return el.calibrate(t_al, t_cu, t_pb, float(p), theta_cut=float(theta_cut), tol=tol)


def theta_RMS_pofx_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    return calibrate_exact(p, X_al, X_cu, X_pb, theta_cut)["th_rms"]


def eps_M_pofx_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    return calibrate_exact(p, X_al, X_cu, X_pb, theta_cut)["eps_M"]


def eps_M_mixed_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    return calibrate_exact(p, X_al, X_cu, X_pb, theta_cut)["eps_mix"]


def p_exit_exact(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    return calibrate_exact(p, X_al, X_cu, X_pb, theta_cut)["p_out"]


# ==========================================================================
# BUCKETED -- for per-event, varying-path use
# ==========================================================================


@lru_cache(maxsize=65536)
def _calib_bucketed(p_key, al_key, cu_key, pb_key, cut_key):
    """(theta_RMS, theta_space_pofx, p_out) for one bucket, or zeros."""
    if al_key == 0 and cu_key == 0 and pb_key == 0:
        return 0.0, 0.0, 0.0
    p = p_key * P_CACHE_STEP
    cut = cut_key * _CUT_CACHE_STEP
    if p <= 0.0 or cut <= 0.0:
        return 0.0, 0.0, 0.0
    try:
        r = calibrate_exact(
            p, al_key * X_CACHE_STEP, cu_key * X_CACHE_STEP, pb_key * X_CACHE_STEP, cut
        )
    except (RuntimeError, ValueError):
        # muon ranges out in this bucket: no exiting event exists to weight
        return 0.0, 0.0, 0.0
    return r["th_rms"], r["th_space"], r["p_out"]


def _eval(p, X_al, X_cu, X_pb, theta_cut, which):
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
    vals = np.empty((unique.shape[0], 3), dtype=float)
    for i, k in enumerate(unique):
        vals[i] = _calib_bucketed(*(int(v) for v in k))
    trms, tspx, pout = vals[inv, 0], vals[inv, 1], vals[inv, 2]

    if which == "rms":
        out = trms
    elif which == "tspace":
        out = tspx
    elif which == "pout":
        out = pout
    elif which == "eps":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(tspx > 0.0, trms / tspx - 1.0, 0.0)
    elif which == "eps_mix":
        ts0 = np.asarray(
            theta_space_highland(p_f, _x_over_x0(al_f, cu_f, pb_f)), dtype=float
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(ts0 > 0.0, trms / ts0 - 1.0, 0.0)
    else:
        raise ValueError(which)
    return _compat_shape(out, shape)


def theta_RMS_at_cut_pofx(p, X_al, X_cu, X_pb, theta_cut):
    return _eval(p, X_al, X_cu, X_pb, theta_cut, "rms")


def theta_RMS_pofx(p, X_al, X_cu, X_pb):
    return _eval(p, X_al, X_cu, X_pb, THETA_CUT, "rms")


def theta_space_pofx(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    return _eval(p, X_al, X_cu, X_pb, theta_cut, "tspace")


def p_exit(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    return _eval(p, X_al, X_cu, X_pb, theta_cut, "pout")


def eps_M_pofx(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    """Self-consistent p(X) mismatch: both sides on the same profile."""
    return _eval(p, X_al, X_cu, X_pb, theta_cut, "eps")


def eps_M_mixed(p, X_al, X_cu, X_pb, theta_cut=THETA_CUT):
    """Deployed-estimator mismatch: numerator p(X), denominator at tagged p."""
    return _eval(p, X_al, X_cu, X_pb, theta_cut, "eps_mix")


# ==========================================================================
# Fixed-path, per-event: exact interpolant in momentum
# ==========================================================================
#
# eps_M_marginal_pofx evaluates ONE path (the axial reference) at up to 2e6
# momenta and feeds I_p.  Bucketing X there is a systematic, and bucketing p
# at P_CACHE_STEP costs ~0.03 pp near 1 GeV/c.  Both are avoided by
# tabulating the exact value on a dense ln p grid, exactly as
# eps_quadrature.eps_M_marginal does for the constant-p case.
#
# Building a table costs ~800 calibrate() calls, once per process per
# (theta_cut, mixed) pair.

_MARGINAL_P_LO = 0.30  # GeV/c; below this a muon ranges out of 161 g/cm^2
_MARGINAL_P_HI = 200.0  # GeV/c
_MARGINAL_N = 800
_MARGINAL_CACHE = {}


def _marginal_grid(theta_cut, mixed):
    key = (round(float(theta_cut) / 1e-9), bool(mixed))
    if key in _MARGINAL_CACHE:
        return _MARGINAL_CACHE[key]
    X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
    X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
    field = "eps_mix" if mixed else "eps_M"
    lnp_grid = np.linspace(
        math.log(_MARGINAL_P_LO), math.log(_MARGINAL_P_HI), _MARGINAL_N
    )
    lnp, vals = [], []
    for v in lnp_grid:
        try:
            vals.append(calibrate_exact(math.exp(v), X_al, X_cu, 0.0, theta_cut)[field])
            lnp.append(v)
        except (RuntimeError, ValueError):
            continue  # muon ranges out; left to the fallback branch
    if not lnp:
        raise RuntimeError("marginal p(X) grid is empty; check the path")
    out = (np.asarray(lnp), np.asarray(vals))
    _MARGINAL_CACHE[key] = out
    return out


def eps_M_marginal_pofx(p, mixed=True, theta_cut=THETA_CUT):
    """Momentum-only axial-reference p(X) correction (Al 10 cm + Cu 15 cm).

    EXACT in areal density and in momentum: a dense tabulation of
    calibrate_exact interpolated in ln p, with direct evaluation outside the
    tabulated range.

    ``mixed=True`` (the default) is mandatory for I_p.  I_p's denominator is
    theta_space_highland(p_meas, x/X0|ref) with p_meas the TAGGED incident
    momentum, so the correct factor is theta_rms[p(X)] over the Highland
    width at p_in -- eps_mix, not the self-consistent eps_M.  ``mixed=False``
    returns the Table I quantity and silently under-corrects the image.
    """
    p_arr = np.asarray(p, dtype=float)
    shape = p_arr.shape
    flat = np.atleast_1d(p_arr).ravel()
    out = np.zeros(flat.size, dtype=float)

    lnp, eps = _marginal_grid(theta_cut, mixed)
    good = np.isfinite(flat) & (flat > 0.0)
    inside = good & (flat >= math.exp(lnp[0])) & (flat <= math.exp(lnp[-1]))
    out[inside] = np.interp(np.log(flat[inside]), lnp, eps)

    outside = good & ~inside
    if outside.any():
        X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
        X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
        field = "eps_mix" if mixed else "eps_M"
        for i in np.flatnonzero(outside):
            try:
                out[i] = calibrate_exact(float(flat[i]), X_al, X_cu, 0.0, theta_cut)[
                    field
                ]
            except (RuntimeError, ValueError):
                out[i] = 0.0  # ranged out: never reaches a detector
    return _compat_shape(out, shape)


def _marginal_interp_error(mixed=True, theta_cut=THETA_CUT, n_probe=120):
    """Max |interpolated - exact| over random momenta, in percentage points."""
    X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
    X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
    rng = np.random.default_rng(0)
    ps = np.exp(rng.uniform(math.log(0.5), math.log(20.0), n_probe))
    got = eps_M_marginal_pofx(ps, mixed=mixed, theta_cut=theta_cut)
    field = "eps_mix" if mixed else "eps_M"
    want = np.array(
        [calibrate_exact(float(v), X_al, X_cu, 0.0, theta_cut)[field] for v in ps]
    )
    return float(np.max(np.abs(got - want)) * 100.0)


# ==========================================================================


def verify_radial_pofx():
    """Axial reference table: constant-p, p(X), and the deployed mismatch."""
    from .eps_quadrature import eps_M_exact

    X_al = MATERIALS["Al"]["rho"] * _AXIAL_T_AL
    X_cu = MATERIALS["Cu"]["rho"] * _AXIAL_T_CU
    print(
        f"{'p':>5} {'Dp/p %':>8} {'const-p %':>11} {'p(X) %':>9} "
        f"{'mixed %':>9} {'E[w] mixed':>11}"
    )
    for p in (1.0, 2.0, 3.5, 6.0):
        r = calibrate_exact(p, X_al, X_cu, 0.0)
        e0 = eps_M_exact(p, X_al, X_cu, 0.0) * 100.0
        print(
            f"{p:5.1f} {100 * r['dp_over_p']:+8.2f} {e0:11.3f} "
            f"{100 * r['eps_M']:9.3f} {100 * r['eps_mix']:9.3f} "
            f"{(1 + r['eps_mix']) ** 2:11.4f}"
        )
    print()
    print("All columns EXACT (no areal-density or momentum quantization).")
    print(
        f"eps_M_marginal_pofx interpolation error: "
        f"{_marginal_interp_error():.2e} pp (max over 0.5-20 GeV/c)"
    )


if __name__ == "__main__":
    verify_radial_pofx()
