"""Analytic ray tracer: Al shell (25^3) > Cu block (15^3) > Pb cylinder.

Two geometries are exposed and MUST NOT be confused (see plan Sec. 8):

  trace_true : Al + Cu + Pb            -> Branch A prediction, Moliere sampling
  trace_ref  : Al + Cu only (Pb->Cu)   -> weight denominator, Eq. (2)

Using trace_true in the weight denominator normalises away the imaging
contrast; using trace_ref in Branch A corrupts the Highland test.

All rays are parameterised r(s) = r0 + s*u with u a unit vector, s in cm.
Returns path lengths t_Al, t_Cu, t_Pb in cm (fully vectorised over events).
"""

import numpy as np

from config import (AL_HALF, CU_HALF, MATERIALS, MAT_ORDER, PB_CX, PB_CY,
                    PB_HALF_Z, PB_R)


def _slab_interval(o, d, lo, hi):
    """1-D slab intersection. Returns (t_enter, t_exit) arrays; may be inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tlo = np.minimum(t1, t2)
    thi = np.maximum(t1, t2)
    # ray parallel to slab: inside -> (-inf, +inf), outside -> empty
    par = d == 0.0
    inside = (o >= lo) & (o <= hi)
    tlo = np.where(par, np.where(inside, -np.inf, np.inf), tlo)
    thi = np.where(par, np.where(inside, np.inf, -np.inf), thi)
    return tlo, thi


def _box_path(o, u, half):
    """Path length (cm) inside an axis-aligned cube of half-size `half`."""
    tlo = np.full(o.shape[0], -np.inf)
    thi = np.full(o.shape[0], np.inf)
    for k in range(3):
        a, b = _slab_interval(o[:, k], u[:, k], -half, +half)
        tlo = np.maximum(tlo, a)
        thi = np.minimum(thi, b)
    return np.maximum(thi - tlo, 0.0)


def _cyl_path(o, u, r, cx, cy, half_z):
    """Path length (cm) inside a z-axis cylinder, radius r, centre (cx,cy),
    |z| <= half_z."""
    ox = o[:, 0] - cx
    oy = o[:, 1] - cy
    ux, uy = u[:, 0], u[:, 1]

    a = ux * ux + uy * uy
    b = 2.0 * (ox * ux + oy * uy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4.0 * a * c

    tlo = np.full(o.shape[0], -np.inf)
    thi = np.full(o.shape[0], np.inf)

    # near-axial rays (a ~ 0): radially inside -> infinite interval
    axial = a < 1e-18
    inside_r = c <= 0.0
    tlo = np.where(axial, np.where(inside_r, -np.inf, np.inf), tlo)
    thi = np.where(axial, np.where(inside_r, np.inf, -np.inf), thi)

    ok = (~axial) & (disc > 0.0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = (-b - sq) / (2.0 * a)
        s2 = (-b + sq) / (2.0 * a)
    tlo = np.where(ok, s1, tlo)
    thi = np.where(ok, s2, thi)
    miss = (~axial) & (disc <= 0.0)
    tlo = np.where(miss, np.inf, tlo)
    thi = np.where(miss, -np.inf, thi)

    zlo, zhi = _slab_interval(o[:, 2], u[:, 2], -half_z, +half_z)
    tlo = np.maximum(tlo, zlo)
    thi = np.minimum(thi, zhi)
    return np.maximum(thi - tlo, 0.0)


def _paths(o, u, with_pb):
    o = np.atleast_2d(np.asarray(o, float))
    u = np.atleast_2d(np.asarray(u, float))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)

    l_al_box = _box_path(o, u, AL_HALF)
    l_cu_box = _box_path(o, u, CU_HALF)
    l_pb = _cyl_path(o, u, PB_R, PB_CX, PB_CY, PB_HALF_Z) if with_pb \
        else np.zeros(o.shape[0])

    t_al = np.maximum(l_al_box - l_cu_box, 0.0)
    t_cu = np.maximum(l_cu_box - l_pb, 0.0)
    t_pb = l_pb
    return t_al, t_cu, t_pb


def trace_true(o, u):
    """Al + Cu + Pb. Returns (t_Al, t_Cu, t_Pb) in cm."""
    return _paths(o, u, with_pb=True)


def trace_ref(o, u):
    """Reference geometry: Al + Cu only, Pb volume filled with Cu."""
    return _paths(o, u, with_pb=False)


def x_over_X0(t_al, t_cu, t_pb):
    """Dimensionless radiation-length path."""
    return (t_al / MATERIALS["Al"]["X0"]
            + t_cu / MATERIALS["Cu"]["X0"]
            + t_pb / MATERIALS["Pb"]["X0"])


def areal_densities(t_al, t_cu, t_pb):
    """(X_Al, X_Cu, X_Pb) in g cm^-2, ordered as MAT_ORDER."""
    ts = (t_al, t_cu, t_pb)
    return tuple(MATERIALS[m]["rho"] * t for m, t in zip(MAT_ORDER, ts))
