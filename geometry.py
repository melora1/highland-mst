"""Vectorised ray geometry for the nested Al/Cu/Pb target.

The key output is an ordered five-segment representation
[Al_up, Cu_up, Pb, Cu_down, Al_down].  Energy loss depends on order, so the
revised code never reconstructs p(X) from unordered material totals.
"""
from __future__ import annotations

import numpy as np

from config import AL_HALF, CU_HALF, MATERIALS, PB_CX, PB_CY, PB_HALF_Z, PB_R


def _slab_interval(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    a = np.minimum(t1, t2)
    b = np.maximum(t1, t2)
    par = np.abs(d) < 1e-15
    inside = (o >= lo) & (o <= hi)
    a = np.where(par, np.where(inside, -np.inf, np.inf), a)
    b = np.where(par, np.where(inside, np.inf, -np.inf), b)
    return a, b


def box_interval(o, u, half):
    o = np.asarray(o, float)
    u = np.asarray(u, float)
    lo = np.full(o.shape[0], -np.inf)
    hi = np.full(o.shape[0], np.inf)
    for j in range(3):
        a, b = _slab_interval(o[:, j], u[:, j], -half, half)
        lo = np.maximum(lo, a)
        hi = np.minimum(hi, b)
    valid = hi > lo
    return np.where(valid, lo, np.nan), np.where(valid, hi, np.nan)


def cylinder_interval(o, u, r=PB_R, cx=PB_CX, cy=PB_CY, half_z=PB_HALF_Z):
    o = np.asarray(o, float)
    u = np.asarray(u, float)
    ox = o[:, 0] - cx
    oy = o[:, 1] - cy
    ux, uy = u[:, 0], u[:, 1]
    a = ux * ux + uy * uy
    b = 2.0 * (ox * ux + oy * uy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4.0 * a * c

    rlo = np.full(o.shape[0], np.nan)
    rhi = np.full(o.shape[0], np.nan)
    nonax = (a > 1e-15) & (disc >= 0.0)
    sd = np.sqrt(np.maximum(disc, 0.0))
    rlo[nonax] = (-b[nonax] - sd[nonax]) / (2.0 * a[nonax])
    rhi[nonax] = (-b[nonax] + sd[nonax]) / (2.0 * a[nonax])

    axial = a <= 1e-15
    inside_radial = c <= 0.0
    rlo[axial & inside_radial] = -np.inf
    rhi[axial & inside_radial] = np.inf

    zlo, zhi = _slab_interval(o[:, 2], u[:, 2], -half_z, half_z)
    lo = np.maximum(rlo, zlo)
    hi = np.minimum(rhi, zhi)
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    return np.where(valid, lo, np.nan), np.where(valid, hi, np.nan)


def _clip_interval(inner_lo, inner_hi, outer_lo, outer_hi):
    lo = np.maximum(inner_lo, outer_lo)
    hi = np.minimum(inner_hi, outer_hi)
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    return np.where(valid, lo, np.nan), np.where(valid, hi, np.nan)


def trace_paths(o, u, reference=False):
    """Trace rays and return exact ordered path segments in cm.

    Parameters
    ----------
    o, u : (N,3) arrays
        Ray point and unit direction.
    reference : bool
        If True, Pb is replaced by Cu as in the normalization geometry.

    Returns
    -------
    dict containing `segments` (N,5), total material lengths, entry/exit
    parameters, and target midpoint coordinates.
    """
    o = np.asarray(o, float)
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    oa, ob = box_interval(o, u, AL_HALF)
    ca, cb = box_interval(o, u, CU_HALF)
    ca, cb = _clip_interval(ca, cb, oa, ob)
    pa, pb = cylinder_interval(o, u)
    pa, pb = _clip_interval(pa, pb, ca, cb)

    n = o.shape[0]
    seg = np.zeros((n, 5), float)  # Al_up, Cu_up, Pb, Cu_down, Al_down
    outer = np.isfinite(oa) & np.isfinite(ob)
    in_cu = np.isfinite(ca) & np.isfinite(cb)
    in_pb = (np.isfinite(pa) & np.isfinite(pb)) if not reference else np.zeros(n, dtype=bool)

    # Rays missing the Cu block are pure Al.
    seg[outer & ~in_cu, 0] = ob[outer & ~in_cu] - oa[outer & ~in_cu]

    m = outer & in_cu
    seg[m, 0] = ca[m] - oa[m]
    seg[m, 4] = ob[m] - cb[m]

    no_pb = m & ~in_pb
    seg[no_pb, 1] = cb[no_pb] - ca[no_pb]

    yes_pb = m & in_pb
    seg[yes_pb, 1] = pa[yes_pb] - ca[yes_pb]
    seg[yes_pb, 2] = pb[yes_pb] - pa[yes_pb]
    seg[yes_pb, 3] = cb[yes_pb] - pb[yes_pb]

    # For the reference geometry, Pb is copper, hence a continuous Cu interval.
    if reference:
        seg[m, 1] = cb[m] - ca[m]
        seg[m, 2:4] = 0.0

    t_al = seg[:, 0] + seg[:, 4]
    t_cu = seg[:, 1] + seg[:, 3]
    t_pb = seg[:, 2]
    s_mid = np.where(outer, 0.5 * (oa + ob), 0.0)
    midpoint = o + s_mid[:, None] * u
    return dict(segments=seg, t_Al=t_al, t_Cu=t_cu, t_Pb=t_pb,
                s_entry=oa, s_exit=ob, midpoint=midpoint)


def x_over_x0(t_al, t_cu, t_pb):
    return (np.asarray(t_al) / MATERIALS["Al"].X0 +
            np.asarray(t_cu) / MATERIALS["Cu"].X0 +
            np.asarray(t_pb) / MATERIALS["Pb"].X0)


def areal_densities(t_al, t_cu, t_pb):
    return (np.asarray(t_al) * MATERIALS["Al"].rho,
            np.asarray(t_cu) * MATERIALS["Cu"].rho,
            np.asarray(t_pb) * MATERIALS["Pb"].rho)


def truth_classes(trace):
    """Boolean physical path classes used by the revision diagnostics."""
    return {
        "pb": trace["t_Pb"] > 1e-8,
        "cu_only": (trace["t_Cu"] > 1e-8) & (trace["t_Pb"] <= 1e-8),
        "al_only": (trace["t_Cu"] <= 1e-8) & (trace["t_Pb"] <= 1e-8) & (trace["t_Al"] > 1e-8),
    }
