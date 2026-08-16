"""Controlled detector simulation for the revision experiments.

Production assumptions are explicit: each nominal momentum setting is re-steered
onto the requested raster node; Moliere scattering is sampled from the ordered
p(X) path; the upstream spectrometer tags incident momentum; and PoCA remains a
single-kink diagnostic rather than a full transport reconstruction.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    BL,
    MOM_BITE,
    MOMENTA,
    OUT_DIR,
    RASTER_HALF,
    RASTER_NX,
    RASTER_NY,
    SEED_BASE,
    SIGMA_DIV,
    SIGMA_HIT,
    SIGMA_XY,
    STATION_Z,
    STEER_COMPENSATION,
    THETA_CUT,
    Z_MAGNET_CM,
)
from geometry import trace_paths, truth_classes, x_over_x0
from physics import PofxCache

SEG_NAMES = ("al_up", "cu_up", "pb", "cu_down", "al_down")


def raster_nodes():
    xs = np.linspace(-RASTER_HALF, RASTER_HALF, RASTER_NX)
    ys = np.linspace(-RASTER_HALF, RASTER_HALF, RASTER_NY)
    return np.array([(x, y) for x in xs for y in ys], float)


def seed_entropy(p_set: float, center_xy=(0.0, 0.0), seed: int = 0):
    """Collision-free input tuple for NumPy SeedSequence on the production grid.

    Coordinates and momentum are quantized only to integer micro-units for the
    entropy vector; the generator itself receives the full vector, avoiding the
    hand-composed scalar-seed collision risk of the legacy implementation.
    """

    def zz(v):
        n = int(round(float(v) * 1_000_000))
        return 2 * n if n >= 0 else -2 * n - 1

    return (
        int(SEED_BASE),
        int(seed),
        int(round(float(p_set) * 1_000_000)),
        zz(center_xy[0]),
        zz(center_xy[1]),
    )


def nominal_target_offset_cm(p_set: float, mode: str | None = None) -> float:
    """Mean dipole-induced target offset before momentum-bite residuals.

    ``per_setting`` retunes the incident position for each nominal setting and
    therefore has zero nominal offset. ``none`` leaves the dipole displacement
    uncompensated.
    """
    mode = STEER_COMPENSATION if mode is None else mode
    if mode == "per_setting":
        return 0.0
    if mode == "none":
        return (0.3 * BL / float(p_set)) * (0.0 - Z_MAGNET_CM)
    raise ValueError("STEER_COMPENSATION must be 'per_setting' or 'none'")


def momentum_fractions(
    x, left=(0.70, 0.20, 0.07, 0.03), right=(0.03, 0.07, 0.20, 0.70)
):
    """Linear momentum-mixture gradient with exact unit normalization."""
    t = np.clip((np.asarray(x, float) + RASTER_HALF) / (2.0 * RASTER_HALF), 0.0, 1.0)
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    f = (1.0 - t[..., None]) * a + t[..., None] * b
    return f / f.sum(axis=-1, keepdims=True)


def _integer_counts(total, fractions):
    raw = total * np.asarray(fractions, float)
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())
    if remainder:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base


def _slope(z1, z2, u1, u2):
    return (u2 - u1) / (z2 - z1)


def _propagate(x, tx, z_from, z_to):
    return x + tx * (z_to - z_from)


def _poca(p1, d1, p2, d2):
    w0 = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = np.einsum("ij,ij->i", d1, w0)
    e = np.einsum("ij,ij->i", d2, w0)
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-14, np.sign(den) * 1e-14 + (den == 0) * 1e-14, den)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((p1 + s[:, None] * d1) + (p2 + t[:, None] * d2))


def _store_segments(data, prefix, seg):
    for j, n in enumerate(SEG_NAMES):
        data[f"{prefix}_{n}"] = seg[:, j]


def segment_matrix(df: pd.DataFrame, prefix: str):
    return df[[f"{prefix}_{n}" for n in SEG_NAMES]].to_numpy(float)


def simulate_fixed_node(
    p_set: float,
    n: int,
    center_xy=(0.0, 0.0),
    seed: int = 0,
    calibrator: PofxCache | None = None,
    reference_target: bool = False,
    steer_compensation: str | None = None,
):
    if n <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(
        np.random.SeedSequence(seed_entropy(p_set, center_xy, seed))
    )
    calibrator = calibrator or PofxCache(nmax=2)

    z1, z2, z3, z4, z5, z6 = STATION_Z

    # x0,y0 are defined at the first station. Shift the incident x so the
    # nominal momentum lands at the requested target raster node after the dipole.
    x0 = center_xy[0] + rng.normal(0.0, SIGMA_XY, n)
    y0 = center_xy[1] + rng.normal(0.0, SIGMA_XY, n)
    tx0 = rng.normal(0.0, SIGMA_DIV, n)
    ty0 = rng.normal(0.0, SIGMA_DIV, n)
    p_true = p_set * (1.0 + MOM_BITE * rng.normal(size=n))
    mode = STEER_COMPENSATION if steer_compensation is None else steer_compensation
    if mode == "per_setting":
        nominal_kick = 0.3 * BL / p_set
        x0 -= nominal_kick * (0.0 - Z_MAGNET_CM)
    elif mode != "none":
        raise ValueError("steer_compensation must be 'per_setting' or 'none'")

    delta_true = 0.3 * BL / p_true
    x_at = lambda z: _propagate(x0, tx0, z1, z)
    y_at = lambda z: _propagate(y0, ty0, z1, z)
    xm, ym = x_at(Z_MAGNET_CM), y_at(Z_MAGNET_CM)
    tx1, ty1 = tx0 + delta_true, ty0

    h1x, h2x = x_at(z1), x_at(z2)
    h1y, h2y = y_at(z1), y_at(z2)
    h3x = _propagate(xm, tx1, Z_MAGNET_CM, z3)
    h4x = _propagate(xm, tx1, Z_MAGNET_CM, z4)
    h3y = _propagate(ym, ty1, Z_MAGNET_CM, z3)
    h4y = _propagate(ym, ty1, Z_MAGNET_CM, z4)

    smear = lambda a: a + rng.normal(0.0, SIGMA_HIT, n)
    m1x, m2x, m3x, m4x = map(smear, (h1x, h2x, h3x, h4x))
    m1y, m2y, m3y, m4y = map(smear, (h1y, h2y, h3y, h4y))

    th_pre = _slope(z1, z2, m1x, m2x)
    th_post = _slope(z3, z4, m3x, m4x)
    delta_meas = th_post - th_pre
    p_meas = 0.3 * BL / np.maximum(np.abs(delta_meas), 1e-12)

    tx_in = th_post
    ty_in = _slope(z3, z4, m3y, m4y)
    x_in = m3x + tx_in * (0.0 - z3)
    y_in = m3y + ty_in * (0.0 - z3)

    o_true = np.stack(
        [
            _propagate(xm, tx1, Z_MAGNET_CM, 0.0),
            _propagate(ym, ty1, Z_MAGNET_CM, 0.0),
            np.zeros(n),
        ],
        axis=1,
    )
    u_true = np.stack([tx1, ty1, np.ones(n)], axis=1)
    u_true /= np.linalg.norm(u_true, axis=1, keepdims=True)

    # The gradient experiment can deliberately remove the Pb inclusion so the
    # only imposed spatial intervention is momentum composition.
    true_trace = trace_paths(o_true, u_true, reference=reference_target)
    ref_true = trace_paths(o_true, u_true, reference=True)

    o_reco = np.stack([x_in, y_in, np.zeros(n)], axis=1)
    u_reco = np.stack([tx_in, ty_in, np.ones(n)], axis=1)
    u_reco /= np.linalg.norm(u_reco, axis=1, keepdims=True)
    ref_reco = trace_paths(o_reco, u_reco, reference=True)

    hit = (true_trace["t_Al"] + true_trace["t_Cu"] + true_trace["t_Pb"]) > 0
    thx = np.zeros(n)
    thy = np.zeros(n)
    if np.any(hit):
        thx[hit], thy[hit] = calibrator.sample(
            p_true[hit], true_trace["segments"][hit], rng
        )
    dth_true = np.hypot(thx, thy)

    # Single-kink PoCA diagnostic: place the equivalent scatter at the exact
    # midpoint of the traversed outer-target interval, not blindly at z=0.
    vtx = true_trace["midpoint"]
    tx_out_true = tx1 + thx
    ty_out_true = ty1 + thy
    h5x = _propagate(vtx[:, 0], tx_out_true, vtx[:, 2], z5)
    h6x = _propagate(vtx[:, 0], tx_out_true, vtx[:, 2], z6)
    h5y = _propagate(vtx[:, 1], ty_out_true, vtx[:, 2], z5)
    h6y = _propagate(vtx[:, 1], ty_out_true, vtx[:, 2], z6)
    m5x, m6x, m5y, m6y = map(smear, (h5x, h6x, h5y, h6y))
    tx_out = _slope(z5, z6, m5x, m6x)
    ty_out = _slope(z5, z6, m5y, m6y)
    dth_reco = np.hypot(tx_out - tx_in, ty_out - ty_in)

    p_up = np.stack(
        [m3x + tx_in * (0.0 - z3), m3y + ty_in * (0.0 - z3), np.zeros(n)], axis=1
    )
    d_up = np.stack([tx_in, ty_in, np.ones(n)], axis=1)
    p_dn = np.stack(
        [m5x + tx_out * (0.0 - z5), m5y + ty_out * (0.0 - z5), np.zeros(n)], axis=1
    )
    d_dn = np.stack([tx_out, ty_out, np.ones(n)], axis=1)
    poca = _poca(p_up, d_up, p_dn, d_dn)

    cls = truth_classes(true_trace)
    data = dict(
        p_set=np.full(n, p_set),
        p_true=p_true,
        p_meas=p_meas,
        raster_x=np.full(n, center_xy[0]),
        raster_y=np.full(n, center_xy[1]),
        delta_meas=delta_meas,
        theta_x=thx,
        theta_y=thy,
        dth_true=dth_true,
        dth_reco=dth_reco,
        t_Al=true_trace["t_Al"],
        t_Cu=true_trace["t_Cu"],
        t_Pb=true_trace["t_Pb"],
        xx0_true=x_over_x0(true_trace["t_Al"], true_trace["t_Cu"], true_trace["t_Pb"]),
        xx0_ref_true=x_over_x0(ref_true["t_Al"], ref_true["t_Cu"], ref_true["t_Pb"]),
        xx0_ref_reco=x_over_x0(ref_reco["t_Al"], ref_reco["t_Cu"], ref_reco["t_Pb"]),
        true_pb=cls["pb"],
        true_cu_only=cls["cu_only"],
        true_al_only=cls["al_only"],
        poca_x=poca[:, 0],
        poca_y=poca[:, 1],
        poca_z=poca[:, 2],
    )
    _store_segments(data, "true", true_trace["segments"])
    _store_segments(data, "ref_true", ref_true["segments"])
    _store_segments(data, "ref_reco", ref_reco["segments"])
    df = pd.DataFrame(data)
    df["pass_reco"] = df.dth_reco <= THETA_CUT
    df["pass_true"] = df.dth_true <= THETA_CUT
    return df


def simulate_equal_exposure(n_per_setting=500_000, seed=0, nodes=None, calibrator=None):
    nodes = raster_nodes() if nodes is None else np.asarray(nodes, float)
    calibrator = calibrator or PofxCache(nmax=2)
    frames = []
    # Exactly equal node exposure within each setting.
    counts = _integer_counts(n_per_setting, np.ones(len(nodes)) / len(nodes))
    for p in MOMENTA:
        for c, n in zip(nodes, counts):
            if n:
                frames.append(
                    simulate_fixed_node(
                        p, int(n), tuple(c), seed=seed, calibrator=calibrator
                    )
                )
    return pd.concat(frames, ignore_index=True), calibrator


def simulate_gradient_exposure(
    n_per_cell=20_000, seed=0, nodes=None, calibrator=None, reference_target=True
):
    """Spatial p-mixture intervention with fixed total incident fluence/cell."""
    nodes = raster_nodes() if nodes is None else np.asarray(nodes, float)
    calibrator = calibrator or PofxCache(nmax=2)
    frames = []
    for cell_id, c in enumerate(nodes):
        fr = momentum_fractions(c[0])
        counts = _integer_counts(n_per_cell, fr)
        for p, n in zip(MOMENTA, counts):
            if n:
                d = simulate_fixed_node(
                    p,
                    int(n),
                    tuple(c),
                    seed=seed + 7919 * cell_id,
                    calibrator=calibrator,
                    reference_target=reference_target,
                )
                d["gradient_cell"] = cell_id
                frames.append(d)
    return pd.concat(frames, ignore_index=True), calibrator


def save_events(df, path):
    """Persist an event table without making pyarrow a hard dependency.

    ``path`` may end in .parquet or .pkl.  Parquet is attempted when requested;
    if no parquet engine is installed the function writes a same-stem .pkl and
    returns that actual path.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        df.to_pickle(path)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".pkl")
        df.to_pickle(fallback)
        return fallback


def load_events(path):
    from pathlib import Path

    path = Path(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)
