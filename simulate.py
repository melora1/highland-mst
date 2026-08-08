"""Event pipeline (plan Sec. 3). Produces one flat table per momentum setting.

Everything downstream (Branch A, Branch B, correction) reads these tables.
Nothing downstream re-simulates.

Mode:
  mode='moliere'  -> theta_x, theta_y from the truncated Moliere expansion
  mode='gauss'    -> theta_x, theta_y ~ N(0, theta0^2), theta0 from Eq. (1)
                     at p_true and the per-event x/X0.  This is the n=0
                     control run used to isolate the truncation bias.

Deflection resolution sigma_delta is NOT injected by hand: it emerges from
smearing the station hits, so the stations-3/4 correlation between
p_meas and Delta-theta_space (Sec. 3.4) is present by construction.
"""

import os

import numpy as np
import pandas as pd

from config import (BEAM_MODE, BL, MOM_BITE, N_PER_SETTING, OUT_DIR,
                    RASTER_HALF, RASTER_NX, RASTER_NY, SEED_BASE,
                    STEER_COMPENSATION,
                    SIGMA_DIV, SIGMA_HIT, SIGMA_XY, STATION_Z, THETA_CUT,
                    UNIFORM_HALF, Z_MAGNET_CM)
from geometry import areal_densities, trace_ref, trace_true, x_over_X0
from kinematics import theta0_highland
from moliere import MoliereSampler


def _slope(z1, z2, u1, u2):
    return (u2 - u1) / (z2 - z1)


def _propagate(x, tx, z_from, z_to):
    return x + tx * (z_to - z_from)


def _poca(p1, d1, p2, d2):
    """Midpoint of the shortest segment between two lines. Vectorised."""
    w0 = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = np.einsum("ij,ij->i", d1, w0)
    e = np.einsum("ij,ij->i", d2, w0)
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    c1 = p1 + s[:, None] * d1
    c2 = p2 + t[:, None] * d2
    return 0.5 * (c1 + c2)


def simulate_setting(p_set, n=N_PER_SETTING, mode="moliere", seed_offset=0,
                     sampler=None):
    rng = np.random.default_rng(SEED_BASE + int(p_set * 1000) + seed_offset)

    # ---------------------------------------------------------- 1. beam
    if BEAM_MODE == "pencil":
        x0 = rng.normal(0.0, SIGMA_XY, n)
        y0 = rng.normal(0.0, SIGMA_XY, n)
    elif BEAM_MODE == "uniform":
        # Transverse profile only. Whether the dipole deflection is corrected
        # is STEER_COMPENSATION's job, not this branch's -- doing it here too
        # double-compensates.
        x0 = rng.uniform(-UNIFORM_HALF, UNIFORM_HALF, n)
        y0 = rng.uniform(-UNIFORM_HALF, UNIFORM_HALF, n)
    elif BEAM_MODE == "raster":
        # sigma_xy pencil beam scanned over an NX x NY grid of center
        # positions, equal exposure per node (tiled, not randomly assigned,
        # so exposure is exactly uniform across nodes even at modest n).
        cx = np.linspace(-RASTER_HALF, RASTER_HALF, RASTER_NX)
        cy = np.linspace(-RASTER_HALF, RASTER_HALF, RASTER_NY)
        nodes = np.array([(a, b) for a in cx for b in cy])   # (RASTER_NX*RASTER_NY, 2)
        node_idx = np.arange(n) % nodes.shape[0]
        rng.shuffle(node_idx)                                  # decorrelate from hit order
        centers = nodes[node_idx]
        x0 = centers[:, 0] + rng.normal(0.0, SIGMA_XY, n)
        y0 = centers[:, 1] + rng.normal(0.0, SIGMA_XY, n)
    else:
        raise ValueError(f"unknown BEAM_MODE {BEAM_MODE!r}")
    tx0 = rng.normal(0.0, SIGMA_DIV, n)     # slope dx/dz, pre-magnet
    ty0 = rng.normal(0.0, SIGMA_DIV, n)
    p_true = p_set * (1.0 + MOM_BITE * rng.normal(0.0, 1.0, n))

    # ---------------------------------------------------------- 2. spectrometer
    # true trajectory: straight to magnet centre, kick in x, straight on.
    delta_true = 0.3 * BL / p_true          # rad, Eq. (5)
    if STEER_COMPENSATION == "per_setting":
        # Retune the beamline for this setting: shift the incoming beam so the
        # NOMINAL momentum lands on axis. Events off nominal (momentum bite)
        # still walk, which is the only physical momentum-position correlation
        # left. Applied to x0 before propagation.
        x0 = x0 - (0.3 * BL / p_set) * (0.0 - Z_MAGNET_CM)
    elif STEER_COMPENSATION != "none":
        raise ValueError(f"unknown STEER_COMPENSATION {STEER_COMPENSATION!r}")
    z1, z2, z3, z4, z5, z6 = STATION_Z

    # reference plane for the pre-magnet segment: define x0,y0 at z = z1
    x_at = lambda z: _propagate(x0, tx0, z1, z)
    y_at = lambda z: _propagate(y0, ty0, z1, z)

    h1x, h2x = x_at(z1), x_at(z2)
    h1y, h2y = y_at(z1), y_at(z2)

    xm = x_at(Z_MAGNET_CM)
    ym = y_at(Z_MAGNET_CM)
    tx1 = tx0 + delta_true                  # post-magnet true slope (x)
    ty1 = ty0                               # no bend in y

    h3x = _propagate(xm, tx1, Z_MAGNET_CM, z3)
    h4x = _propagate(xm, tx1, Z_MAGNET_CM, z4)
    h3y = _propagate(ym, ty1, Z_MAGNET_CM, z3)
    h4y = _propagate(ym, ty1, Z_MAGNET_CM, z4)

    # smear hits 1-4
    s = lambda a: a + rng.normal(0.0, SIGMA_HIT, n)
    m1x, m2x, m3x, m4x = s(h1x), s(h2x), s(h3x), s(h4x)
    m1y, m2y, m3y, m4y = s(h1y), s(h2y), s(h3y), s(h4y)

    # ---------------------------------------------------------- 3. momentum tag
    th_pre = _slope(z1, z2, m1x, m2x)
    th_post = _slope(z3, z4, m3x, m4x)
    delta_meas = th_post - th_pre                       # Eq. (6)
    p_meas = 0.3 * BL / np.abs(delta_meas)              # Eq. (5) inverted

    # reconstructed incoming track (stations 3-4)
    tx_in = th_post
    ty_in = _slope(z3, z4, m3y, m4y)
    x_in = m3x + tx_in * (0.0 - z3)                     # in-track x at z=0
    y_in = m3y + ty_in * (0.0 - z3)

    # ---------------------------------------------------------- 4. path lengths
    # TRUE trajectory (for sampling + Branch A prediction)
    o_true = np.stack([_propagate(xm, tx1, Z_MAGNET_CM, 0.0),
                       _propagate(ym, ty1, Z_MAGNET_CM, 0.0),
                       np.zeros(n)], axis=1)
    u_true = np.stack([tx1, ty1, np.ones(n)], axis=1)
    u_true /= np.linalg.norm(u_true, axis=1, keepdims=True)
    tAl, tCu, tPb = trace_true(o_true, u_true)
    xx0_true = x_over_X0(tAl, tCu, tPb)
    X_al, X_cu, X_pb = areal_densities(tAl, tCu, tPb)

    # RECO trajectory through the REFERENCE geometry (weight denominator)
    o_rec = np.stack([x_in, y_in, np.zeros(n)], axis=1)
    u_rec = np.stack([tx_in, ty_in, np.ones(n)], axis=1)
    u_rec /= np.linalg.norm(u_rec, axis=1, keepdims=True)
    rAl, rCu, rPb = trace_ref(o_rec, u_rec)
    xx0_ref = x_over_X0(rAl, rCu, rPb)
    X_al_ref, X_cu_ref, X_pb_ref = areal_densities(rAl, rCu, rPb)  # X_pb_ref == 0 always

    # ---------------------------------------------------------- 5. scatter
    hit_target = xx0_true > 0.0
    thx = np.zeros(n)
    thy = np.zeros(n)
    if mode == "moliere":
        if sampler is None:
            sampler = MoliereSampler(nmax=2)
        m = hit_target
        thx[m], thy[m] = sampler.sample(p_true[m], X_al[m], X_cu[m], X_pb[m], rng)
    elif mode == "gauss":
        t0 = theta0_highland(p_true, xx0_true)
        thx = rng.normal(0.0, 1.0, n) * t0
        thy = rng.normal(0.0, 1.0, n) * t0
    else:
        raise ValueError(mode)

    dth_true = np.sqrt(thx ** 2 + thy ** 2)

    # kink applied at the midpoint of the in-target path
    # (entry point + half the traversed length along u_true)
    L_tot = tAl + tCu + tPb
    # entry parameter: find where the ray enters the Al cube
    # (re-derive cheaply: PoCA-truth vertex only needs to be inside the target)
    z_vtx = np.zeros(n)      # target is centred at z=0; midpoint ~ z=0 for
    # near-axial beams. Exact entry-point solve is in geometry._box_path;
    # for the small divergence here the z=0 midpoint is accurate to <1 mm.
    x_vtx = o_true[:, 0]
    y_vtx = o_true[:, 1]

    tx_out_true = tx1 + thx
    ty_out_true = ty1 + thy

    # ---------------------------------------------------------- 6. downstream
    h5x = _propagate(x_vtx, tx_out_true, z_vtx, z5)
    h6x = _propagate(x_vtx, tx_out_true, z_vtx, z6)
    h5y = _propagate(y_vtx, ty_out_true, z_vtx, z5)
    h6y = _propagate(y_vtx, ty_out_true, z_vtx, z6)
    m5x, m6x, m5y, m6y = s(h5x), s(h6x), s(h5y), s(h6y)

    tx_out = _slope(z5, z6, m5x, m6x)
    ty_out = _slope(z5, z6, m5y, m6y)

    dth_reco = np.sqrt((tx_out - tx_in) ** 2 + (ty_out - ty_in) ** 2)

    # ---------------------------------------------------------- 7. PoCA
    p_up = np.stack([m3x + tx_in * (0.0 - z3),
                     m3y + ty_in * (0.0 - z3),
                     np.zeros(n)], axis=1)
    d_up = np.stack([tx_in, ty_in, np.ones(n)], axis=1)
    p_dn = np.stack([m5x + tx_out * (0.0 - z5),
                     m5y + ty_out * (0.0 - z5),
                     np.zeros(n)], axis=1)
    d_dn = np.stack([tx_out, ty_out, np.ones(n)], axis=1)
    poca = _poca(p_up, d_up, p_dn, d_dn)

    # ---------------------------------------------------------- 8. table
    df = pd.DataFrame(dict(
        p_set=np.full(n, p_set),
        p_true=p_true,
        p_meas=p_meas,
        delta_meas=delta_meas,
        theta_x=thx,
        theta_y=thy,
        dth_true=dth_true,
        dth_reco=dth_reco,
        xx0_true=xx0_true,
        xx0_ref=xx0_ref,
        X_al_ref=X_al_ref, X_cu_ref=X_cu_ref, X_pb_ref=X_pb_ref,
        t_Al=tAl, t_Cu=tCu, t_Pb=tPb,
        poca_x=poca[:, 0], poca_y=poca[:, 1], poca_z=poca[:, 2],
    ))
    df["pass_reco"] = df.dth_reco <= THETA_CUT
    df["pass_true"] = df.dth_true <= THETA_CUT
    return df


def run(mode="moliere", tag=None, momenta=None):
    from config import MOMENTA
    momenta = momenta or MOMENTA
    tag = tag or mode
    os.makedirs(OUT_DIR, exist_ok=True)
    sampler = MoliereSampler(nmax=2) if mode == "moliere" else None
    for p in momenta:
        df = simulate_setting(p, mode=mode, sampler=sampler)
        path = os.path.join(OUT_DIR, f"events_{tag}_p{p:.1f}.parquet")
        df.to_parquet(path, index=False)
        print(f"[{tag}] p={p} GeV/c  N={len(df)}  "
              f"pass={df.pass_reco.mean():.4f}  -> {path}")
    if sampler is not None:
        print(f"[{tag}] max clipped pdf fraction = {sampler.max_clipped:.3e} "
              f"(must be << 1e-2; see moliere.py warning)")


if __name__ == "__main__":
    import sys
    run(mode=sys.argv[1] if len(sys.argv) > 1 else "moliere")