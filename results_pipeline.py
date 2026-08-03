#!/usr/bin/env python3
"""
results_pipeline.py  --  fills manuscript Sec. 5.2-5.4 (currently empty).

Two configuration choices matter here, both now reflected in config.py's
DEFAULTS (raster beam, config.py's own change) or forced at runtime
(re-steering, since it's a per-run experimental choice, not a fixed spec):

  BEAM_MODE = 'raster'            (config.py DEFAULT as of this pipeline)
      Sec. 4.1 originally specified a single sigma_xy=1cm spot; the repo's
      own test_beam_covers_target_face proved that misses the 15 cm Cu
      block's 95% span by a factor of ~2 -- no tomogram to reconstruct.
      config.py now defaults to a sigma_xy=1cm pencil beam RASTERED over a
      7x7 grid spanning the Cu face (RASTER_NX/NY/HALF), which is how tagged
      momentum beams actually cover an extended target: narrow,
      well-characterized spot, scanned -- not flooded. This is config.py's
      default now, so no override is needed here.

  STEER_COMPENSATION = 'per_setting'   (config.py default: 'none')
      Sec. 3.3 states the beamline IS re-steered per momentum setting ("as
      in a tuned beamline"); config.py's default leaves the dipole kick
      uncorrected, which is what actually manufactures the large
      momentum-position correlation Sec. 2.3's spatially-structured-artifact
      argument leans on (repo's own
      test_momentum_position_correlation_exists documents this: 4.90 cm
      spread with steer='none' vs 0.04 cm with steer='per_setting', against
      a 0.6 cm voxel). Running with 'per_setting' is what the manuscript AS
      WRITTEN calls for; whether genuine spatially-structured signal
      survives it is exactly the open question this script is built to
      answer -- see the orthogonal-fraction number in the report. If it does
      not survive, Sec. 2.3/4.4's claim needs restating (e.g. to a cosmic
      spectrum, where the momentum-angle correlation is intrinsic rather than
      beamline-imposed), not the config flipped back to 'none' to force a
      result.

eps_M is sourced from eps_quadrature.py (deterministic, Sec. 2.3), NOT from
branch_a.py's Monte-Carlo/bootstrap fit -- branch_a's eps_M decomposes
RECONSTRUCTION noise/resolution/truncation (a detector-characterisation
question), which is a different quantity from the imaging weight's
calibration target.

Four images per Sec. 4.4 (theta_space always at the REFERENCE geometry;
eps_M is what differs between them -- see the clarifying sentence "the
calibration is indexed by x/X0|ref when applied" in Sec. 4.4):

    I_nom   : eps = 0
    I_p     : eps = eps_M_hat(p)                  momentum-only marginal
    I_ideal : eps = eps_M_hat(p, x/X0_true)        full per-event calibration
    I_const : eps = eps_bar (event-weighted mean of the I_ideal eps)  [null control]

Usage:
    python3 results_pipeline.py --n 20000     # fast pass, sanity-check shapes
    python3 results_pipeline.py --n 500000    # manuscript scale (2e6 total)
"""
import argparse
import os

import numpy as np

# ---- force STEER_COMPENSATION before simulate.py is imported ----
# (BEAM_MODE='raster' is now config.py's own default -- see docstring above)
import config
config.STEER_COMPENSATION = "per_setting"

import pandas as pd
import simulate                                  # noqa: E402  (import after config patch)
import branch_b as bb                             # noqa: E402  (roi_masks, metrics, edge_response, voxelise)
from kinematics import theta_space_highland        # noqa: E402
from eps_quadrature import eps_M, eps_M_marginal   # noqa: E402
from config import MATERIALS, MOMENTA, OUT_DIR, THETA_CUT  # noqa: E402


def _xX0(t_al, t_cu, t_pb):
    return (t_al / MATERIALS["Al"]["X0"] + t_cu / MATERIALS["Cu"]["X0"]
            + t_pb / MATERIALS["Pb"]["X0"])


def build_weights(df):
    """Returns (w_nom, w_p, w_ideal, w_const, eps_bar). All theta_space terms
    use the REFERENCE geometry (xx0_ref, i.e. Al+Cu, Pb->Cu) -- this is the
    denominator of Eq. (2)/(7) in every image; only eps_M varies."""
    p = df.p_meas.values
    tspace_ref = theta_space_highland(p, df.xx0_ref.values)

    eps_p = eps_M_marginal(p)                      # momentum-only, at ref path

    X_al_true = MATERIALS["Al"]["rho"] * df.t_Al.values
    X_cu_true = MATERIALS["Cu"]["rho"] * df.t_Cu.values
    X_pb_true = MATERIALS["Pb"]["rho"] * df.t_Pb.values
    eps_ideal = eps_M(p, X_al_true, X_cu_true, X_pb_true)  # full, at TRUE path

    dth = df.dth_reco.values
    with np.errstate(divide="ignore", invalid="ignore"):
        w_nom = np.where(tspace_ref > 0, (dth / tspace_ref) ** 2, 0.0)
        w_p = np.where(tspace_ref > 0, (dth / ((1 + eps_p) * tspace_ref)) ** 2, 0.0)
        w_ideal = np.where(tspace_ref > 0, (dth / ((1 + eps_ideal) * tspace_ref)) ** 2, 0.0)

    eps_bar = float(np.average(eps_ideal, weights=np.where(w_nom > 0, w_nom, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        w_const = np.where(tspace_ref > 0, (dth / ((1 + eps_bar) * tspace_ref)) ** 2, 0.0)

    return w_nom, w_p, w_ideal, w_const, eps_bar, eps_ideal


def artifact_decomposition(img_nom, img_ideal, img_p, counts, min_count=20):
    """A = I_nom - I_ideal, projected onto I_ideal; orthogonal component is
    the genuine momentum/path-length structure (Sec. 4.4 null control).
    R = I_p - I_ideal is the residual after a momentum-only correction."""
    mask = counts >= min_count
    A = np.where(mask, img_nom - img_ideal, np.nan)
    I = np.where(mask, img_ideal, np.nan)
    R = np.where(mask, img_p - img_ideal, np.nan)

    a = A[np.isfinite(A)]; i = I[np.isfinite(I)]; r = R[np.isfinite(R)]
    if a.size == 0 or (i @ i) == 0:
        return dict(artifact_rms=np.nan, orth_fraction=np.nan,
                    residual_rms=np.nan, reduction=np.nan)
    proj_coef = (a @ i) / (i @ i)
    orth = a - proj_coef * i
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan
    art_rms, orth_rms, res_rms = rms(a), rms(orth), rms(r)
    return dict(
        artifact_rms=art_rms,
        orth_fraction=(orth_rms / art_rms) if art_rms else np.nan,
        residual_rms=res_rms,
        reduction=(1 - res_rms / art_rms) if art_rms else np.nan,
        proj_coef=float(proj_coef),
    )


def run(n_per_setting, seed_offset=0, save=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    from moliere import MoliereSampler
    sampler = MoliereSampler(nmax=2)

    frames = []
    for p in MOMENTA:
        df = simulate.simulate_setting(p, n=n_per_setting, mode="moliere",
                                       seed_offset=seed_offset, sampler=sampler)
        frames.append(df)
        print(f"p={p:>4} GeV/c: N={len(df)}  pass_reco={df.pass_reco.mean():.4f}")

    cat = pd.concat(frames, ignore_index=True)
    keep = cat.pass_reco.values
    df = cat[keep].reset_index(drop=True)

    w_nom, w_p, w_ideal, w_const, eps_bar, eps_ideal = build_weights(df)
    print(f"\nevent-weighted mean eps_M (I_const bias) = {eps_bar*100:+.2f}%")

    sample = np.stack([df.poca_x.values, df.poca_y.values, df.poca_z.values], 1)
    bins = (bb.EDGES, bb.EDGES, bb.EDGES)
    counts, _ = np.histogramdd(sample, bins=bins)

    def _acc(w):
        sw, _ = np.histogramdd(sample, bins=bins, weights=w)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(counts > 0, sw / counts, 0.0)

    img_nom, img_p, img_ideal, img_const = (_acc(w_nom), _acc(w_p),
                                             _acc(w_ideal), _acc(w_const))

    # --- illumination check (this is what the beam-coverage fix targets) ---
    pb, cu = bb.roi_masks()
    print(f"\nROI illumination: Pb empty={100*(counts[pb]==0).mean():.1f}%  "
          f"Cu empty={100*(counts[cu]==0).mean():.1f}%")

    # --- SNR/CNR/DP and edge response for all four images ---
    rows = []
    for name, img in [("I_nom", img_nom), ("I_p", img_p),
                      ("I_ideal", img_ideal), ("I_const", img_const)]:
        m = bb.metrics(img, counts=counts)
        m.update(bb.edge_response(img, counts=counts))
        m["image"] = name
        rows.append(m)
    metrics_df = pd.DataFrame(rows)[
        ["image", "SNR_Pb", "SNR_Pb_err", "CNR", "CNR_err",
         "sigma_PSF", "sigma_PSF_err", "edge_10_90", "edge_fit_status"]]
    print("\n" + metrics_df.to_string(index=False))

    # --- artifact map, null control, residual (Sec. 4.4) ---
    art = artifact_decomposition(img_nom, img_ideal, img_p, counts)
    print(f"\nartifact RMS (I_nom - I_ideal)      = {art['artifact_rms']:.4g}")
    print(f"orthogonal (genuine structure) frac = {art['orth_fraction']:.3f}"
         f"   <-- decisive number for Sec. 2.3/4.4's claim")
    print(f"per-momentum residual RMS (I_p-I_ideal) = {art['residual_rms']:.4g}")
    print(f"per-momentum correction reduction       = {art['reduction']*100:.1f}%")

    # --- momentum-position correlation actually realised (cross-check vs.
    #     the repo's own test_momentum_position_correlation_exists) ---
    med_x = df.groupby(np.round(df.p_meas, 1)).poca_x.median()
    print(f"\nmedian PoCA x by (rounded) p_meas:\n{med_x.to_string()}")
    print(f"spread = {med_x.max()-med_x.min():.3f} cm  "
         f"(voxel = {config.VOX_SIZE:.2f} cm)")

    if save:
        np.savez_compressed(
            os.path.join(OUT_DIR, "results_images.npz"),
            I_nom=img_nom, I_p=img_p, I_ideal=img_ideal, I_const=img_const,
            counts=counts, centers=bb.CENTERS, eps_bar=eps_bar,
        )
        metrics_df.to_csv(os.path.join(OUT_DIR, "results_metrics.csv"), index=False)
        pd.Series(art).to_csv(os.path.join(OUT_DIR, "results_artifact.csv"))

    return dict(metrics=metrics_df, artifact=art, eps_bar=eps_bar,
                med_x_spread=float(med_x.max() - med_x.min()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000,
                    help="muons per momentum setting (manuscript: 500000)")
    ap.add_argument("--seed_offset", type=int, default=0)
    a = ap.parse_args()
    run(a.n, seed_offset=a.seed_offset)