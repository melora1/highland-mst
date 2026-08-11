#!/usr/bin/env python3
"""
results_pipeline.py  --  fills manuscript Sec. 5.2-5.4.

Two configuration choices matter here, both now reflected in config.py's
DEFAULTS (raster beam, config.py's own change) or forced at runtime
(re-steering, since it's a per-run experimental choice, not a fixed spec):

  BEAM_MODE = 'raster'            (config.py DEFAULT as of this pipeline)
  STEER_COMPENSATION = 'per_setting'   (config.py default: 'none')

eps_M is sourced from eps_quadrature.py (deterministic, Sec. 2.3).

Five images per Sec. 5.4.  ALL use theta_space evaluated at the REFERENCE
geometry (Al+Cu, Pb excluded) in their denominators -- this is the
definition of the estimator; only the denominator's functional form varies:

    I_nom   : w_nom  = (dth / theta_space(p, x/X0|ref))^2
                       Standard Highland weight -- Eq. (6).

    I_p     : w_p    = (dth / ((1+eps_M_hat(p)) * theta_space(p, x/X0|ref)))^2
                       Per-momentum correction only (marginal at axial ref path).

    I_ideal : w_ideal = w_Q  (see below)
                       DEFINITION: I_ideal is the acceptance-matched estimator
                       I_Q evaluated at the REFERENCE geometry.  Both names
                       refer to the same weight; I_ideal is used as the
                       artifact benchmark and I_Q as the estimator name in
                       Sec. 6.  They are identical by construction: the
                       denominator theta_RMS(theta_cut, p, x/X0|ref) is the
                       quadrature-computed truncated second moment of the
                       Moliere distribution within the same 200 mrad acceptance
                       the numerator is measured in, evaluated on the REFERENCE
                       path.  This gives E[w_ideal] = E[w_Q] = 1 exactly under
                       the reference hypothesis for every (p, x/X0, theta_cut).

                       NOTE: a previous implementation incorrectly used
                       eps_M(p, x/X0_TRUE) -- the true path including Pb -- in
                       the denominator.  That is incoherent because (a) it
                       uses oracle knowledge unavailable in deployment, (b) it
                       gives E[w_ideal] != 1 under the reference hypothesis
                       wherever Pb is present, and (c) it makes I_ideal != I_Q
                       contrary to the manuscript's claim.  The fix is to
                       define I_ideal = I_Q throughout, using ONLY reference-
                       geometry quantities.

    I_const : w_const = (dth / ((1+eps_bar) * theta_space(p, x/X0|ref)))^2
                        Null control: eps_bar is the w_nom-weighted mean of
                        eps_M(p, x/X0|ref) over the event sample, giving a
                        momentum-independent uniform rescale so that
                        I_const - I_ideal is exactly proportional to I_ideal
                        and carries no spatial structure beyond the image.
                        NOTE: eps_bar is now computed from the REFERENCE-path
                        eps_M (consistent with w_ideal/w_Q), not the true-path
                        eps_M used in the old implementation.

    I_Q     : w_Q = w_ideal  (alias; same array, kept for pipeline clarity)

Usage:
    python3 results_pipeline.py --n 20000     # fast pass, sanity-check shapes
    python3 results_pipeline.py --n 500000    # manuscript scale (2e6 total)
"""
import argparse
import os

import numpy as np

# ---- force STEER_COMPENSATION before simulate.py is imported ----
import config
config.STEER_COMPENSATION = "per_setting"

import pandas as pd
import simulate                                  # noqa: E402
import branch_b as bb                             # noqa: E402
from kinematics import theta_space_highland        # noqa: E402
from eps_quadrature import (eps_M, eps_M_marginal, theta_RMS, K_OPT,
                            optimal_cut, theta_RMS_at_cut)   # noqa: E402
from config import MATERIALS, MOMENTA, OUT_DIR, THETA_CUT  # noqa: E402


def _xX0(t_al, t_cu, t_pb):
    return (t_al / MATERIALS["Al"]["X0"] + t_cu / MATERIALS["Cu"]["X0"]
            + t_pb / MATERIALS["Pb"]["X0"])


def build_weights(df):
    """Returns (w_nom, w_p, w_ideal, w_const, w_Q, eps_bar).

    KEY INVARIANT: w_ideal IS w_Q.  Both point to the same array.
    I_ideal and I_Q are two names for the same image: the acceptance-matched
    estimator with denominator theta_RMS(theta_cut, p, x/X0|ref), evaluated
    entirely on the REFERENCE geometry.

    All five weights use theta_space or theta_RMS evaluated at the REFERENCE
    geometry (xx0_ref, i.e. Al+Cu with Pb replaced by Cu).  No weight ever
    uses the true path (t_Pb) in its denominator.  This is required because:
      - The imaging signal is EXCESS scattering relative to the reference.
      - Using the true path in the denominator would normalize away the
        Pb contrast being imaged.
      - E[w] = 1 under the reference hypothesis requires the denominator to
        be the expected value of the numerator under that same hypothesis.

    w_Q / w_ideal denominator: theta_RMS(theta_cut, p, x/X0|ref)
      = the quadrature-computed truncated second moment of the Moliere
        distribution within the 200 mrad acceptance, on the reference path.
      = (1 + eps_M(p, x/X0|ref)) * theta_space(p, x/X0|ref)
      where eps_M is evaluated on X_al_ref, X_cu_ref, X_pb_ref=0.

    This gives E[w_Q] = 1 exactly for every (p, x/X0|ref, theta_cut),
    unlike w_nom which uses Highland theta_space and has E[w_nom] != 1.
    """
    p = df.p_meas.values
    tspace_ref = theta_space_highland(p, df.xx0_ref.values)

    # --- momentum-only correction (I_p) ---
    # eps_M_marginal evaluates at the axial reference path (Al 10cm + Cu 15cm,
    # no Pb): the per-momentum marginal of the reference-geometry eps_M.
    eps_p = eps_M_marginal(p)

    dth = df.dth_reco.values

    # --- I_nom: standard Highland weight ---
    with np.errstate(divide="ignore", invalid="ignore"):
        w_nom = np.where(tspace_ref > 0, (dth / tspace_ref) ** 2, 0.0)

    # --- I_p: per-momentum correction only ---
    with np.errstate(divide="ignore", invalid="ignore"):
        w_p = np.where(tspace_ref > 0,
                       (dth / ((1.0 + eps_p) * tspace_ref)) ** 2, 0.0)

    # --- I_ideal = I_Q: acceptance-matched estimator at the REFERENCE path ---
    # theta_RMS(theta_cut, p, x/X0|ref) is computed from the per-event
    # reference-geometry material breakdown: X_al_ref, X_cu_ref, X_pb_ref.
    # X_pb_ref = 0 everywhere by definition (Pb is excluded from the reference
    # geometry: trace_ref maps Pb -> Cu, so df.X_pb_ref is identically zero).
    trms_ref = theta_RMS(p,
                         df.X_al_ref.values,
                         df.X_cu_ref.values,
                         df.X_pb_ref.values)   # X_pb_ref == 0 by construction
    with np.errstate(divide="ignore", invalid="ignore"):
        w_Q = np.where(trms_ref > 0, (dth / trms_ref) ** 2, 0.0)

    # I_ideal is defined as I_Q.  Same array, two names.
    w_ideal = w_Q

    # --- I_const: null control ---
    # eps_bar is the w_nom-weighted mean of eps_M on the REFERENCE path,
    # giving a momentum-independent uniform rescale.
    # eps_M on the reference path = (trms_ref / tspace_ref) - 1
    # (equivalent to eps_M(p, X_al_ref, X_cu_ref, X_pb_ref=0)).
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_ref_per_event = np.where(tspace_ref > 0,
                                     trms_ref / tspace_ref - 1.0, 0.0)
    eps_bar = float(np.average(eps_ref_per_event,
                               weights=np.where(w_nom > 0, w_nom, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        w_const = np.where(tspace_ref > 0,
                           (dth / ((1.0 + eps_bar) * tspace_ref)) ** 2, 0.0)

    return w_nom, w_p, w_ideal, w_const, w_Q, eps_bar, eps_ref_per_event


def artifact_decomposition(img_nom, img_ideal, img_p, counts, min_count=20):
    """A = I_nom - I_ideal, projected onto I_ideal; orthogonal component is
    the genuine momentum/path-length structure (Sec. 5.4 null control).
    R = I_p - I_ideal is the residual after a momentum-only correction.

    Because I_ideal = I_Q (acceptance-matched, reference-path estimator),
    this decomposition measures the artifact of the Highland mismatch relative
    to the bias-free benchmark, and the residual measures what a momentum-only
    correction leaves behind (the path-length dependence of eps_M).
    """
    mask = counts >= min_count
    A = np.where(mask, img_nom - img_ideal, np.nan)
    I = np.where(mask, img_ideal, np.nan)
    R = np.where(mask, img_p - img_ideal, np.nan)

    a = A[np.isfinite(A)]
    i = I[np.isfinite(I)]
    r = R[np.isfinite(R)]
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


def speckle_metric(img, counts, min_count=20, n_boot=500, rng=None):
    """Voxel-to-voxel coefficient of variation in a flat-truth (pure-Al)
    annulus (r in [9,11] cm, |z| <= 6 cm).  Any spread there is estimator
    noise, not signal.  Returns CV with voxel-level bootstrap CI.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # build the annular mask in the 50^3 voxel grid
    edges = bb.EDGES
    cx = 0.5 * (edges[:-1] + edges[1:])
    X, Y, Z = np.meshgrid(cx, cx, cx, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    annulus = (R >= 9.0) & (R <= 11.0) & (np.abs(Z) <= 6.0) & (counts >= min_count)
    vals = img[annulus]
    if vals.size < 10:
        return dict(cv=np.nan, cv_err=np.nan, n_vox=int(vals.size))
    cv = float(np.std(vals) / np.abs(np.mean(vals))) if np.mean(vals) != 0 else np.nan
    # bootstrap
    boot = [float(np.std(rng.choice(vals, size=vals.size, replace=True)) /
                  np.abs(np.mean(rng.choice(vals, size=vals.size, replace=True))))
            for _ in range(n_boot)]
    cv_err = float(np.std(boot))
    return dict(cv=cv, cv_err=cv_err, n_vox=int(vals.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from moliere import MoliereSampler
    sampler = MoliereSampler(nmax=2)

    print(f"Simulating {args.n} muons/setting x {len(MOMENTA)} settings "
          f"(seed={args.seed}) ...")
    frames = []
    for p in MOMENTA:
        df = simulate.simulate_setting(p, n=args.n, seed_offset=args.seed, sampler=sampler)
        frames.append(df)
    cat = pd.concat(frames, ignore_index=True)
    print(f"Total events generated: {len(cat)}")
    print(f"Pass reco (200 mrad cut): {cat.pass_reco.mean()*100:.1f}%")

    # ------------------------------------------------------------------ adaptive cut
    # Applied to the full (unfiltered) catalogue because the adaptive cut is
    # generally tighter than 200 mrad and selects a subset of those events.
    theta_cut_opt = optimal_cut(cat.p_meas.values,
                                cat.X_al_ref.values,
                                cat.X_cu_ref.values,
                                cat.X_pb_ref.values,
                                k_opt=K_OPT)
    adaptive_pass = cat.dth_reco.values < theta_cut_opt
    df_adapt = cat[adaptive_pass].reset_index(drop=True)
    cut_adapt = theta_cut_opt[adaptive_pass]
    trms_adapt = theta_RMS_at_cut(df_adapt.p_meas.values,
                                  df_adapt.X_al_ref.values,
                                  df_adapt.X_cu_ref.values,
                                  df_adapt.X_pb_ref.values,
                                  cut_adapt)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_Q_adapt = np.where(trms_adapt > 0,
                             (df_adapt.dth_reco.values / trms_adapt) ** 2, 0.0)
    print(f"\nadaptive cut (k_opt={K_OPT}): "
          f"{adaptive_pass.mean()*100:.1f}% of ALL generated events pass "
          f"(fixed 200 mrad cut passes {cat.pass_reco.mean()*100:.1f}%)")

    # ------------------------------------------------------------------ fixed-cut images
    keep = cat.pass_reco.values
    df = cat[keep].reset_index(drop=True)

    w_nom, w_p, w_ideal, w_const, w_Q, eps_bar, eps_ref = build_weights(df)
    print(f"\nevent-weighted mean eps_M (reference path, I_const bias) = "
          f"{eps_bar*100:+.2f}%")

    sample = np.stack([df.poca_x.values, df.poca_y.values, df.poca_z.values], 1)
    bins = (bb.EDGES, bb.EDGES, bb.EDGES)
    counts, _ = np.histogramdd(sample, bins=bins)

    def _acc(w):
        sw, _ = np.histogramdd(sample, bins=bins, weights=w)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(counts > 0, sw / counts, 0.0)

    img_nom   = _acc(w_nom)
    img_p     = _acc(w_p)
    img_ideal = _acc(w_ideal)   # = img_Q by definition
    img_const = _acc(w_const)
    img_Q     = img_ideal       # alias; same array

    # ------------------------------------------------------------------ adaptive image
    sample_a = np.stack([df_adapt.poca_x.values, df_adapt.poca_y.values,
                         df_adapt.poca_z.values], 1)
    counts_adapt, _ = np.histogramdd(sample_a, bins=bins)
    sw_a, _ = np.histogramdd(sample_a, bins=bins, weights=w_Q_adapt)
    with np.errstate(divide="ignore", invalid="ignore"):
        img_Q_adapt = np.where(counts_adapt > 0, sw_a / counts_adapt, 0.0)

    # ------------------------------------------------------------------ illumination
    pb, cu = bb.roi_masks()
    print(f"\nROI illumination: Pb empty={100*(counts[pb]==0).mean():.1f}%  "
          f"Cu empty={100*(counts[cu]==0).mean():.1f}%")

    # ------------------------------------------------------------------ metrics
    rows = []
    for name, img in [("I_nom",   img_nom),
                      ("I_p",     img_p),
                      ("I_ideal", img_ideal),
                      ("I_const", img_const),
                      ("I_Q",     img_Q)]:
        m = bb.metrics(img, counts=counts)
        m.update(bb.edge_response(img, counts=counts))
        m["image"] = name
        rows.append(m)
    m_adapt = bb.metrics(img_Q_adapt, counts=counts_adapt)
    m_adapt.update(bb.edge_response(img_Q_adapt, counts=counts_adapt))
    m_adapt["image"] = "I_Q_adaptive"
    rows.append(m_adapt)
    metrics_df = pd.DataFrame(rows)[
        ["image", "SNR_Pb", "SNR_Pb_err", "CNR", "CNR_err",
         "sigma_PSF", "sigma_PSF_err", "edge_10_90", "edge_fit_status"]]
    print("\n" + metrics_df.to_string(index=False))

    # adaptive-cut gain vs I_nom
    snr_nom   = metrics_df.loc[metrics_df.image == "I_nom",        "SNR_Pb"].iloc[0]
    snr_adapt = metrics_df.loc[metrics_df.image == "I_Q_adaptive", "SNR_Pb"].iloc[0]
    cnr_nom   = metrics_df.loc[metrics_df.image == "I_nom",        "CNR"].iloc[0]
    cnr_adapt = metrics_df.loc[metrics_df.image == "I_Q_adaptive", "CNR"].iloc[0]
    print(f"\nADAPTIVE-CUT GAIN (empirical vs. I_nom @ 200 mrad):"
          f"\n  SNR_Pb: {snr_nom:.4f} -> {snr_adapt:.4f}  "
          f"({(snr_adapt/snr_nom-1)*100:+.1f}%)"
          f"\n  CNR:    {cnr_nom:.4f} -> {cnr_adapt:.4f}  "
          f"({(cnr_adapt/cnr_nom-1)*100:+.1f}%)")

    # ------------------------------------------------------------------ speckle
    # I_nom vs I_Q at fixed 200 mrad cut.
    # Because I_ideal = I_Q, img_Q = img_ideal here.
    sp_nom = speckle_metric(img_nom, counts)
    sp_Q   = speckle_metric(img_Q,   counts)
    gap     = sp_nom["cv"] - sp_Q["cv"]
    gap_err = np.hypot(sp_nom["cv_err"], sp_Q["cv_err"])
    sig     = abs(gap) / gap_err if gap_err else np.nan
    print(f"\nspeckle (coeff. of variation, flat Al annulus, "
          f"n_vox={sp_nom['n_vox']}):"
          f"\n  I_nom:   {sp_nom['cv']:.4f} +/- {sp_nom['cv_err']:.4f}"
          f"\n  I_Q:     {sp_Q['cv']:.4f}  +/- {sp_Q['cv_err']:.4f}"
          f"\n  gap = {gap:.4f} +/- {gap_err:.4f}  ({sig:.1f} sigma)"
          f"   {'REDUCED' if gap > 0 else 'not reduced'} vs I_nom"
          f"{' -- NOT significant at this n' if sig < 2 else ''}")

    speckle_df = pd.DataFrame([
        dict(image="I_nom", **sp_nom),
        dict(image="I_Q",   **sp_Q),
        dict(image="gap",   cv=gap, cv_err=gap_err, n_vox=sp_nom["n_vox"]),
    ])

    # ------------------------------------------------------------------ artifact decomposition
    # I_ideal = I_Q, so this measures the Highland artifact against the
    # bias-free, acceptance-matched benchmark.
    art = artifact_decomposition(img_nom, img_ideal, img_p, counts)
    print(f"\nartifact RMS (I_nom - I_ideal)           = {art['artifact_rms']:.4g}")
    print(f"orthogonal (genuine structure) fraction  = {art['orth_fraction']:.3f}"
          f"   <-- decisive number for Sec. 5.4")
    print(f"residual RMS (I_p  - I_ideal)            = {art['residual_rms']:.4g}")
    print(f"per-momentum correction reduction        = {art['reduction']:.3f}")

    # ------------------------------------------------------------------ save outputs
    out = args.outdir
    metrics_df.to_csv(os.path.join(out, f"metrics_seed{args.seed}.csv"), index=False)
    speckle_df.to_csv(os.path.join(out, f"speckle_seed{args.seed}.csv"), index=False)
    for name, arr in [("img_nom",   img_nom),
                      ("img_p",     img_p),
                      ("img_ideal", img_ideal),
                      ("img_const", img_const),
                      ("img_Q",     img_Q),
                      ("img_Q_adaptive", img_Q_adapt),
                      ("counts",    counts),
                      ("counts_adaptive", counts_adapt)]:
        np.save(os.path.join(out, f"{name}_seed{args.seed}.npy"), arr)
    print(f"\nOutputs written to {out}/  (seed={args.seed})")


if __name__ == "__main__":
    main()