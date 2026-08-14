#!/usr/bin/env python3
"""
results_pipeline.py  --  fills manuscript Sec. 5.2-5.4.

Each run writes to its own directory:
    out/seed{seed}_n{n}/          <-- all .npy, .csv, .npz outputs
    out/seed{seed}_n{n}/figs/     <-- figures (via plot_results.py)

Run:
    python3 results_pipeline.py --n 500000 --seed 0
    python3 plot_results.py --outdir out/seed0_n500000

Five images (Sec. 5.4), all denominators evaluated on the REFERENCE geometry:
    I_nom   : w = (dth / theta_space(p, x/X0|ref))^2
    I_p     : w = (dth / ((1+eps_M_hat(p)) * theta_space))^2
    I_ideal : w = w_Q  [same array; both names refer to same image]
    I_const : w = (dth / ((1+eps_bar) * theta_space))^2  [null control]
    I_Q     : w = (dth / theta_RMS(theta_cut, p, x/X0|ref))^2
"""

import os
import argparse

import numpy as np
import pandas as pd

import simulate
import branch_b as bb
from config import MOMENTA, OUT_DIR, MATERIALS
from eps_quadrature import (theta_RMS, theta_RMS_at_cut, optimal_cut,
                             eps_M_marginal, K_OPT)
from kinematics import theta_space_highland


# ------------------------------------------------------------------ weights

def build_weights(df):
    """Compute all five per-event weights for the fixed 200 mrad cut.

    Returns (w_nom, w_p, w_ideal, w_const, w_Q, eps_bar, eps_ref_per_event).
    w_ideal and w_Q are the same array.
    """
    dth      = df.dth_reco.values
    p        = df.p_meas.values
    X_al_ref = df.X_al_ref.values
    X_cu_ref = df.X_cu_ref.values
    X_pb_ref = df.X_pb_ref.values   # zero by construction

    xX0_ref = (X_al_ref / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
               + X_cu_ref / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"])
    tspace_ref = theta_space_highland(p, xX0_ref)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_nom = np.where(tspace_ref > 0, (dth / tspace_ref) ** 2, 0.0)

    eps_p = eps_M_marginal(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_p = np.where(tspace_ref > 0,
                       (dth / ((1.0 + eps_p) * tspace_ref)) ** 2, 0.0)

    trms_ref = theta_RMS(p, X_al_ref, X_cu_ref, X_pb_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_Q = np.where(trms_ref > 0, (dth / trms_ref) ** 2, 0.0)
    w_ideal = w_Q

    with np.errstate(divide="ignore", invalid="ignore"):
        eps_ref_per_event = np.where(tspace_ref > 0,
                                     trms_ref / tspace_ref - 1.0, 0.0)
    eps_bar = float(np.average(eps_ref_per_event,
                               weights=np.where(w_nom > 0, w_nom, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        w_const = np.where(tspace_ref > 0,
                           (dth / ((1.0 + eps_bar) * tspace_ref)) ** 2, 0.0)

    return w_nom, w_p, w_ideal, w_const, w_Q, eps_bar, eps_ref_per_event


# ------------------------------------------------------------------ artifact

def artifact_decomposition(img_nom, img_ideal, img_p, counts, min_count=20):
    mask = counts >= min_count
    A = np.where(mask, img_nom - img_ideal, np.nan)
    I = np.where(mask, img_ideal,           np.nan)
    R = np.where(mask, img_p - img_ideal,   np.nan)

    a = A[np.isfinite(A)]
    i = I[np.isfinite(I)]
    r = R[np.isfinite(R)]

    if a.size == 0 or (i @ i) == 0:
        return dict(artifact_rms=np.nan, orth_fraction=np.nan,
                    residual_rms=np.nan, reduction=np.nan, proj_coef=np.nan)

    proj_coef = float((a @ i) / (i @ i))
    orth      = a - proj_coef * i

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan

    art_rms = rms(a)
    return dict(
        artifact_rms  = art_rms,
        orth_fraction = rms(orth) / art_rms if art_rms else np.nan,
        residual_rms  = rms(r),
        reduction     = (1.0 - rms(r) / art_rms) if art_rms else np.nan,
        proj_coef     = proj_coef,
    )


# ------------------------------------------------------------------ speckle

def speckle_metric(img, counts, min_count=20, n_boot=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    edges = bb.EDGES
    cx = 0.5 * (edges[:-1] + edges[1:])
    X, Y, Z = np.meshgrid(cx, cx, cx, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    annulus = (R >= 9.0) & (R <= 11.0) & (np.abs(Z) <= 6.0) & (counts >= min_count)
    vals = img[annulus]
    if vals.size < 10:
        return dict(cv=np.nan, cv_err=np.nan, n_vox=int(vals.size))
    cv = (float(np.std(vals) / np.abs(np.mean(vals)))
          if np.mean(vals) != 0 else np.nan)
    boot = [float(np.std(rng.choice(vals, size=vals.size, replace=True)) /
                  np.abs(np.mean(rng.choice(vals, size=vals.size, replace=True))))
            for _ in range(n_boot)]
    cv_err = float(np.std(boot))
    return dict(cv=cv, cv_err=cv_err, n_vox=int(vals.size))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int, default=500_000)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--outdir", default=OUT_DIR,
                    help="Base output directory. A run-specific subdirectory "
                         "seed{seed}_n{n} is created automatically.")
    args = ap.parse_args()

    # each run gets its own subdirectory
    run_tag = f"seed{args.seed}_n{args.n}"
    out = os.path.join(args.outdir, run_tag)
    os.makedirs(out, exist_ok=True)

    from moliere import MoliereSampler
    sampler = MoliereSampler(nmax=2)

    print(f"Simulating {args.n} muons/setting x {len(MOMENTA)} settings "
          f"(seed={args.seed}) ...")
    frames = []
    for p in MOMENTA:
        df = simulate.simulate_setting(p, n=args.n, seed_offset=args.seed,
                                       sampler=sampler)
        frames.append(df)
    cat = pd.concat(frames, ignore_index=True)
    print(f"Total events generated: {len(cat)}")
    print(f"Pass reco (200 mrad cut): {cat.pass_reco.mean()*100:.1f}%")

    # ------------------------------------------------------------------ adaptive cut
    theta_cut_opt = optimal_cut(cat.p_meas.values,
                                cat.X_al_ref.values,
                                cat.X_cu_ref.values,
                                cat.X_pb_ref.values,
                                k_opt=K_OPT)
    adaptive_pass = cat.dth_reco.values < theta_cut_opt
    df_adapt      = cat[adaptive_pass].reset_index(drop=True)
    cut_adapt     = theta_cut_opt[adaptive_pass]
    trms_adapt    = theta_RMS_at_cut(df_adapt.p_meas.values,
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
    df   = cat[keep].reset_index(drop=True)

    w_nom, w_p, w_ideal, w_const, w_Q, eps_bar, eps_ref = build_weights(df)
    print(f"\nevent-weighted mean eps_M (reference path, I_const bias) = "
          f"{eps_bar*100:+.2f}%")

    sample = np.stack([df.poca_x.values, df.poca_y.values, df.poca_z.values], 1)
    bins   = (bb.EDGES, bb.EDGES, bb.EDGES)
    counts, _ = np.histogramdd(sample, bins=bins)

    def _acc(w):
        sw, _ = np.histogramdd(sample, bins=bins, weights=w)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(counts > 0, sw / counts, 0.0)

    img_nom   = _acc(w_nom)
    img_p     = _acc(w_p)
    img_ideal = _acc(w_ideal)
    img_const = _acc(w_const)
    img_Q     = img_ideal

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
    sp_nom  = speckle_metric(img_nom, counts)
    sp_Q    = speckle_metric(img_Q,   counts)
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

    # ------------------------------------------------------------------ artifact
    art = artifact_decomposition(img_nom, img_ideal, img_p, counts)
    print(f"\nartifact RMS (I_nom - I_ideal)           = {art['artifact_rms']:.4g}")
    print(f"orthogonal (genuine structure) fraction  = {art['orth_fraction']:.3f}"
          f"   <-- decisive number for Sec. 5.4")
    print(f"residual RMS (I_p  - I_ideal)            = {art['residual_rms']:.4g}")
    print(f"per-momentum correction reduction        = {art['reduction']:.3f}")

    # ------------------------------------------------------------------ save
    # flat per-file outputs
    metrics_df.to_csv(os.path.join(out, "metrics.csv"), index=False)
    speckle_df.to_csv(os.path.join(out, "speckle.csv"), index=False)
    pd.Series(art).to_csv(os.path.join(out, "artifact.csv"))

    for name, arr in [("img_nom",         img_nom),
                      ("img_p",           img_p),
                      ("img_ideal",       img_ideal),
                      ("img_const",       img_const),
                      ("img_Q",           img_Q),
                      ("img_Q_adaptive",  img_Q_adapt),
                      ("counts",          counts),
                      ("counts_adaptive", counts_adapt)]:
        np.save(os.path.join(out, f"{name}.npy"), arr)

    # bundled npz for plot_results.py
    centers = 0.5 * (bb.EDGES[:-1] + bb.EDGES[1:])
    np.savez_compressed(
        os.path.join(out, "images.npz"),
        centers=centers,
        counts=counts,
        counts_adaptive=counts_adapt,
        **{"I_nom": img_nom, "I_p": img_p, "I_ideal": img_ideal,
           "I_const": img_const, "I_Q": img_Q,
           "I_Q_adaptive": img_Q_adapt})

    print(f"\nOutputs written to {out}/")
    print(f"Run:  python3 plot_results.py --outdir {out}")


if __name__ == "__main__":
    main()