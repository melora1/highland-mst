#!/usr/bin/env python3
"""
plot_results.py  --  journal-ready figures from results_pipeline.py's output.

Journal-style conventions applied throughout (Overleaf/LaTeX target):
  * No in-figure titles/notes/explanatory text. Captions belong in the
    manuscript's \\caption{} on Overleaf, not baked into the raster -- every
    figure below prints its caption text to stdout instead, ready to paste.
  * Multi-panel figures get (a)/(b)/(c)... labels, the standard referencing
    convention, instead of per-panel titles.
  * Serif font (matches Computer Modern / most journal templates), consistent
    sizing, no constrained_layout surprises (explicit margins).
  * Every figure saved as BOTH .pdf (vector, for LaTeX \\includegraphics) and
    .png (raster, for quick viewing) at 300 dpi.
  * A fixed, colorblind-safe categorical palette used consistently across
    every figure for the same image (I_nom/I_p/I_ideal/I_const/I_Q).

Usage:
    python3 plot_results.py                 # reads out/, writes out/figs/
    python3 plot_results.py --outdir myrun
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import (CU_ROI_R, CU_ROI_ZHALF, PB_CX, PB_CY, PB_ROI_R,
                    PB_ROI_ZHALF, VOX_HALF, N_VOX, OUT_DIR)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})

IMAGE_ORDER = ["I_nom", "I_p", "I_ideal", "I_const", "I_Q"]
COLORS = {  # colorblind-safe (Okabe-Ito), fixed per image across all figures
    "I_nom": "#0072B2", "I_p": "#009E73", "I_ideal": "#D55E00",
    "I_const": "#CC79A7", "I_Q": "#E69F00",
}
PANEL_LABELS = "abcdefghij"


def _save(fig, figdir, name):
    fig.savefig(os.path.join(figdir, f"{name}.pdf"))
    fig.savefig(os.path.join(figdir, f"{name}.png"))
    plt.close(fig)


def _panel_label(ax, letter):
    ax.text(-0.02, 1.05, f"({letter})", transform=ax.transAxes,
           fontsize=11, fontweight="bold", va="bottom", ha="right")


def _add_roi_outlines(ax):
    ax.add_patch(Circle((PB_CX, PB_CY), PB_ROI_R, fill=False,
                        edgecolor="white", linewidth=1.0, linestyle="--"))
    ax.add_patch(Circle((0, 0), CU_ROI_R, fill=False,
                        edgecolor="white", linewidth=0.8, linestyle=":"))


def _central_z_index(centers):
    return int(np.argmin(np.abs(centers)))


def _load(outdir):
    imgs = np.load(os.path.join(outdir, "results_images.npz"))
    metrics = pd.read_csv(os.path.join(outdir, "results_metrics.csv"))
    artifact = pd.read_csv(os.path.join(outdir, "results_artifact.csv"),
                           index_col=0).squeeze("columns")
    speckle_path = os.path.join(outdir, "results_speckle.csv")
    speckle = pd.read_csv(speckle_path) if os.path.exists(speckle_path) else None
    return imgs, metrics, artifact, speckle


# ---------------------------------------------------------------- figures
def plot_slices(imgs, figdir, min_count=20):
    centers = imgs["centers"]
    iz = _central_z_index(centers)
    names = [n for n in IMAGE_ORDER if n in imgs.files]
    counts_sl = imgs["counts"][:, :, iz]
    mask = counts_sl >= min_count
    slabs = {n: np.where(mask, imgs[n][:, :, iz], np.nan) for n in names}
    vmax = np.nanpercentile(np.stack(list(slabs.values())), 99)

    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4))
    fig.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.16, wspace=0.15)
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        im = ax.imshow(slabs[name].T, origin="lower",
                       extent=[-VOX_HALF, VOX_HALF] * 2,
                       vmin=0, vmax=vmax, cmap="viridis")
        _add_roi_outlines(ax)
        ax.set_xlabel(r"$x$ (cm)")
        if i == 0:
            ax.set_ylabel(r"$y$ (cm)")
        else:
            ax.set_yticklabels([])
        _panel_label(ax, PANEL_LABELS[i])
        ax.set_title(name.replace("_", r"$_{\mathrm{") + "}}$" if "_" in name else name,
                    fontsize=9)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.72])
    fig.colorbar(im, cax=cax, label="mean weight")
    _save(fig, figdir, "slices_all")
    print("\n[slices_all] suggested caption:\n"
         f"  Reconstructed central slice ($z\\approx{centers[iz]:.2f}$ cm) for each "
         "weighting scheme, masked to voxels with counts $\\geq 20$. Dashed/dotted "
         "circles mark the Pb and Cu ROIs, respectively.")


def plot_metrics_bars(metrics, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.45)
    panels = [("SNR_Pb", "SNR_Pb_err", r"SNR$_{\mathrm{Pb}}$"),
             ("CNR", "CNR_err", "CNR"),
             ("sigma_PSF", "sigma_PSF_err", r"$\sigma_{\mathrm{PSF}}$ (cm)")]
    for i, (ax, (col, err, ylab)) in enumerate(zip(axes, panels)):
        colors = [COLORS.get(im, "gray") for im in metrics["image"]]
        ax.bar(metrics["image"], metrics[col], yerr=metrics[err],
              capsize=3, color=colors, linewidth=0)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=30)
        _panel_label(ax, PANEL_LABELS[i])
    _save(fig, figdir, "metrics_bars")
    print("\n[metrics_bars] suggested caption:\n"
         "  Image-quality metrics for each weighting scheme, with bootstrap "
         "$1\\sigma$ error bars over the respective ROI.")


def plot_artifact_map(imgs, figdir):
    centers = imgs["centers"]
    iz = _central_z_index(centers)
    counts = imgs["counts"][:, :, iz]
    A = imgs["I_nom"][:, :, iz] - imgs["I_ideal"][:, :, iz]
    A = np.where(counts >= 20, A, np.nan)
    vmax = np.nanpercentile(np.abs(A), 99) or 1.0

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.15)
    im = ax.imshow(A.T, origin="lower", extent=[-VOX_HALF, VOX_HALF] * 2,
                   vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    _add_roi_outlines(ax)
    ax.set_xlabel(r"$x$ (cm)"); ax.set_ylabel(r"$y$ (cm)")
    fig.colorbar(im, ax=ax, label=r"$I_{\mathrm{nom}}-I_{\mathrm{ideal}}$", shrink=0.85)
    _save(fig, figdir, "artifact_map")
    print("\n[artifact_map] suggested caption:\n"
         f"  Artifact map $I_{{\\mathrm{{nom}}}}-I_{{\\mathrm{{ideal}}}}$ "
         f"($z\\approx{centers[iz]:.2f}$ cm), masked to counts $\\geq 20$. "
         "Dashed/dotted circles mark the Pb and Cu ROIs.")


def plot_illumination(imgs, figdir):
    centers = imgs["centers"]
    iz = _central_z_index(centers)
    counts = imgs["counts"][:, :, iz]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.15)
    im = ax.imshow(np.log10(counts + 1.0).T, origin="lower",
                   extent=[-VOX_HALF, VOX_HALF] * 2, cmap="magma")
    _add_roi_outlines(ax)
    ax.set_xlabel(r"$x$ (cm)"); ax.set_ylabel(r"$y$ (cm)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}(\mathrm{counts}+1)$", shrink=0.85)
    _save(fig, figdir, "illumination")
    print("\n[illumination] suggested caption:\n"
         f"  Voxel occupancy ($z\\approx{centers[iz]:.2f}$ cm) confirming raster "
         "coverage of the target face. Dashed/dotted circles mark the Pb and Cu ROIs.")


def plot_artifact_summary(artifact, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.22, wspace=0.4)

    ax = axes[0]
    labels = [r"artifact RMS" "\n" r"($I_{\mathrm{nom}}-I_{\mathrm{ideal}}$)",
             r"residual RMS" "\n" r"($I_{p}-I_{\mathrm{ideal}}$)"]
    vals = [artifact.get("artifact_rms", np.nan), artifact.get("residual_rms", np.nan)]
    bars = ax.bar(labels, vals, color=["#D55E00", "#009E73"], linewidth=0)
    ax.set_ylabel("RMS (weight units)")
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v),
                       ha="center", va="bottom", fontsize=8)
    _panel_label(ax, "a")

    ax = axes[1]
    orth = artifact.get("orth_fraction", np.nan) * 100
    reduction = artifact.get("reduction", np.nan) * 100
    labels2 = ["orthogonal\nfraction", "per-momentum\nreduction"]
    bars2 = ax.bar(labels2, [orth, reduction], color=["#D55E00", "#009E73"], linewidth=0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    for b, v in zip(bars2, [orth, reduction]):
        if np.isfinite(v):
            ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                       ha="center", va="bottom", fontsize=8)
    _panel_label(ax, "b")
    _save(fig, figdir, "artifact_summary")
    print("\n[artifact_summary] suggested caption:\n"
         "  (a) RMS magnitude of the artifact map before and after the "
         "per-momentum correction. (b) Fraction of the artifact orthogonal to "
         "a uniform rescale (i.e., not explained by the $I_{\\mathrm{const}}$ "
         "null control), and the RMS reduction achieved by the per-momentum "
         "correction alone.")


def plot_speckle(speckle, figdir):
    if speckle is None:
        print("\n[speckle] results_speckle.csv not found -- skipping "
             "(re-run results_pipeline.py to generate it)")
        return
    row_nom = speckle[speckle.image == "I_nom"].iloc[0]
    row_Q = speckle[speckle.image == "I_Q"].iloc[0]
    row_gap = speckle[speckle.image == "gap"].iloc[0]
    sig = abs(row_gap.cv) / row_gap.cv_err if row_gap.cv_err else np.nan

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.15)
    xs = [0, 1]
    vals = [row_nom.cv, row_Q.cv]
    errs = [row_nom.cv_err, row_Q.cv_err]
    colors = [COLORS["I_nom"], COLORS["I_Q"]]
    ax.bar(xs, vals, yerr=errs, capsize=3, color=colors, linewidth=0, width=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([r"$I_{\mathrm{nom}}$", r"$I_{Q}$"])
    ax.set_ylabel("coefficient of variation")
    _save(fig, figdir, "speckle")
    print("\n[speckle] suggested caption:\n"
         "  Voxel-to-voxel coefficient of variation in a flat-truth (pure-Al) "
         "annulus, comparing the nominal and acceptance-matched estimators. "
         f"Gap $= {row_gap.cv:.4f} \\pm {row_gap.cv_err:.4f}$ "
         f"(${sig:.1f}\\sigma$)"
         f"{', consistent with zero at this statistic' if sig < 2 else ''}.")


def plot_psf_profile(imgs, figdir, image_name="I_nom", z_band=5.0,
                     oversample=3, min_bin_count=5):
    from scipy.optimize import curve_fit
    from scipy.special import erfc
    from config import CU_HALF, VOX_SIZE

    centers = imgs["centers"]
    img = imgs[image_name]
    counts = imgs["counts"]
    X, Y, Z = np.meshgrid(centers, centers, centers, indexing="ij")
    r = np.hypot(X - PB_CX, Y - PB_CY)
    band = np.abs(Z) <= z_band
    r_max = CU_HALF - np.hypot(PB_CX, PB_CY)

    bin_w = VOX_SIZE / oversample
    edges = np.arange(0.0, r_max + 1e-9, bin_w)
    rc, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & band & (counts > 0)
        if m.sum() < min_bin_count:
            continue
        rc.append(0.5 * (lo + hi)); val.append(img[m].mean())
    rc, val = np.asarray(rc), np.asarray(val)
    if rc.size < 6:
        print(f"[psf_profile_{image_name}] too few radial bins ({rc.size}) -- skipping")
        return

    def _erf_edge(x, A, C, x0, sigma):
        return C + 0.5 * A * erfc((x - x0) / (np.sqrt(2.0) * sigma))

    inner = val[rc < PB_ROI_R].mean() if (rc < PB_ROI_R).any() else val[0]
    outer = val[rc > PB_ROI_R].mean() if (rc > PB_ROI_R).any() else val[-1]
    p0 = [inner - outer, outer, PB_ROI_R, 0.5]
    bounds = ([-np.inf, -np.inf, 0.5 * PB_ROI_R, 0.05],
              [np.inf, np.inf, 1.5 * PB_ROI_R, 0.5 * r_max])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.92, bottom=0.17)
    ax.plot(rc, val, "o", ms=3.5, color=COLORS.get(image_name, "black"),
           label="radial profile")
    sigma_txt = ""
    try:
        popt, _ = curve_fit(_erf_edge, rc, val, p0=p0, bounds=bounds, maxfev=20000)
        rr = np.linspace(rc.min(), rc.max(), 200)
        ax.plot(rr, _erf_edge(rr, *popt), "-", color="black", linewidth=1.2,
               label="erf fit")
        sigma_txt = f"$\\sigma={abs(popt[3]):.3f}$ cm"
    except Exception as exc:
        sigma_txt = f"fit failed ({type(exc).__name__})"
    ax.axvline(PB_ROI_R, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("radius from Pb axis (cm)")
    ax.set_ylabel("mean weight")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, figdir, f"psf_profile_{image_name}")
    print(f"\n[psf_profile_{image_name}] suggested caption:\n"
         f"  Azimuthally averaged radial profile across the Pb--Cu boundary "
         f"for {image_name.replace('_', r'$_{')}"
         f"{'}' if '_' in image_name else ''}$, with an erf fit; {sigma_txt}. "
         "Dashed line marks the Pb ROI radius.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT_DIR)
    a = ap.parse_args()
    figdir = os.path.join(a.outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    imgs, metrics, artifact, speckle = _load(a.outdir)
    print(metrics.to_string(index=False))
    print(artifact.to_string())
    if speckle is not None:
        print(speckle.to_string(index=False))

    plot_slices(imgs, figdir)
    plot_metrics_bars(metrics, figdir)
    plot_artifact_map(imgs, figdir)
    plot_illumination(imgs, figdir)
    plot_artifact_summary(artifact, figdir)
    plot_speckle(speckle, figdir)
    plot_psf_profile(imgs, figdir, "I_nom")
    plot_psf_profile(imgs, figdir, "I_ideal")

    print(f"\nfigures written to {figdir}/ (.pdf for LaTeX, .png for preview)")


if __name__ == "__main__":
    main()