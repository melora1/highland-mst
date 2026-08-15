#!/usr/bin/env python3
"""
plot_results.py  --  journal-ready figures from results_pipeline.py output.

Usage:
    python3 plot_results.py --outdir out/seed0_n500000

Reads from:  {outdir}/images.npz, metrics.csv, artifact.csv, speckle.csv
Writes to:   {outdir}/figs/  (.pdf for LaTeX, .png for preview)

Journal conventions:
  * No in-figure titles. Captions printed to stdout, ready to paste.
  * Panel labels (a)(b)(c)... only.
  * Serif font (Computer Modern), consistent sizing.
  * Saved as both .pdf (vector) and .png (300 dpi).
  * Fixed colorblind-safe palette (Okabe-Ito) per image name.
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
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "DejaVu Serif",
                          "Times New Roman"],
    "mathtext.fontset":  "cm",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.linewidth":    0.8,
})

IMAGE_ORDER = ["I_nom", "I_p", "I_ideal", "I_const", "I_Q"]
COLORS = {   # Okabe-Ito, fixed per image name across all figures
    "I_nom":   "#0072B2",
    "I_p":     "#009E73",
    "I_ideal": "#D55E00",
    "I_const": "#CC79A7",
    "I_Q":     "#E69F00",
}
PANEL_LABELS = "abcdefghij"


# ---------------------------------------------------------------- helpers

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
    """Load pipeline outputs from a run-specific directory."""
    imgs     = np.load(os.path.join(outdir, "images.npz"))
    metrics  = pd.read_csv(os.path.join(outdir, "metrics.csv"))
    artifact = pd.read_csv(os.path.join(outdir, "artifact.csv"),
                           index_col=0).squeeze("columns")
    sp_path  = os.path.join(outdir, "speckle.csv")
    speckle  = pd.read_csv(sp_path) if os.path.exists(sp_path) else None
    return imgs, metrics, artifact, speckle


# ---------------------------------------------------------------- figures

def plot_slices(imgs, figdir, min_count=20):
    centers = imgs["centers"]
    iz      = _central_z_index(centers)
    names   = [n for n in IMAGE_ORDER if n in imgs.files]
    counts_sl = imgs["counts"][:, :, iz]
    mask    = counts_sl >= min_count
    slabs   = {n: np.where(mask, imgs[n][:, :, iz], np.nan) for n in names}
    vmax    = np.nanpercentile(np.stack(list(slabs.values())), 99)

    fig, axes = plt.subplots(1, len(names),
                             figsize=(3.6 * len(names), 3.4))
    fig.subplots_adjust(left=0.05, right=0.90, top=0.88,
                        bottom=0.16, wspace=0.15)
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
        ax.set_title(
            name.replace("_", r"$_{\mathrm{") + "}}$"
            if "_" in name else name, fontsize=9)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.72])
    fig.colorbar(im, cax=cax, label="mean weight")
    _save(fig, figdir, "slices_all")
    print("\n[slices_all] suggested caption:\n"
          f"  Reconstructed central slice ($z\\approx{centers[iz]:.2f}$~cm) "
          "for each weighting scheme, masked to voxels with counts $\\geq20$. "
          "Dashed and dotted circles mark the Pb and Cu ROIs, respectively.")


def plot_metrics_bars(metrics, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88,
                        bottom=0.28, wspace=0.45)
    panels = [
        ("SNR_Pb",    "SNR_Pb_err",    r"SNR$_{\mathrm{Pb}}$"),
        ("CNR",       "CNR_err",       "CNR"),
        ("sigma_PSF", "sigma_PSF_err", r"$\sigma_{\mathrm{PSF}}$ (cm)"),
    ]
    for i, (ax, (col, err, ylab)) in enumerate(zip(axes, panels)):
        colors = [COLORS.get(im, "gray") for im in metrics["image"]]
        xpos = np.arange(len(metrics))
        ax.bar(xpos, metrics[col], yerr=metrics[err],
               capsize=3, color=colors, linewidth=0)
        ax.set_ylabel(ylab)
        # LaTeX tick labels
        labels = [
            r"$I_{\mathrm{nom}}$"   if t == "I_nom"        else
            r"$I_{p}$"              if t == "I_p"           else
            r"$I_{\mathrm{ideal}}$" if t == "I_ideal"       else
            r"$I_{\mathrm{const}}$" if t == "I_const"       else
            r"$I_{Q}$"              if t == "I_Q"           else
            r"$I_{Q,\mathrm{adap}}$"
            for t in metrics["image"]]
        ax.set_xticks(xpos, labels, rotation=30, ha="right")
        _panel_label(ax, PANEL_LABELS[i])
    _save(fig, figdir, "metrics_bars")
    print("\n[metrics_bars] suggested caption:\n"
          "  Image-quality metrics for each weighting scheme, with "
          "bootstrap $1\\sigma$ error bars over the respective ROI.")


def plot_artifact_map(imgs, figdir):
    centers = imgs["centers"]
    iz      = _central_z_index(centers)
    counts  = imgs["counts"][:, :, iz]
    A = imgs["I_nom"][:, :, iz] - imgs["I_Q"][:, :, iz]
    A = np.where(counts >= 20, A, np.nan)
    vmax = np.nanpercentile(np.abs(A), 99) or 1.0

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.15)
    im = ax.imshow(A.T, origin="lower",
                   extent=[-VOX_HALF, VOX_HALF] * 2,
                   vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    _add_roi_outlines(ax)
    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$y$ (cm)")
    fig.colorbar(im, ax=ax,
                 label=r"$I_{\mathrm{nom}}-I_{Q}$",
                 shrink=0.85)
    _save(fig, figdir, "artifact_map")
    print("\n[artifact_map] suggested caption:\n"
          f"  Artifact map $I_{{\\mathrm{{nom}}}}-I_Q$ "
          f"($z\\approx{centers[iz]:.2f}$~cm), masked to counts $\\geq20$. "
          "Dashed and dotted circles mark the Pb and Cu ROIs.")


def plot_illumination(imgs, figdir):
    centers = imgs["centers"]
    iz      = _central_z_index(centers)
    counts  = imgs["counts"][:, :, iz]

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.15)
    im = ax.imshow(np.log10(counts + 1.0).T, origin="lower",
                   extent=[-VOX_HALF, VOX_HALF] * 2, cmap="magma")
    _add_roi_outlines(ax)
    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$y$ (cm)")
    fig.colorbar(im, ax=ax,
                 label=r"$\log_{10}(\mathrm{counts}+1)$", shrink=0.85)
    _save(fig, figdir, "illumination")
    print("\n[illumination] suggested caption:\n"
          f"  Voxel occupancy ($z\\approx{centers[iz]:.2f}$~cm) confirming "
          "raster coverage of the target face. Dashed and dotted circles "
          "mark the Pb and Cu ROIs.")


def plot_artifact_summary(artifact, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88,
                        bottom=0.22, wspace=0.45)

    # (a) RMS magnitudes
    ax = axes[0]
    vals_a = [artifact["artifact_rms"], artifact["residual_rms"]]
    labels_a = [r"artifact RMS" "\n" r"($I_{\mathrm{nom}}-I_{Q}$)",
                r"residual RMS" "\n" r"($I_{p}-I_{Q}$)"]
    bars = ax.bar(labels_a, vals_a,
                  color=["#D55E00", "#009E73"], linewidth=0)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_ylabel("RMS (weight units)")
    _panel_label(ax, "a")

    # (b) fractions
    ax = axes[1]
    vals_b  = [artifact["orth_fraction"] * 100,
               artifact["reduction"]     * 100]
    labels_b = ["orthogonal\nfraction",
                "per-momentum\nreduction"]
    bars = ax.bar(labels_b, vals_b,
                  color=["#D55E00", "#009E73"], linewidth=0)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel(r"\%")
    _panel_label(ax, "b")

    _save(fig, figdir, "artifact_summary")
    print("\n[artifact_summary] suggested caption:\n"
          "  (a)~RMS magnitude of the artifact map before and after the "
          "per-momentum correction. "
          "(b)~Fraction of the artifact orthogonal to a uniform rescale "
          "(i.e., not explained by the $I_{\\mathrm{const}}$ null control), "
          "and the RMS reduction achieved by the per-momentum correction alone.")


def plot_speckle(speckle, figdir):
    if speckle is None:
        print("\n[speckle] speckle.csv not found -- skipping")
        return
    row_nom = speckle[speckle.image == "I_nom"].iloc[0]
    row_Q   = speckle[speckle.image == "I_Q"].iloc[0]
    row_gap = speckle[speckle.image == "gap"].iloc[0]
    sig     = (abs(row_gap.cv) / row_gap.cv_err
               if row_gap.cv_err else np.nan)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.15)
    xs    = [0, 1]
    vals  = [row_nom.cv, row_Q.cv]
    errs  = [row_nom.cv_err, row_Q.cv_err]
    colors = [COLORS["I_nom"], COLORS["I_Q"]]
    ax.bar(xs, vals, yerr=errs, capsize=3, color=colors,
           linewidth=0, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$I_{\mathrm{nom}}$", r"$I_{Q}$"])
    ax.set_ylabel("coefficient of variation")
    _save(fig, figdir, "speckle")
    print("\n[speckle] suggested caption:\n"
          "  Voxel-to-voxel coefficient of variation in a flat-truth "
          "(pure-Al) annulus, comparing the nominal and acceptance-matched "
          "estimators. "
          f"Gap $= {row_gap.cv:.4f} \\pm {row_gap.cv_err:.4f}$ "
          f"(${sig:.1f}\\sigma$)"
          f"{', consistent with zero at this statistic' if sig < 2 else ''}.")


def plot_psf_profile(imgs, figdir, image_name="I_nom",
                     z_band=5.0, oversample=1, min_bin_count=5):
    from scipy.optimize import curve_fit
    from scipy.special import erfc
    from config import CU_HALF, VOX_SIZE

    centers = imgs["centers"]
    img     = imgs[image_name]
    counts  = imgs["counts"]
    X, Y, Z = np.meshgrid(centers, centers, centers, indexing="ij")
    r       = np.hypot(X - PB_CX, Y - PB_CY)
    band    = np.abs(Z) <= z_band
    r_max   = CU_HALF - np.hypot(PB_CX, PB_CY)

    bin_w = VOX_SIZE / oversample
    edges = np.arange(0.0, r_max + 1e-9, bin_w)
    rc, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & band & (counts > 0)
        if m.sum() < min_bin_count:
            continue
        rc.append(0.5 * (lo + hi))
        val.append(img[m].mean())
    rc, val = np.asarray(rc), np.asarray(val)
    if rc.size < 6:
        print(f"[psf_profile_{image_name}] too few radial bins "
              f"({rc.size}) -- skipping")
        return

    def _erf_edge(x, A, C, x0, sigma):
        return C + 0.5 * A * erfc((x - x0) / (np.sqrt(2.0) * sigma))

    inner = val[rc < PB_ROI_R].mean() if (rc < PB_ROI_R).any() else val[0]
    outer = val[rc > PB_ROI_R].mean() if (rc > PB_ROI_R).any() else val[-1]
    p0     = [inner - outer, outer, PB_ROI_R, 0.5]
    bounds = ([-np.inf, -np.inf, 0.5 * PB_ROI_R, 0.05],
              [ np.inf,  np.inf, 1.5 * PB_ROI_R, 0.5 * r_max])

    color = COLORS.get(image_name, "black")
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.92, bottom=0.17)
    ax.plot(rc, val, "o", ms=3.5, color=color, label="radial profile")
    sigma_txt = ""
    try:
        popt, _ = curve_fit(_erf_edge, rc, val, p0=p0,
                            bounds=bounds, maxfev=20000)
        rr = np.linspace(rc.min(), rc.max(), 200)
        ax.plot(rr, _erf_edge(rr, *popt), "-", color="black",
                linewidth=1.2, label="erf fit")
        sigma_txt = f"$\\sigma={abs(popt[3]):.3f}$~cm"
    except Exception as exc:
        sigma_txt = f"fit failed ({type(exc).__name__})"
    ax.axvline(PB_ROI_R, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("radius from Pb axis (cm)")
    ax.set_ylabel("mean weight")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, figdir, f"psf_profile_{image_name}")

    # LaTeX-friendly name for caption
    latex_name = (r"$I_{\mathrm{" + image_name.replace("I_", "") + r"}}$"
                  if image_name.startswith("I_") else image_name)
    print(f"\n[psf_profile_{image_name}] suggested caption:\n"
          f"  Azimuthally averaged radial profile across the Pb--Cu boundary "
          f"for {latex_name}, with an erf fit; {sigma_txt}. "
          "Dashed line marks the Pb ROI radius.")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Generate figures from a results_pipeline.py run directory.")
    ap.add_argument("--outdir", default=None,
                    help="Run directory produced by results_pipeline.py, "
                         "e.g. out/seed0_n500000. "
                         "Figures are written to {outdir}/figs/.")
    a = ap.parse_args()

    if a.outdir is None:
        # try to find the most recent run subdirectory
        base = OUT_DIR
        candidates = [
            os.path.join(base, d) for d in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, d))
            and os.path.exists(os.path.join(base, d, "images.npz"))
        ] if os.path.isdir(base) else []
        if not candidates:
            raise SystemExit(
                "No run directory found. "
                "Run: python3 results_pipeline.py --n 500000 --seed 0\n"
                "Then: python3 plot_results.py --outdir out/seed0_n500000")
        outdir = candidates[-1]
        print(f"[plot_results] using most recent run: {outdir}")
    else:
        outdir = a.outdir

    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    imgs, metrics, artifact, speckle = _load(outdir)
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
    plot_psf_profile(imgs, figdir, "I_Q")

    print(f"\nfigures written to {figdir}/ (.pdf for LaTeX, .png for preview)")


if __name__ == "__main__":
    main()