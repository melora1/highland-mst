#!/usr/bin/env python3
"""
plot_results.py  --  visualize results_pipeline.py's outputs.

Reads out/results_images.npz, out/results_metrics.csv, out/results_artifact.csv
(written by results_pipeline.py) and produces a set of PNGs in out/figs/:

  slices_<image>.png    central-z voxel slice for each of I_nom/I_p/I_ideal/I_const
  metrics_bars.png      SNR_Pb, CNR, DP grouped bar chart across the four images
  artifact_map.png      I_nom - I_ideal central slice, with the Pb/Cu ROI outlines
  illumination.png       counts central slice (log scale) -- coverage check
  psf_profile.png        radial profile + erf fit at the Pb-Cu boundary (I_nom)

Usage:
    python3 plot_results.py                 # reads out/, writes out/figs/
    python3 plot_results.py --outdir myrun   # different results dir
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


def _central_z_index(centers):
    return int(np.argmin(np.abs(centers)))


def _load(outdir):
    imgs = np.load(os.path.join(outdir, "results_images.npz"))
    metrics = pd.read_csv(os.path.join(outdir, "results_metrics.csv"))
    artifact = pd.read_csv(os.path.join(outdir, "results_artifact.csv"),
                           index_col=0).squeeze("columns")
    return imgs, metrics, artifact


def plot_slices(imgs, figdir, min_count=20):
    centers = imgs["centers"]
    iz = _central_z_index(centers)
    names = ["I_nom", "I_p", "I_ideal", "I_const"]
    counts_sl = imgs["counts"][:, :, iz]
    mask = counts_sl >= min_count
    slabs = {n: np.where(mask, imgs[n][:, :, iz], np.nan) for n in names}
    vmax = np.nanpercentile(np.stack(list(slabs.values())), 99)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    for ax, name in zip(axes, names):
        im = ax.imshow(slabs[name].T, origin="lower",
                       extent=[-VOX_HALF, VOX_HALF] * 2,
                       vmin=0, vmax=vmax, cmap="viridis")
        _add_roi_outlines(ax)
        ax.set_title(name)
        ax.set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean weight (accumulated)")
    fig.suptitle(f"Central z-slice (z ~ {centers[iz]:.2f} cm), "
                f"masked to counts >= {min_count}")
    fig.savefig(os.path.join(figdir, "slices_all.png"), dpi=150)
    plt.close(fig)


def _add_roi_outlines(ax):
    ax.add_patch(Circle((PB_CX, PB_CY), PB_ROI_R, fill=False,
                        edgecolor="white", linewidth=1.2, linestyle="--"))
    ax.add_patch(Circle((0, 0), CU_ROI_R, fill=False,
                        edgecolor="white", linewidth=1.0, linestyle=":"))


def plot_metrics_bars(metrics, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, col, err in [(axes[0], "SNR_Pb", "SNR_Pb_err"),
                         (axes[1], "CNR", "CNR_err"),
                         (axes[2], "sigma_PSF", "sigma_PSF_err")]:
        ax.bar(metrics["image"], metrics[col], yerr=metrics[err],
              capsize=4, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Image-quality metrics (bootstrap errors)")
    fig.savefig(os.path.join(figdir, "metrics_bars.png"), dpi=150)
    plt.close(fig)


def plot_artifact_map(imgs, figdir):
    centers = imgs["centers"]
    iz = _central_z_index(centers)
    counts = imgs["counts"][:, :, iz]
    A = imgs["I_nom"][:, :, iz] - imgs["I_ideal"][:, :, iz]
    A = np.where(counts >= 20, A, np.nan)
    vmax = np.nanpercentile(np.abs(A), 99) or 1.0

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im = ax.imshow(A.T, origin="lower", extent=[-VOX_HALF, VOX_HALF] * 2,
                   vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    _add_roi_outlines(ax)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"Artifact map I_nom - I_ideal (z ~ {centers[iz]:.2f} cm)\n"
                "dashed = Pb ROI, dotted = Cu ROI")
    fig.colorbar(im, ax=ax, label="weight difference")
    fig.savefig(os.path.join(figdir, "artifact_map.png"), dpi=150)
    plt.close(fig)


def plot_illumination(imgs, figdir):
    centers = imgs["centers"]
    iz = _central_z_index(centers)
    counts = imgs["counts"][:, :, iz]
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    disp = np.log10(counts + 1.0)
    im = ax.imshow(disp.T, origin="lower", extent=[-VOX_HALF, VOX_HALF] * 2,
                   cmap="magma")
    _add_roi_outlines(ax)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"log10(counts+1), z ~ {centers[iz]:.2f} cm  (coverage check)")
    fig.colorbar(im, ax=ax, label="log10(counts+1)")
    fig.savefig(os.path.join(figdir, "illumination.png"), dpi=150)
    plt.close(fig)


def plot_psf_profile(imgs, figdir, image_name="I_nom", z_band=5.0,
                     oversample=3, min_bin_count=5):
    """Reproduces the radial profile branch_b.edge_response fits, oversampled
    (bin width = VOX_SIZE/oversample) so the fit doesn't rest on the ~6 points
    a bare VOX_SIZE binning gives at this r_max. Each voxel still contributes
    to whichever radial shell its center falls in -- oversampling just uses
    narrower shells, relying on azimuthal averaging (many voxels per shell at
    a given radius) to keep per-bin statistics above min_bin_count."""
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
    rc, val, n_in_bin = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & band & (counts > 0)
        if m.sum() < min_bin_count:
            continue
        rc.append(0.5 * (lo + hi)); val.append(img[m].mean())
        n_in_bin.append(int(m.sum()))
    rc, val = np.asarray(rc), np.asarray(val)
    if rc.size < 6:
        print(f"psf_profile ({image_name}): too few radial bins ({rc.size}) "
             "to plot -- skipping")
        return

    def _erf_edge(x, A, C, x0, sigma):
        return C + 0.5 * A * erfc((x - x0) / (np.sqrt(2.0) * sigma))

    inner = val[rc < PB_ROI_R].mean() if (rc < PB_ROI_R).any() else val[0]
    outer = val[rc > PB_ROI_R].mean() if (rc > PB_ROI_R).any() else val[-1]
    p0 = [inner - outer, outer, PB_ROI_R, 0.5]
    bounds = ([-np.inf, -np.inf, 0.5 * PB_ROI_R, 0.05],
              [np.inf, np.inf, 1.5 * PB_ROI_R, 0.5 * r_max])
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    ax.plot(rc, val, "o", ms=4, label=f"radial profile ({rc.size} bins, "
                                      f"{bin_w:.2f} cm wide)")
    try:
        popt, _ = curve_fit(_erf_edge, rc, val, p0=p0, bounds=bounds, maxfev=20000)
        rr = np.linspace(rc.min(), rc.max(), 200)
        ax.plot(rr, _erf_edge(rr, *popt), "-", label=f"erf fit, sigma={abs(popt[3]):.3f} cm")
    except Exception as exc:
        ax.set_title(f"{image_name}: fit failed ({type(exc).__name__})")
    ax.axvline(PB_ROI_R, color="gray", linestyle="--", linewidth=1, label="Pb ROI radius")
    ax.set_xlabel("radius from Pb axis (cm)"); ax.set_ylabel("mean weight")
    ax.set_title(f"Edge profile -- {image_name}")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(figdir, f"psf_profile_{image_name}.png"), dpi=150)
    plt.close(fig)


def plot_artifact_summary(artifact, figdir):
    """Turns the results_artifact.csv numbers into a figure: RMS magnitudes
    (artifact vs. per-momentum residual) on one panel, and the two decisive
    fractions (orthogonal/genuine-structure, and per-momentum reduction) on
    a second panel, scaled 0-100%."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.30, wspace=0.35)

    ax = axes[0]
    labels = ["artifact RMS\n(I_nom - I_ideal)", "residual RMS\n(I_p - I_ideal)"]
    vals = [artifact.get("artifact_rms", np.nan), artifact.get("residual_rms", np.nan)]
    bars = ax.bar(labels, vals, color=["#c44e52", "#55a868"])
    ax.set_ylabel("RMS (weight units)")
    ax.set_title("Artifact magnitude")
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v),
                       ha="center", va="bottom")

    ax = axes[1]
    orth = artifact.get("orth_fraction", np.nan) * 100
    reduction = artifact.get("reduction", np.nan) * 100
    labels2 = ["orthogonal fraction\n(genuine structure)",
              "per-momentum\ncorrection reduction"]
    vals2 = [orth, reduction]
    colors2 = ["#c44e52" if np.isfinite(orth) and orth > 10 else "#8172b2",
              "#55a868"]
    bars2 = ax.bar(labels2, vals2, color=colors2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Decisive fractions (Sec. 2.3/4.4)")
    for b, v in zip(bars2, vals2):
        if np.isfinite(v):
            ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                       ha="center", va="bottom")

    note = ("orthogonal fraction: share of the artifact map NOT explained by a uniform rescale\n"
           "(I_const null control) -- >0 means genuine spatial structure survives the re-steered beamline.")
    fig.suptitle("Artifact decomposition", y=0.97)
    fig.text(0.5, 0.06, note, ha="center", va="top", fontsize=8, style="italic")
    fig.savefig(os.path.join(figdir, "artifact_summary.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT_DIR)
    a = ap.parse_args()
    figdir = os.path.join(a.outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    imgs, metrics, artifact = _load(a.outdir)
    print(metrics.to_string(index=False))
    print(artifact.to_string())

    plot_slices(imgs, figdir)
    plot_metrics_bars(metrics, figdir)
    plot_artifact_map(imgs, figdir)
    plot_illumination(imgs, figdir)
    plot_artifact_summary(artifact, figdir)
    plot_psf_profile(imgs, figdir, "I_nom")
    plot_psf_profile(imgs, figdir, "I_ideal")

    print(f"\nfigures written to {figdir}/")


if __name__ == "__main__":
    main()