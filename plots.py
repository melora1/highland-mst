"""Publication-focused plotting for the revised Highland-MST study.

All figures are written to one directory, by default ``out/figs``.

This script can plot one results directory or scan the whole ``out`` tree.
In batch mode it:
  * deduplicates identical ``images.npz``/``gradient_maps.npz`` inputs;
  * prefers ``*_analysis`` directories when identical copies exist;
  * uses fixed per-panel symmetric color scales across all selected image runs;
  * masks voxels below ``MIN_VOX_COUNT`` before plotting;
  * writes a three-panel gradient figure (observed, mechanism predictor, residual);
  * generates theory/composition/tail/truncation/energy-loss summary figures;
  * generates off-Cu, adaptive-retention, artifact, and paired-seed summaries.

Examples
--------
Generate every available figure under out/::

    python plots_all.py --root out --all

Plot one directory only::

    python plots_all.py --outdir out/production_analysis --kind images
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from config import MIN_VOX_COUNT, VOX_HALF
except Exception:
    MIN_VOX_COUNT = 20
    VOX_HALF = 15.0


# ---------------------------------------------------------------------------
# Style

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    }
)


# ---------------------------------------------------------------------------
# Paths / saving / deduplication


def _find_out_root(path: Path) -> Path:
    path = path.resolve()
    for p in (path, *path.parents):
        if p.name == "out":
            return p
    return path if path.is_dir() else path.parent


def _figdir(source: Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        root = Path(explicit)
    else:
        root = _find_out_root(source) / "figs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tag(source: Path) -> str:
    name = source.resolve().name
    if name.endswith("_analysis"):
        name = name[:-9]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or "run"


def _save(fig, source: Path, stem: str, figdir: Path):
    tag = _tag(source)
    base = figdir / f"{tag}_{stem}"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _preference_score(directory: Path) -> tuple[int, str]:
    name = directory.name
    score = 0
    if name.endswith("_analysis"):
        score += 20
    if re.search(r"_n\d+", name):
        score += 10
    if name.startswith("production"):
        score += 5
    return score, name


def _dedupe_dirs(dirs: list[Path], filename: str) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    for d in dirs:
        f = d / filename
        if not f.exists():
            continue
        groups.setdefault(_sha256(f), []).append(d)
    selected = []
    for group in groups.values():
        selected.append(max(group, key=_preference_score))
    return sorted(selected, key=lambda p: p.name)


def _scan_dirs(root: Path, filename: str) -> list[Path]:
    # Results are expected one level below out; recursive scan also tolerates
    # additional organization without treating out/figs as input.
    candidates = []
    for f in root.rglob(filename):
        if "figs" in f.parts:
            continue
        candidates.append(f.parent)
    return _dedupe_dirs(candidates, filename)


# ---------------------------------------------------------------------------
# Shared image helpers


def _central(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D array, got shape {arr.shape}")
    return arr[:, :, arr.shape[2] // 2]


def _masked_central(z, name: str, min_count: int = MIN_VOX_COUNT) -> np.ndarray:
    a = _central(z[name]).astype(float, copy=True)
    if "counts" in z.files:
        c = _central(z["counts"])
        a[c < min_count] = np.nan
    return a


def _finite_abs(a: np.ndarray) -> np.ndarray:
    x = np.abs(a[np.isfinite(a)])
    return x[x >= 0]


def _robust_scale(values: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = [_finite_abs(v) for v in values]
    finite = [x for x in finite if x.size]
    if not finite:
        return 1.0
    pooled = np.concatenate(finite)
    v = float(np.nanpercentile(pooled, percentile))
    return max(v, np.finfo(float).eps)


def image_scales(dirs: list[Path], percentile: float = 99.0) -> tuple[float, float]:
    nom, pres = [], []
    for d in dirs:
        with np.load(d / "images.npz") as z:
            if not {"I_nom", "I_p", "I_Q"}.issubset(z.files):
                continue
            n = _masked_central(z, "I_nom") - _masked_central(z, "I_Q")
            p = _masked_central(z, "I_p") - _masked_central(z, "I_Q")
            nom.append(n)
            pres.append(p)
    return _robust_scale(nom, percentile), _robust_scale(pres, percentile)


# ---------------------------------------------------------------------------
# Theory figures


def plot_theory(outdir: str | Path, figdir: Path | None = None):
    out = Path(outdir)
    figdir = _figdir(out) if figdir is None else figdir
    d = pd.read_csv(out / "theory_collapse.csv")

    from analysis import PATHS
    from physics import reduced_parameters, mu2_eta

    rp = reduced_parameters(PATHS["AlCu"], 1.0)
    eta = np.geomspace(2.0, 30.0, 200)
    eps = np.array(
        [
            math.sqrt(rp["R"] * rp["B"] * mu2_eta(float(e), rp["B"], 2)) - 1.0
            for e in eta
        ]
    )

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.plot(eta, 100 * eps, lw=1.6, label="Al+Cu reduced curve")
    ax.plot(d.eta_cut, 100 * d.epsilon, "o", ms=5, label="momentum settings")
    ax.set_xscale("log")
    ticks = [2, 3, 5, 10, 20, 30]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks])
    ax.set_xlim(1.8, 33)
    ax.set_xlabel(r"$\eta_{\rm cut}=\theta_{\rm cut}/(\chi_c\sqrt{B})$")
    ax.set_ylabel(r"$\epsilon_M$ (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out, "collapse_eta", figdir)

    e = pd.read_csv(out / "eta1_protocol.csv")
    deep = e[e.window_role == "eta1_asymptote"].copy()
    deepest_hi = deep.eta_max.max()
    deepest_lo = deep.loc[deep.eta_max == deepest_hi, "eta_min"].max()
    deep = deep[(deep.eta_min == deepest_lo) & (deep.eta_max == deepest_hi)]
    diag = e[(e.window_role == "slope_diagnostic") & (e.nmax == 2)].copy()
    diag["eta_center"] = np.sqrt(diag.eta_min * diag.eta_max)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))

    paths = list(dict.fromkeys(deep.path.tolist()))
    xpos = np.arange(len(paths), dtype=float)
    for nmax, offset, marker in ((1, -0.045, "o"), (2, 0.045, "s")):
        g = deep[deep.nmax == nmax].set_index("path").reindex(paths)
        axes[0].plot(
            xpos + offset,
            g.eta1_joint.to_numpy(float),
            marker=marker,
            lw=1.1,
            label=rf"$n\leq {nmax}$",
        )
    axes[0].axhline(1.0, lw=0.8, color="k")
    axes[0].set_xticks(xpos, paths)
    axes[0].set_ylabel(r"asymptotic $\eta_1$")
    axes[0].set_xlabel("path")
    axes[0].legend(frameon=False)

    markers = ["o", "s", "^", "D", "v", "P"]
    for marker, (path, g) in zip(markers, diag.groupby("path")):
        g = g.sort_values("eta_center")
        x = g.eta_center.to_numpy(float)
        lo = g.eta_min.to_numpy(float)
        hi = g.eta_max.to_numpy(float)
        y = 100.0 * (g.slope_ratio.to_numpy(float) - 1.0)
        axes[1].errorbar(
            x,
            y,
            xerr=np.vstack([x - lo, hi - x]),
            marker=marker,
            capsize=2,
            lw=1.0,
            label=path,
        )
    eta0 = float(e.eta_table_max.iloc[0])
    axes[1].axvline(eta0, lw=0.9, ls="--", color="k", label=rf"$\eta_0={eta0:g}$")
    axes[1].axhline(0.0, lw=0.8, color="k")
    axes[1].set_xscale("log")
    axes[1].set_xlim(7.0, eta0 * 1.08)
    axes[1].set_ylabel(r"$(m/2R-1)$ (%)")
    axes[1].set_xlabel(r"fit-window $\eta$ (center and span)")
    axes[1].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    _save(fig, out, "eta1_paths", figdir)

    p = out / "composition_matched.csv"
    if p.exists():
        comp = pd.read_csv(p)
        for match, xlabel, stem in (
            ("k", r"matched $k=\theta_{\rm cut}/\theta_0$", "composition_matched_k"),
            ("eta", r"matched $\eta_{\rm cut}$", "composition_matched_eta"),
        ):
            q = comp[comp["match"] == match]
            fig, ax = plt.subplots(figsize=(4.6, 3.4))
            for path, g in q.groupby("path"):
                g = g.sort_values("value")
                ax.plot(g.value, 100.0 * g.epsilon, "o-", lw=1.2, ms=4, label=path)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\epsilon_M$ (%)")
            ax.legend(frameon=False)
            fig.tight_layout()
            _save(fig, out, stem, figdir)

    p = out / "tail_check.csv"
    if p.exists():
        tail = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(4.3, 3.2))
        ax.plot(tail.theta_mrad, tail.ratio, "o-", lw=1.2)
        ax.axhline(1.0, ls="--", lw=0.9, color="k")
        ax.set_xlabel(r"$\Theta$ (mrad)")
        ax.set_ylabel(r"$h_M(\Theta)\Theta^3/(2\chi_c^2)$")
        fig.tight_layout()
        _save(fig, out, "tail_check", figdir)

    p = out / "truncation_convergence.csv"
    if p.exists():
        conv = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        ax.axhline(0.0, lw=0.8, color="k")
        ax.plot(conv.p, 100.0 * conv.shift_abs, "o-", lw=1.2)
        ax.set_xlabel(r"$p$ (GeV/$c$)")
        ax.set_ylabel(r"$\epsilon_M^{(n\leq2)}-\epsilon_M^{(n\leq1)}$ (pp)")
        fig.tight_layout()
        _save(fig, out, "truncation_sensitivity", figdir)

    p = out / "energy_loss_calibration.csv"
    if p.exists():
        cal = pd.read_csv(p)
        alcu = cal[cal.path == "AlCu"].sort_values("p")
        if not alcu.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
            axes[0].plot(alcu.p, 100.0 * alcu.dp_over_p, "o-")
            axes[0].set_xlabel(r"$p_{\rm in}$ (GeV/$c$)")
            axes[0].set_ylabel(r"$\Delta p/p$ (%)")
            axes[1].plot(
                alcu.p, 100.0 * alcu.epsilon_matched_dchi, "o-", label="matched"
            )
            axes[1].plot(
                alcu.p, 100.0 * alcu.epsilon_mixed_dchi, "s-", label="upstream-tagged"
            )
            axes[1].set_xlabel(r"$p_{\rm in}$ (GeV/$c$)")
            axes[1].set_ylabel(r"$\epsilon$ (%)")
            axes[1].legend(frameon=False)
            fig.tight_layout()
            _save(fig, out, "energy_loss_effect", figdir)


# ---------------------------------------------------------------------------
# Detector-level difference maps and summaries


def plot_images(
    outdir: str | Path,
    figdir: Path | None = None,
    nominal_vmax: float | None = None,
    p_vmax: float | None = None,
    percentile: float = 99.0,
):
    out = Path(outdir)
    figdir = _figdir(out) if figdir is None else figdir
    with np.load(out / "images.npz") as z:
        required = {"I_nom", "I_p", "I_Q"}
        if not required.issubset(z.files):
            return
        In = _masked_central(z, "I_nom")
        Ip = _masked_central(z, "I_p")
        IQ = _masked_central(z, "I_Q")

    maps = [In - IQ, Ip - IQ]
    labels = [r"$I_{\rm nom}-I_Q$", r"$I_p-I_Q$"]
    scales = [
        nominal_vmax or _robust_scale([maps[0]], percentile),
        p_vmax or _robust_scale([maps[1]], percentile),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25))
    for ax, m, label, vmax in zip(axes, maps, labels, scales):
        im = ax.imshow(
            m.T,
            origin="lower",
            extent=[-VOX_HALF, VOX_HALF, -VOX_HALF, VOX_HALF],
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu_r",
            interpolation="nearest",
        )
        ax.set_xlabel(r"$x$ (cm)")
        ax.set_title(label)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("weight difference")
    axes[0].set_ylabel(r"$y$ (cm)")
    fig.tight_layout()
    _save(fig, out, "difference_maps", figdir)

    p = out / "path_residuals.csv"
    if p.exists():
        d = pd.read_csv(p)
        if not d.empty:
            fig, ax = plt.subplots(figsize=(4.2, 3.1))
            bars = ax.bar(d.region, d.image_rms)
            ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
            ax.set_ylabel(r"image RMS of $I_p-I_Q$")
            ax.set_xlabel("reference-path class")
            fig.tight_layout()
            _save(fig, out, "offcu_path_residual", figdir)

    p = out / "adaptive_retention.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.denominator == "all generated"].copy()
        if not d.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.1), sharey=True)
            for ax, (classification, g) in zip(axes, d.groupby("classification")):
                bars = ax.bar(g.group, 100.0 * g.retention)
                ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
                ax.set_ylim(0, 100)
                ax.set_title(classification)
                ax.tick_params(axis="x", rotation=15)
            axes[0].set_ylabel("adaptive-cut retention (%)")
            fig.tight_layout()
            _save(fig, out, "adaptive_retention", figdir)

    p = out / "artifact_summary.csv"
    if p.exists():
        a = pd.read_csv(p).iloc[0]
        fig, ax = plt.subplots(figsize=(4.3, 3.1))
        vals = [a.artifact_rms, a.p_residual_rms]
        bars = ax.bar([r"$I_{\rm nom}-I_Q$", r"$I_p-I_Q$"], vals)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_ylabel("image RMS difference")
        ax.text(
            0.98,
            0.96,
            f"reduction = {100.0 * a.p_reduction:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        fig.tight_layout()
        _save(fig, out, "artifact_summary", figdir)


# ---------------------------------------------------------------------------
# Gradient figure


def plot_gradient(
    outdir: str | Path, figdir: Path | None = None, percentile: float = 99.0
):
    out = Path(outdir)
    figdir = _figdir(out) if figdir is None else figdir
    with np.load(out / "gradient_maps.npz") as z:
        required = {"observed", "predicted_unweighted", "residual_unweighted"}
        if not required.issubset(z.files):
            return
        vals = {name: _masked_central(z, name) for name in required}

    observed = vals["observed"]
    predicted = vals["predicted_unweighted"]
    residual = vals["residual_unweighted"]

    # Observed and residual share a scale so the residual fraction is visually
    # meaningful.  The normalization-field predictor gets its own scale.
    obs_scale = _robust_scale([observed, residual], percentile)
    pred_scale = _robust_scale([predicted], percentile)

    names = [observed, predicted, residual]
    labels = [
        r"observed $I_{\rm nom}-I_Q$",
        "normalization-field predictor",
        "one-amplitude residual",
    ]
    scales = [obs_scale, pred_scale, obs_scale]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))
    for ax, v, label, vmax in zip(axes, names, labels, scales):
        im = ax.imshow(
            v.T,
            origin="lower",
            extent=[-VOX_HALF, VOX_HALF, -VOX_HALF, VOX_HALF],
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu_r",
            interpolation="nearest",
        )
        ax.set_xlabel(r"$x$ (cm)")
        ax.set_title(label, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(r"$y$ (cm)")

    p = out / "gradient_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        r = s[s.predictor == "normalization_field"]
        if not r.empty:
            row = r.iloc[0]
            axes[2].text(
                0.03,
                0.97,
                f"r = {row.correlation:.3f}\nresidual = {100.0 * row.residual_fraction:.1f}%",
                transform=axes[2].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

    fig.tight_layout()
    _save(fig, out, "gradient_causal_maps", figdir)


# ---------------------------------------------------------------------------
# Paired-seed summary


def plot_paired(csv_path: Path, figdir: Path | None = None):
    if not csv_path.exists():
        return
    d = pd.read_csv(csv_path)
    if d.empty:
        return
    source = csv_path.parent / "paired"
    figdir = _figdir(csv_path.parent) if figdir is None else figdir
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    x = np.arange(len(d))
    axes[0].errorbar(x, d.dSNR_mean, yerr=d.dSNR_sd, fmt="o", capsize=3)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xticks(x, d.comparison, rotation=20, ha="right")
    axes[0].set_ylabel(r"paired $\Delta$SNR")
    axes[1].errorbar(x, d.dCNR_mean, yerr=d.dCNR_sd, fmt="o", capsize=3)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xticks(x, d.comparison, rotation=20, ha="right")
    axes[1].set_ylabel(r"paired $\Delta$CNR")
    fig.tight_layout()
    _save(fig, source, "seed_summary", figdir)


# ---------------------------------------------------------------------------
# CLI


def run_all(
    root: Path, explicit_figdir: str | Path | None = None, percentile: float = 99.0
):
    root = root.resolve()
    figdir = _figdir(root, explicit_figdir)

    theory_dirs = _scan_dirs(root, "theory_collapse.csv")
    image_dirs = _scan_dirs(root, "images.npz")
    gradient_dirs = _scan_dirs(root, "gradient_maps.npz")

    nominal_vmax, p_vmax = (
        image_scales(image_dirs, percentile=percentile) if image_dirs else (None, None)
    )

    for d in theory_dirs:
        plot_theory(d, figdir)
    for d in image_dirs:
        plot_images(
            d, figdir, nominal_vmax=nominal_vmax, p_vmax=p_vmax, percentile=percentile
        )
    for d in gradient_dirs:
        plot_gradient(d, figdir, percentile=percentile)

    paired = root / "paired_seed_summary.csv"
    if paired.exists():
        plot_paired(paired, figdir)

    print(f"figures written to {figdir}")
    print(f"theory sources:   {len(theory_dirs)}")
    print(f"image sources:    {len(image_dirs)}")
    print(f"gradient sources: {len(gradient_dirs)}")
    if image_dirs:
        print(f"common nominal-map |scale|: {nominal_vmax:.6g}")
        print(f"common p-residual |scale|:  {p_vmax:.6g}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate centralized publication figures."
    )
    ap.add_argument("--root", default="out", help="results root; default: out")
    ap.add_argument(
        "--figdir", default=None, help="override figure directory; default: <root>/figs"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="scan the entire results tree and plot all unique runs",
    )
    ap.add_argument("--outdir", default=None, help="plot one results directory")
    ap.add_argument(
        "--kind", choices=["theory", "images", "gradient", "all"], default="all"
    )
    ap.add_argument(
        "--percentile", type=float, default=99.0, help="robust display scale percentile"
    )
    a = ap.parse_args()

    if a.all:
        run_all(Path(a.root), a.figdir, percentile=a.percentile)
        return

    if a.outdir is None:
        raise SystemExit("Use --all or provide --outdir PATH")

    out = Path(a.outdir)
    figdir = _figdir(out, a.figdir)
    if a.kind in ("theory", "all") and (out / "theory_collapse.csv").exists():
        plot_theory(out, figdir)
    if a.kind in ("images", "all") and (out / "images.npz").exists():
        plot_images(out, figdir, percentile=a.percentile)
    if a.kind in ("gradient", "all") and (out / "gradient_maps.npz").exists():
        plot_gradient(out, figdir, percentile=a.percentile)


if __name__ == "__main__":
    main()
