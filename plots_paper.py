"""Publication-quality figures for the revised Highland-MST study.

All figures are written to one directory, by default ``out/figs``.

Recommended manuscript mode::

    python plots.py --root out --paper

This produces only the canonical theory, production, gradient, and paired-seed
figures.  ``--all`` remains available for convergence/seed diagnostics.
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
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib.patches import Circle, Rectangle, Patch

try:
    from config import (
        MIN_VOX_COUNT, VOX_HALF, CU_HALF,
        PB_CX, PB_CY, PB_ROI_R,
    )
except Exception:
    MIN_VOX_COUNT = 20
    VOX_HALF = 15.0
    CU_HALF = 7.5
    PB_CX = 3.0
    PB_CY = 2.0
    PB_ROI_R = 2.0


# ---------------------------------------------------------------------------
# Journal style and consistent semantic colors

plt.rcParams.update({
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
    "lines.linewidth": 1.35,
    "savefig.dpi": 300,
})

PATH_ORDER = ["Al25", "Cu15", "AlCu", "Pb15"]
PATH_LABEL = {
    "Al25": "Al 25 cm",
    "Cu15": "Cu 15 cm",
    "AlCu": "Al+Cu",
    "Pb15": "Pb 15 cm",
}
# Okabe-Ito-inspired, colorblind-safe mapping used everywhere.
PATH_COLOR = {
    "Al25": "#0072B2",
    "Cu15": "#009E73",
    "AlCu": "#E69F00",
    "Pb15": "#D55E00",
}
PATH_MARKER = {"Al25": "o", "Cu15": "^", "AlCu": "s", "Pb15": "D"}


def _panel(ax, letter: str):
    ax.text(-0.11, 1.04, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=10)


def _diverging_cmap():
    cmap = plt.get_cmap("RdBu_r").copy()
    # Missing / under-populated voxels must not be confused with zero difference.
    cmap.set_bad("#e8e8e8")
    return cmap


# ---------------------------------------------------------------------------
# Paths / saving / deduplication


def _find_out_root(path: Path) -> Path:
    path = path.resolve()
    for p in (path, *path.parents):
        if p.name == "out":
            return p
    return path if path.is_dir() else path.parent


def _figdir(source: Path, explicit: str | Path | None = None) -> Path:
    root = Path(explicit) if explicit is not None else _find_out_root(source) / "figs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tag(source: Path) -> str:
    name = source.resolve().name
    if name.endswith("_analysis"):
        name = name[:-9]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or "run"


def _save(fig, source: Path, stem: str, figdir: Path):
    base = figdir / f"{_tag(source)}_{stem}"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _preference_score(directory: Path) -> tuple[int, str]:
    name = directory.name
    score = 0
    if name.endswith("_analysis"):
        score += 30
    if name.startswith("production"):
        score += 20
    if re.search(r"_n500000$", name):
        score += 10
    return score, name


def _dedupe_dirs(dirs: list[Path], filename: str) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    for d in dirs:
        f = d / filename
        if f.exists():
            groups.setdefault(_sha256(f), []).append(d)
    return sorted((max(g, key=_preference_score) for g in groups.values()), key=lambda p: p.name)


def _scan_dirs(root: Path, filename: str) -> list[Path]:
    dirs = [f.parent for f in root.rglob(filename) if "figs" not in f.parts]
    return _dedupe_dirs(dirs, filename)


def _first_existing(root: Path, names: list[str], filename: str) -> Path | None:
    for name in names:
        d = root / name
        if (d / filename).exists():
            return d
    matches = _scan_dirs(root, filename)
    return max(matches, key=_preference_score) if matches else None


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


def _robust_scale(values: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = [np.abs(v[np.isfinite(v)]) for v in values]
    finite = [x for x in finite if x.size]
    if not finite:
        return 1.0
    v = float(np.nanpercentile(np.concatenate(finite), percentile))
    return max(v, np.finfo(float).eps)


def image_scales(dirs: list[Path], percentile: float = 99.0) -> tuple[float, float]:
    nom, pres = [], []
    for d in dirs:
        with np.load(d / "images.npz") as z:
            if not {"I_nom", "I_p", "I_Q"}.issubset(z.files):
                continue
            nom.append(_masked_central(z, "I_nom") - _masked_central(z, "I_Q"))
            pres.append(_masked_central(z, "I_p") - _masked_central(z, "I_Q"))
    return _robust_scale(nom, percentile), _robust_scale(pres, percentile)


def _add_target_overlay(ax):
    # Cu is a cube: use its projected square, not a circular proxy.
    ax.add_patch(Rectangle(
        (-CU_HALF, -CU_HALF), 2 * CU_HALF, 2 * CU_HALF,
        fill=False, edgecolor="0.35", lw=0.75, ls=":"
    ))
    ax.add_patch(Circle(
        (PB_CX, PB_CY), PB_ROI_R,
        fill=False, edgecolor="0.25", lw=0.8, ls="--"
    ))


# ---------------------------------------------------------------------------
# Theory figures


def plot_theory(outdir: str | Path, figdir: Path):
    out = Path(outdir)
    d = pd.read_csv(out / "theory_collapse.csv")

    from analysis import PATHS
    from physics import reduced_parameters, mu2_eta

    rp = reduced_parameters(PATHS["AlCu"], 1.0)
    eta = np.geomspace(2.0, 30.0, 250)
    eps = np.array([
        math.sqrt(rp["R"] * rp["B"] * mu2_eta(float(e), rp["B"], 2)) - 1.0
        for e in eta
    ])

    # Fixed-path reduced-variable collapse.
    fig, ax = plt.subplots(figsize=(4.7, 3.45), constrained_layout=True)
    ax.plot(eta, 100 * eps, color=PATH_COLOR["AlCu"], label="Al+Cu reduced curve")
    ax.scatter(d.eta_cut, 100 * d.epsilon, s=28, color="black", zorder=3,
               label="momentum settings")
    label_offsets = {1.0: (6, 6), 2.0: (7, 8), 3.5: (9, 8), 6.0: (9, 10)}
    for _, row in d.iterrows():
        dx, dy = label_offsets.get(float(row.p), (6, 6))
        ax.annotate(fr"${row.p:g}\,\mathrm{{GeV}}/c$",
                    (row.eta_cut, 100 * row.epsilon),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7.2,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.2))
    ax.set_xscale("log")
    ticks = [2, 3, 5, 10, 20, 30]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1.85, 32)
    ax.set_xlabel(r"$\eta_{\rm cut}=\theta_{\rm cut}/(\chi_c\sqrt{B})$")
    ax.set_ylabel(r"$\epsilon_M$ (%)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out, "collapse_eta", figdir)

    # Matched-k vs matched-eta composition dependence in one figure.
    p = out / "composition_matched.csv"
    if p.exists():
        comp = pd.read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.25), constrained_layout=True)
        for ax, match, xlabel, letter in (
            (axes[0], "k", r"matched $k=\theta_{\rm cut}/\theta_0$", "a"),
            (axes[1], "eta", r"matched $\eta_{\rm cut}$", "b"),
        ):
            q = comp[comp["match"] == match]
            for path in PATH_ORDER:
                g = q[q.path == path].sort_values("value")
                if g.empty:
                    continue
                ax.plot(g.value, 100.0 * g.epsilon, marker=PATH_MARKER[path], ms=4.5,
                        color=PATH_COLOR[path], label=PATH_LABEL[path])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\epsilon_M$ (%)")
            _panel(ax, letter)
        axes[1].legend(frameon=False, loc="upper left")
        _save(fig, out, "composition_comparison", figdir)

    # eta1 and numerical sub-table slope approach.
    p = out / "eta1_protocol.csv"
    if p.exists():
        e = pd.read_csv(p)
        deep = e[e.window_role == "eta1_asymptote"].copy()
        deepest_hi = deep.eta_max.max()
        deepest_lo = deep.loc[deep.eta_max == deepest_hi, "eta_min"].max()
        deep = deep[(deep.eta_min == deepest_lo) & (deep.eta_max == deepest_hi)]
        diag = e[(e.window_role == "slope_diagnostic") & (e.nmax == 2)].copy()
        diag["eta_center"] = np.sqrt(diag.eta_min * diag.eta_max)

        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), constrained_layout=True)
        xpos = np.arange(len(PATH_ORDER), dtype=float)
        for nmax, offset, marker in ((1, -0.055, "o"), (2, 0.055, "s")):
            g = deep[deep.nmax == nmax].set_index("path").reindex(PATH_ORDER)
            axes[0].plot(xpos + offset, g.eta1_joint.to_numpy(float), marker=marker,
                         color="0.2" if nmax == 1 else PATH_COLOR["AlCu"],
                         label=fr"$n\leq {nmax}$")
        axes[0].axhline(1.0, lw=0.8, color="k")
        axes[0].set_xticks(xpos, [PATH_LABEL[p] for p in PATH_ORDER], rotation=15, ha="right")
        axes[0].set_ylabel(r"asymptotic $\eta_1$")
        axes[0].set_xlabel("material path")
        axes[0].legend(frameon=False)
        _panel(axes[0], "a")

        for path in PATH_ORDER:
            g = diag[diag.path == path].sort_values("eta_center")
            if g.empty:
                continue
            x = g.eta_center.to_numpy(float)
            lo = g.eta_min.to_numpy(float)
            hi = g.eta_max.to_numpy(float)
            y = 100.0 * (g.slope_ratio.to_numpy(float) - 1.0)
            axes[1].errorbar(x, y, xerr=np.vstack([x - lo, hi - x]),
                             marker=PATH_MARKER[path], ms=4.5, capsize=2,
                             color=PATH_COLOR[path], label=PATH_LABEL[path])
        eta0 = float(e.eta_table_max.iloc[0])
        axes[1].axvline(eta0, lw=0.9, ls="--", color="k", label=fr"$\eta_0={eta0:g}$")
        axes[1].axhline(0.0, lw=0.8, color="k")
        axes[1].set_xscale("log")
        axes[1].set_xlim(7.0, eta0 * 1.08)
        axes[1].set_ylabel(r"$(m/2R-1)$ (%)")
        axes[1].set_xlabel(r"fit-window $\eta$ (center and span)")
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        _panel(axes[1], "b")
        _save(fig, out, "eta1_and_slope", figdir)

    # Energy-loss confound.
    p = out / "energy_loss_calibration.csv"
    if p.exists():
        cal = pd.read_csv(p)
        alcu = cal[cal.path == "AlCu"].sort_values("p")
        if not alcu.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.15), constrained_layout=True)
            axes[0].axhline(0.0, color="k", lw=0.8)
            axes[0].plot(alcu.p, 100.0 * alcu.dp_over_p, "o-", color=PATH_COLOR["Al25"])
            axes[0].set_xlabel(r"$p_{\rm in}$ (GeV/$c$)")
            axes[0].set_ylabel(r"$\Delta p/p$ (%)")
            _panel(axes[0], "a")

            axes[1].plot(alcu.p, 100.0 * alcu.epsilon_matched_dchi, "o-",
                         color=PATH_COLOR["Al25"], label="matched")
            axes[1].plot(alcu.p, 100.0 * alcu.epsilon_mixed_dchi, "s-",
                         color=PATH_COLOR["AlCu"], label="upstream-tagged")
            axes[1].set_xlabel(r"$p_{\rm in}$ (GeV/$c$)")
            axes[1].set_ylabel(r"normalization mismatch $\epsilon$ (%)")
            axes[1].legend(frameon=False, loc="lower right")
            _panel(axes[1], "b")
            _save(fig, out, "energy_loss_effect", figdir)

    # Tail and truncation are model checks; combine them into one compact figure.
    tail_path = out / "tail_check.csv"
    conv_path = out / "truncation_convergence.csv"
    if tail_path.exists() or conv_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.15), constrained_layout=True)
        if tail_path.exists():
            tail = pd.read_csv(tail_path)
            axes[0].plot(tail.theta_mrad, tail.ratio, "o-", color=PATH_COLOR["Al25"])
            axes[0].axhline(1.0, ls="--", lw=0.9, color="k")
            axes[0].set_xlabel(r"$\Theta$ (mrad)")
            axes[0].set_ylabel(r"$h_M(\Theta)\Theta^3/(2\chi_c^2)$")
            for _, row in tail.iterrows():
                axes[0].annotate(f"{row.ratio:.3f}", (row.theta_mrad, row.ratio),
                                 xytext=(4, 3), textcoords="offset points", fontsize=7.5)
            _panel(axes[0], "a")
        else:
            axes[0].axis("off")
        if conv_path.exists():
            conv = pd.read_csv(conv_path)
            axes[1].axhline(0.0, lw=0.8, color="k")
            axes[1].plot(conv.p, 100.0 * conv.shift_abs, "o-", color=PATH_COLOR["Pb15"])
            axes[1].set_xlabel(r"$p$ (GeV/$c$)")
            axes[1].set_ylabel(r"$\epsilon_M^{(n\leq2)}-\epsilon_M^{(n\leq1)}$ (pp)")
            _panel(axes[1], "b")
        else:
            axes[1].axis("off")
        _save(fig, out, "model_checks", figdir)


# ---------------------------------------------------------------------------
# Detector-level figures


def plot_images(
    outdir: str | Path,
    figdir: Path,
    nominal_vmax: float | None = None,
    p_vmax: float | None = None,
    percentile: float = 99.0,
):
    out = Path(outdir)
    with np.load(out / "images.npz") as z:
        required = {"I_nom", "I_p", "I_Q"}
        if not required.issubset(z.files):
            return
        In = _masked_central(z, "I_nom")
        Ip = _masked_central(z, "I_p")
        IQ = _masked_central(z, "I_Q")
        count_mask = None
        if "counts" in z.files:
            count_mask = _central(z["counts"]) < MIN_VOX_COUNT

    maps = [In - IQ, Ip - IQ]
    labels = [r"$I_{\rm nom}-I_Q$", r"$I_p-I_Q$"]
    scales = [
        nominal_vmax or _robust_scale([maps[0]], percentile),
        p_vmax or _robust_scale([maps[1]], percentile),
    ]
    cmap = _diverging_cmap()

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.45), constrained_layout=False)
    for i, (ax, m, label, vmax) in enumerate(zip(axes, maps, labels, scales)):
        im = ax.imshow(m.T, origin="lower",
                       extent=[-VOX_HALF, VOX_HALF, -VOX_HALF, VOX_HALF],
                       vmin=-vmax, vmax=vmax, cmap=cmap, interpolation="nearest")
        _add_target_overlay(ax)
        ax.set_xlabel(r"$x$ (cm)")
        ax.set_title(label)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.ax.set_ylabel("")
        _panel(ax, "ab"[i])
    axes[0].set_ylabel(r"$y$ (cm)")
    if count_mask is not None and np.any(count_mask):
        axes[1].legend(handles=[Patch(facecolor="#e8e8e8", edgecolor="0.6",
                                      label=fr"masked: $N_{{\rm voxel}}<{MIN_VOX_COUNT}$")],
                       loc="lower center", frameon=True, fontsize=7)
    _save(fig, out, "difference_maps", figdir)

    # Combine the two scalar production summaries.
    path_csv = out / "path_residuals.csv"
    art_csv = out / "artifact_summary.csv"
    if path_csv.exists() or art_csv.exists():
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)
        if art_csv.exists():
            a = pd.read_csv(art_csv).iloc[0]
            vals = [a.artifact_rms, a.p_residual_rms]
            bars = axes[0].bar([r"$I_{\rm nom}-I_Q$", r"$I_p-I_Q$"], vals,
                               color=[PATH_COLOR["Pb15"], PATH_COLOR["Cu15"]])
            for bar, val in zip(bars, vals):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f"{val:.2g}", ha="center", va="bottom", fontsize=8)
            axes[0].set_ylabel("image RMS difference")
            axes[0].text(0.97, 0.94, f"reduction = {100.0*a.p_reduction:.1f}%",
                         transform=axes[0].transAxes, ha="right", va="top", fontsize=8)
            _panel(axes[0], "a")
        else:
            axes[0].axis("off")
        if path_csv.exists():
            d = pd.read_csv(path_csv)
            order = [x for x in ["Al-only", "Cu-bearing"] if x in set(d.region)]
            d = d.set_index("region").reindex(order).reset_index()
            bars = axes[1].bar(d.region, d.image_rms,
                               color=[PATH_COLOR["Al25"], PATH_COLOR["AlCu"]][:len(d)])
            for bar, val in zip(bars, d.image_rms.to_numpy(float)):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f"{val:.2g}", ha="center", va="bottom", fontsize=8)
            axes[1].set_ylabel(r"image RMS of $I_p-I_Q$")
            axes[1].set_xlabel("reference-path class")
            _panel(axes[1], "b")
        else:
            axes[1].axis("off")
        _save(fig, out, "production_summary", figdir)

    # Adaptive retention: truth first, reconstruction second, same semantic colors.
    p = out / "adaptive_retention.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.denominator == "all generated"].copy()
        if not d.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.05), sharey=True, constrained_layout=True)
            specs = [
                ("truth", ["Pb-crossing", "Cu-only"], "truth classification", "a"),
                ("reconstructed", ["Pb ROI", "Cu ROI"], "reconstructed ROI", "b"),
            ]
            for ax, (classification, order, title, letter) in zip(axes, specs):
                g = d[d.classification == classification].set_index("group").reindex(order).reset_index()
                vals = 100.0 * g.retention.to_numpy(float)
                bars = ax.bar(g.group, vals, color=[PATH_COLOR["Pb15"], PATH_COLOR["Cu15"]])
                ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
                ax.set_ylim(0, 100)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=12)
                _panel(ax, letter)
            axes[0].set_ylabel("adaptive-cut retention (%)")
            _save(fig, out, "adaptive_retention", figdir)


# ---------------------------------------------------------------------------
# Gradient experiment


def plot_gradient(outdir: str | Path, figdir: Path, percentile: float = 99.0):
    out = Path(outdir)
    with np.load(out / "gradient_maps.npz") as z:
        required = {"observed", "predicted_unweighted", "residual_unweighted"}
        if not required.issubset(z.files):
            return
        observed = _masked_central(z, "observed")
        predicted = _masked_central(z, "predicted_unweighted")
        residual = _masked_central(z, "residual_unweighted")

    obs_scale = _robust_scale([observed, residual], percentile)
    pred_scale = _robust_scale([predicted], percentile)
    cmap = _diverging_cmap()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), constrained_layout=True)
    im_obs = None
    for i, (ax, v, label, vmax) in enumerate(zip(
        axes,
        [observed, predicted, residual],
        [r"observed $I_{\rm nom}-I_Q$", "normalization-field predictor", "one-amplitude residual"],
        [obs_scale, pred_scale, obs_scale],
    )):
        im = ax.imshow(v.T, origin="lower",
                       extent=[-VOX_HALF, VOX_HALF, -VOX_HALF, VOX_HALF],
                       vmin=-vmax, vmax=vmax, cmap=cmap, interpolation="nearest")
        if i == 0:
            im_obs = im
        ax.set_xlabel(r"$x$ (cm)")
        ax.set_title(label, fontsize=9)
        _panel(ax, "abc"[i])
    axes[0].set_ylabel(r"$y$ (cm)")

    # One shared scale for observed/residual; predictor gets a separate scale.
    fig.colorbar(im_obs, ax=[axes[0], axes[2]], fraction=0.026, pad=0.02,
                 label="weight difference")
    pred_im = axes[1].images[0]
    fig.colorbar(pred_im, ax=axes[1], fraction=0.046, pad=0.03,
                 label="normalization excess")

    p = out / "gradient_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        r = s[s.predictor == "normalization_field"]
        if not r.empty:
            row = r.iloc[0]
            axes[2].text(0.03, 0.97,
                         f"$r={row.correlation:.3f}$\nresidual = {100.0*row.residual_fraction:.1f}%",
                         transform=axes[2].transAxes, ha="left", va="top", fontsize=8,
                         bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.75", pad=2.0))
    _save(fig, out, "gradient_causal_maps", figdir)


# ---------------------------------------------------------------------------
# Paired-seed summary


def _comparison_label(x: str) -> str:
    return {
        "I_nom-I_Q": r"$I_{\rm nom}-I_Q$",
        "I_p-I_Q": r"$I_p-I_Q$",
    }.get(x, x)


def plot_paired(csv_path: Path, figdir: Path):
    if not csv_path.exists():
        return
    d = pd.read_csv(csv_path)
    if d.empty:
        return
    source = csv_path.parent / "paired"
    labels = [_comparison_label(x) for x in d.comparison]
    x = np.arange(len(d))
    colors = [PATH_COLOR["Pb15"], PATH_COLOR["Cu15"]][:len(d)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), constrained_layout=False)
    for ax, mean, sd, ylabel, letter in (
        (axes[0], d.dSNR_mean, d.dSNR_sd, r"paired $\Delta$SNR", "a"),
        (axes[1], d.dCNR_mean, d.dCNR_sd, r"paired $\Delta$CNR", "b"),
    ):
        ax.axhline(0.0, color="k", lw=0.8)
        for i in range(len(d)):
            ax.errorbar(x[i], mean.iloc[i], yerr=sd.iloc[i], fmt="o", capsize=3,
                        color=colors[i], ms=5)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.text(-0.10, 1.01, f"({letter})", transform=ax.transAxes,
                ha="left", va="bottom", fontweight="bold", fontsize=10)
    fig.subplots_adjust(top=0.92, bottom=0.19, wspace=0.32)
    _save(fig, source, "seed_summary", figdir)


# ---------------------------------------------------------------------------
# CLI modes


def run_paper(root: Path, explicit_figdir: str | Path | None = None, percentile: float = 99.0):
    """Generate only the canonical manuscript figures, not every seed diagnostic."""
    root = root.resolve()
    figdir = _figdir(root, explicit_figdir)

    theory = _first_existing(root, ["theory"], "theory_collapse.csv")
    production = _first_existing(root, ["production_analysis", "production"], "images.npz")
    gradient = _first_existing(root, ["gradient_analysis", "gradient"], "gradient_maps.npz")

    if theory:
        plot_theory(theory, figdir)
    if production:
        # Paper production map uses its own robust scale; seed-to-seed comparison
        # scales are only needed in --all diagnostic mode.
        plot_images(production, figdir, percentile=percentile)
    if gradient:
        plot_gradient(gradient, figdir, percentile=percentile)

    paired = root / "paired_seed_summary.csv"
    if paired.exists():
        plot_paired(paired, figdir)

    print(f"paper figures written to {figdir}")
    print(f"theory source:     {theory}")
    print(f"production source: {production}")
    print(f"gradient source:   {gradient}")


def run_all(root: Path, explicit_figdir: str | Path | None = None, percentile: float = 99.0):
    """Generate all unique seed/count diagnostics as well as theory outputs."""
    root = root.resolve()
    figdir = _figdir(root, explicit_figdir)

    theory_dirs = _scan_dirs(root, "theory_collapse.csv")
    image_dirs = _scan_dirs(root, "images.npz")
    gradient_dirs = _scan_dirs(root, "gradient_maps.npz")
    nominal_vmax, p_vmax = image_scales(image_dirs, percentile) if image_dirs else (None, None)

    for d in theory_dirs:
        plot_theory(d, figdir)
    for d in image_dirs:
        plot_images(d, figdir, nominal_vmax=nominal_vmax, p_vmax=p_vmax, percentile=percentile)
    for d in gradient_dirs:
        plot_gradient(d, figdir, percentile=percentile)

    paired = root / "paired_seed_summary.csv"
    if paired.exists():
        plot_paired(paired, figdir)

    print(f"all figures written to {figdir}")
    print(f"theory sources:   {len(theory_dirs)}")
    print(f"image sources:    {len(image_dirs)}")
    print(f"gradient sources: {len(gradient_dirs)}")
    if image_dirs:
        print(f"common nominal-map |scale|: {nominal_vmax:.6g}")
        print(f"common p-residual |scale|:  {p_vmax:.6g}")


def main():
    ap = argparse.ArgumentParser(description="Generate centralized publication figures.")
    ap.add_argument("--root", default="out", help="results root; default: out")
    ap.add_argument("--figdir", default=None, help="override figure directory; default: <root>/figs")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="canonical manuscript figures only")
    mode.add_argument("--all", action="store_true", help="all unique seed/count diagnostics")
    ap.add_argument("--outdir", default=None, help="plot one results directory")
    ap.add_argument("--kind", choices=["theory", "images", "gradient", "all"], default="all")
    ap.add_argument("--percentile", type=float, default=99.0, help="robust display-scale percentile")
    a = ap.parse_args()

    if a.paper:
        run_paper(Path(a.root), a.figdir, percentile=a.percentile)
        return
    if a.all:
        run_all(Path(a.root), a.figdir, percentile=a.percentile)
        return

    if a.outdir is None:
        raise SystemExit("Use --paper, --all, or provide --outdir PATH")

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
