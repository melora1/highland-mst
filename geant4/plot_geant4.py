"""Plotting/analysis helpers for the Geant4 cross-check.

Each stage of the validation plan is one function here. Run a stage from the
CLI (`python plot_geant4.py <stage>`) or import the functions elsewhere.

Stages that only read the Geant4 angle dumps (`convergence`) run in any env
with numpy+matplotlib. Stages that compare against the quadrature model
(`cu_sweep`) import `physics.py`/`analysis.py` and must run in the project
venv where those are importable.
"""

from __future__ import annotations

import os
import re
import sys
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")  # file output, no GUI
import matplotlib.pyplot as plt

# Make the repo root importable so `from physics import ...` works when this
# file lives in a subfolder (e.g. geant4/). Done once, at import time.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_angles(path):
    """Read one angle-dump file -> 1D array of theta_space (rad)."""
    return np.loadtxt(path)


def truncated_moments(a, cut):
    """Fc, truncated theta_rms, and mean over theta < cut."""
    keep = a[a < cut]
    if keep.size == 0:
        return dict(Fc=0.0, thrms=np.nan, mean=np.nan, n=0)
    return dict(
        Fc=keep.size / a.size,
        thrms=float(np.sqrt(np.mean(keep**2))),
        mean=float(keep.mean()),
        n=int(keep.size),
    )


def _const_cal(material, thickness_cm, p_gev, cut):
    """Quadrature truncated moments for a single-material slab via physics.py."""
    from physics import constant_calibration
    from analysis import path_X

    X = path_X({material: thickness_cm})
    r = constant_calibration(X, p_gev, theta_cut=cut)
    return dict(Fc=r["Fc"], thrms=r["theta_rms"], theta0=r["theta_space"] / np.sqrt(2))


def _boot_rms_frac(a, cut, thrms_quad, nboot=200, seed=0):
    """95% bootstrap CI on (theta_rms_g4 / theta_rms_quad - 1)."""
    rng = np.random.default_rng(seed)
    keep = a[a < cut]
    if keep.size < 20 or not np.isfinite(thrms_quad) or thrms_quad <= 0:
        return np.nan, np.nan
    fr = np.empty(nboot)
    for i in range(nboot):
        s = rng.choice(keep, keep.size, replace=True)
        fr[i] = np.sqrt(np.mean(s**2)) / thrms_quad - 1.0
    return float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))


# ---------------------------------------------------------------------------
# Stage 1a: statistical convergence (reads dumps only)
# ---------------------------------------------------------------------------
def convergence(
    pattern="out/conv_Cu15_p2_urban_N*.txt",
    theta0=0.0681 / np.sqrt(2),  # Cu15 @2GeV/c core; only sets the k-cut scale
    k=10,
    out_png="out/convergence.png",
):
    """Statistical convergence of truncated theta_rms vs N.

    Returns the summary array [N, yield, thrms_mrad, Fc] and writes a 2-panel
    figure. `theta0` only sets the k-cut scale; the convergence shape is
    insensitive to its exact value.
    """
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r"N(\d+)", f).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no files match {pattern}")

    cut = k * theta0
    print(f"{'N':>10} {'yield':>8} {'thrms_k%d(mrad)' % k:>16} {'Fc':>7} {'mean(mrad)':>11}")
    rows = []
    for f in files:
        a = load_angles(f)
        N = int(re.search(r"N(\d+)", f).group(1))
        m = truncated_moments(a, cut)
        rows.append((N, a.size, m["thrms"] * 1e3, m["Fc"]))
        print(f"{N:>10} {a.size:>8} {m['thrms'] * 1e3:>16.4f} {m['Fc']:>7.4f} {a.mean() * 1e3:>11.3f}")

    rows = np.array(rows, float)
    ref = rows[-1, 2]
    drift = 100.0 * np.abs(rows[:, 2] / ref - 1.0)
    print("\ndrift vs largest N (%):", np.round(drift, 3))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogx(rows[:, 0], rows[:, 2], "o-")
    ax[0].set_xlabel("N events")
    ax[0].set_ylabel(f"truncated theta_rms @k={k} (mrad)")
    ax[0].set_title("convergence")
    ax[1].loglog(rows[:, 0], np.maximum(drift, 1e-3), "s-")
    ax[1].axhline(0.5, ls="--", c="r", label="0.5% target")
    ax[1].set_xlabel("N events")
    ax[1].set_ylabel("|drift| vs max-N (%)")
    ax[1].legend()
    ax[1].set_title("statistical convergence")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"\nwrote {out_png}")
    return rows


# ---------------------------------------------------------------------------
# Stage 1: kernel validation on single-material slabs vs Geant4
# ---------------------------------------------------------------------------
def cu_sweep(
    material="Cu",
    thicknesses=(3.0, 15.0),
    momenta=(1.0, 2.0, 3.5, 6.0),
    ks=(2, 5, 10, 20, 40),
    outdir="out",
    csv_out=None,
    png_out=None,
):
    """Compare quadrature moments against Geant4 (Urban/Wentzel) per (t,p,k).

    Pass criterion (pre-registered): at each (t,p,k) the point passes if
    |rms_frac| lies within the LARGER of the Urban/Wentzel spread or the 95%
    bootstrap CI half-width. A (t,p) config passes if all k<=20 pass.
    """
    import pandas as pd

    csv_out = csv_out or f"{outdir}/stage1_{material}.csv"
    png_out = png_out or f"{outdir}/stage1_{material}.png"

    rows = []
    for t in thicknesses:
        for p in momenta:
            # theta0 at the physical 200 mrad cut sets the k -> cut mapping.
            theta0 = _const_cal(material, t, p, 0.200)["theta0"]
            dumps = {
                m: load_angles(f"{outdir}/{material}_t{t}_p{p}_{m}.txt")
                for m in ("urban", "wentzel")
            }
            for k in ks:
                cut = k * theta0
                q = _const_cal(material, t, p, cut)
                g = {m: truncated_moments(dumps[m], cut) for m in dumps}
                rms_frac = g["urban"]["thrms"] / q["thrms"] - 1.0
                lo, hi = _boot_rms_frac(dumps["urban"], cut, q["thrms"])
                boot_half = 0.5 * (hi - lo) if np.isfinite(hi) else np.nan
                spread = abs(g["urban"]["thrms"] / g["wentzel"]["thrms"] - 1.0)
                band = max(spread, boot_half) if np.isfinite(boot_half) else spread
                passed = (abs(rms_frac) <= band) if np.isfinite(band) else False
                rows.append(
                    dict(
                        material=material, t=t, p=p, k=k,
                        Fc_quad=q["Fc"], Fc_g4=g["urban"]["Fc"],
                        thrms_quad=q["thrms"], thrms_g4=g["urban"]["thrms"],
                        rms_frac=rms_frac, boot_half=boot_half,
                        model_spread=spread, band=band,
                        gated=(k <= 20), pass_=bool(passed),
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)

    print(f"\n{'t':>6}{'p':>7}  verdict   (gated k<=20)")
    for (t, p), g in df.groupby(["t", "p"]):
        gk = g[g.gated]
        verdict = "PASS" if gk.pass_.all() else "FAIL"
        worst = gk.loc[gk.rms_frac.abs().idxmax()]
        print(
            f"{t:>6}{p:>7}  {verdict}   worst |rms_frac|={abs(worst.rms_frac) * 100:5.2f}% "
            f"@k={int(worst.k)} (band {worst.band * 100:4.2f}%)"
        )
    print(f"\nwrote {csv_out}")

    fig, axes = plt.subplots(
        len(thicknesses), len(momenta),
        figsize=(4 * len(momenta), 3.2 * len(thicknesses)),
        squeeze=False, sharex=True,
    )
    for i, t in enumerate(thicknesses):
        for j, p in enumerate(momenta):
            ax = axes[i][j]
            g = df[(df.t == t) & (df.p == p)]
            ax.fill_between(g.k, -g.band * 100, g.band * 100, alpha=0.2,
                            color="gray", label="validation band")
            ax.errorbar(g.k, g.rms_frac * 100, yerr=g.boot_half * 100,
                        fmt="o-", capsize=3, label="rms_frac")
            ax.axvline(20, ls=":", c="r")
            ax.axhline(0, ls="-", lw=0.5, c="k")
            ax.set_title(f"{material} {t}cm, {p} GeV/c")
            if i == len(thicknesses) - 1:
                ax.set_xlabel("k")
            if j == 0:
                ax.set_ylabel("rms_frac (%)")
    axes[0][0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png_out, dpi=120)
    print(f"wrote {png_out}")
    return df


# ---------------------------------------------------------------------------
# Impact table: truncated-RMS difference at the PHYSICAL 200 mrad cut
# ---------------------------------------------------------------------------
def impact_table(
    configs=(
        ("Cu", 3.0), ("Cu", 15.0), ("Pb", 2.0), ("Pb", 8.0),
    ),
    momenta=(1.0, 2.0, 3.5, 6.0),
    theta_cut=0.200,   # physical production cut (rad)
    outdir="out",
    csv_out="out/impact_200mrad.csv",
):
    """Quantify where the tail discrepancy actually touches production results.

    For each (material, thickness, momentum) config, evaluate the quadrature-vs-
    Geant4 truncated-RMS fractional difference at the PHYSICAL 200 mrad cut (not
    an arbitrary reduced k), with a 95% bootstrap CI. Also report where 200 mrad
    falls in reduced-angle k, so the reader sees which configs sit in the core
    (low k, good agreement) vs the tail (high k, larger discrepancy).

    Reads existing urban/wentzel dumps only -- no new simulation.
    """
    import pandas as pd

    rows = []
    for material, t in configs:
        for p in momenta:
            cal = _const_cal(material, t, p, theta_cut)
            theta0 = cal["theta0"]
            k_at_cut = theta_cut / theta0
            rec = dict(
                material=material, t=t, p=p,
                theta0_mrad=theta0 * 1e3, k_at_200mrad=k_at_cut,
                thrms_quad=cal["thrms"],
            )
            for m in ("urban", "wentzel"):
                f = f"{outdir}/{material}_t{t}_p{p}_{m}.txt"
                try:
                    a = load_angles(f)
                except OSError:
                    rec[f"rms_frac_{m}"] = np.nan
                    rec[f"ci_lo_{m}"] = np.nan
                    rec[f"ci_hi_{m}"] = np.nan
                    continue
                g = truncated_moments(a, theta_cut)
                rec[f"rms_frac_{m}"] = g["thrms"] / cal["thrms"] - 1.0
                lo, hi = _boot_rms_frac(a, theta_cut, cal["thrms"])
                rec[f"ci_lo_{m}"] = lo
                rec[f"ci_hi_{m}"] = hi
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)

    print(f"\nTruncated-RMS difference at {theta_cut*1e3:.0f} mrad "
          f"(quadrature vs Geant4), with 95% bootstrap CI")
    print(f"{'mat':>4}{'t':>6}{'p':>6}{'k@cut':>7}  "
          f"{'urban %':>16}   {'wentzel %':>16}")
    for _, r in df.iterrows():
        def fmt(m):
            v = r.get(f"rms_frac_{m}", np.nan)
            lo = r.get(f"ci_lo_{m}", np.nan)
            hi = r.get(f"ci_hi_{m}", np.nan)
            if not np.isfinite(v):
                return f"{'MISSING':>16}"
            half = 0.5 * (hi - lo) * 100 if np.isfinite(hi) else float("nan")
            return f"{v*100:7.2f} +/- {half:4.2f}"
        print(f"{r.material:>4}{r.t:>6}{r.p:>6}{r.k_at_200mrad:>7.1f}  "
              f"{fmt('urban')}   {fmt('wentzel')}")

    print(f"\nwrote {csv_out}")
    print("\nRead: configs with k@cut <~ 10 sit in the core (few-% agreement);")
    print("configs with k@cut >~ 20 sit in the tail (larger, Z-ordered discrepancy).")
    return df


# ---------------------------------------------------------------------------
# Impact-summary figure: difference at 200 mrad vs reduced angle, by material
# ---------------------------------------------------------------------------
def impact_figure(
    csv_in="out/impact_200mrad.csv",
    out_png="out/geant4_impact_summary.png",
):
    """One-panel summary: model-vs-Geant4 truncated-RMS difference at the
    physical 200 mrad cut, versus the reduced angle k at which that cut falls,
    colored by material. This is the visual version of Table tab:g4impact.

    If the impact CSV is absent, it is generated first via impact_table(), so
    this stage is self-sufficient given the angle dumps.
    """
    import os
    import pandas as pd

    if not os.path.exists(csv_in):
        impact_table(csv_out=csv_in)
    df = pd.read_csv(csv_in)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = {"Cu": "tab:orange", "Pb": "tab:blue"}
    markers = {2.0: "o", 3.0: "o", 8.0: "s", 15.0: "s"}  # thin=circle, thick=square

    for mat in ("Cu", "Pb"):
        sub = df[df.material == mat].sort_values("k_at_200mrad")
        # use the urban column; wentzel agrees to ~0.2 pp
        y = sub["rms_frac_urban"] * 100
        lo = sub["ci_lo_urban"] * 100
        hi = sub["ci_hi_urban"] * 100
        yerr = [(y - lo).abs(), (hi - y).abs()]
        ax.errorbar(
            sub["k_at_200mrad"], y, yerr=yerr,
            fmt="o", color=colors[mat], capsize=3, label=mat, zorder=3,
        )

    ax.axhline(0, ls="-", lw=0.6, c="k", zorder=1)
    ax.axvline(20, ls=":", c="r", lw=1, zorder=1, label="$k=20$")
    ax.set_xlabel(r"reduced angle $k=\theta_{\rm cut}/\theta_0$ at 200 mrad")
    ax.set_ylabel(r"$\theta_{\rm rms}^{\rm model}/\theta_{\rm rms}^{\rm G4}-1$ (%)")
    ax.set_title("Truncated-RMS difference at the 200 mrad production cut")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    # also emit a PDF for LaTeX inclusion
    pdf = out_png.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf)
    print(f"wrote {out_png} and {pdf}")
    return df


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------
_STAGES = {
    "convergence": convergence,
    "cu_sweep": cu_sweep,
    "pb_sweep": lambda: cu_sweep(material="Pb", thicknesses=(2.0, 8.0)),
    "impact_table": impact_table,
    "impact_figure": impact_figure,
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "convergence"
    fn = _STAGES.get(stage)
    if fn is None:
        raise SystemExit(f"unknown stage '{stage}'; choose from {list(_STAGES)}")
    fn()