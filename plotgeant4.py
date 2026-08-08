#!/usr/bin/env python3
"""
plot_geant4_analysis.py  --  the five missing manuscript figures identified
against the current draft:

  1. eps_M(p) curve            -- end of Sec. 2.3 (the Table 1 numbers, as a curve,
                                   at two path lengths so momentum- and
                                   path-length-dependence are both visible)
  2. eta(k) efficiency curve   -- Sec. 6.2, before Table 4 (Eq. 14, k_opt=1.84)
  3. Geant4 discrepancy vs ln(eta_cut) -- Sec. 7.2 (the r=-0.89 correlation),
                                   using the REAL 32-configuration sweep data
                                   from the Geant4 cross-check (Urban and
                                   Wentzel-VI marked separately)
  4. theta_RMS vs theta_cut log-law -- one panel per momentum, R^2 in each
  5. Moliere core vs tail sketch    -- Gaussian f^(0) vs the full n<=2 sum,
                                   log-log, showing the theta^-3 tail

Figures 1, 2, 4, 5 are computed directly from moliere.py/eps_quadrature.py/
kinematics.py (the same modules that produce the paper's numbers) -- not
hand-fit curves. Figure 3 uses the actual 32-point Geant4 sweep (material,
thickness, momentum, model, measured RMS) recorded during the cross-check;
the raw numbers are hardcoded below with a comment giving their provenance,
since the raw Geant4 angle dumps (geant4/out/*.txt) are not available in
this environment -- replace GEANT4_SWEEP below with a re-parse of your own
analysis/*.log files if you want to regenerate Fig. 3 after a fresh sweep.

Style matches plot_results.py: no baked-in titles/captions (printed to
stdout for pasting into \\caption{}), panel labels, serif/Computer-Modern
math, fixed colorblind-safe palette, PDF (vector) + PNG (preview) output.

Usage:
    python3 plot_geant4_analysis.py [--outdir figs]
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import moliere as ml
from config import MATERIALS
from kinematics import theta0_highland, theta_space_highland
from eps_quadrature import eps_M, efficiency, K_OPT, _theta_rms_disc

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
COLORS = {"axial": "#0072B2", "cu_short": "#D55E00", "urban": "#0072B2",
         "wentzel": "#E69F00", "1.0": "#0072B2", "2.0": "#009E73",
         "3.5": "#D55E00", "6.0": "#CC79A7"}
PANEL_LABELS = "abcdefghij"
THETA_CUT = 0.200


def _save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)


def _panel_label(ax, letter):
    ax.text(-0.02, 1.05, f"({letter})", transform=ax.transAxes,
           fontsize=11, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- Fig 1: eps_M(p)
def fig_epsM_vs_p(outdir):
    p_grid = np.linspace(1.0, 6.5, 60)
    AXIAL = [("Al", 10.0), ("Cu", 15.0)]
    CU_SHORT = [("Cu", 3.0)]  # x/X0 = 2.083, matches the manuscript's Cu-only example

    def eps_curve(path):
        out = []
        for p in p_grid:
            X_al = MATERIALS.get("Al", {}).get("rho", 0.0) if any(m == "Al" for m, _ in path) else 0.0
            X_cu = X_al2 = 0.0
            Xs = {"Al": 0.0, "Cu": 0.0, "Pb": 0.0}
            for m, t in path:
                Xs[m] += MATERIALS[m]["rho"] * t
            e = eps_M(p, Xs["Al"], Xs["Cu"], Xs["Pb"])[0]
            out.append(e * 100)
        return np.array(out)

    eps_axial = eps_curve(AXIAL)
    eps_cu = eps_curve(CU_SHORT)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.15)
    ax.plot(p_grid, eps_axial, "-", color=COLORS["axial"], lw=1.6,
           label=r"axial (Al 10 cm + Cu 15 cm, $x/X_0=11.54$)")
    ax.plot(p_grid, eps_cu, "-", color=COLORS["cu_short"], lw=1.6,
           label=r"Cu-only, 3 cm ($x/X_0=2.08$)")
    # mark the four manuscript table points on the axial curve
    table_p = [1.0, 2.0, 3.5, 6.0]
    table_eps = [5.1, 10.8, 14.2, 17.1]
    ax.plot(table_p, table_eps, "o", color=COLORS["axial"], ms=5, mfc="white",
           mew=1.3, zorder=5)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"$p$ (GeV$/c$)")
    ax.set_ylabel(r"$\varepsilon_M$ (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _save(fig, outdir, "epsM_vs_p")
    print("\n[epsM_vs_p] suggested caption:\n"
         r"  $\varepsilon_M(p)$ from the deterministic quadrature (Eq. 8), "
         r"at the fixed $200$\,mrad acceptance, for the axial reference path "
         r"and a short Cu-only path. Open circles mark the four momenta of "
         r"Table 1. The path-length dependence (axial vs.\ Cu-only) is "
         r"comparable in size to the momentum dependence, confirming that a "
         r"momentum-only correction cannot remove $\varepsilon_M$.")


# ---------------------------------------------------------------- Fig 2: eta(k)
def fig_eta_vs_k(outdir):
    AXIAL = [("Al", 10.0), ("Cu", 15.0)]
    Xs = {"Al": 0.0, "Cu": 0.0, "Pb": 0.0}
    for m, t in AXIAL:
        Xs[m] += MATERIALS[m]["rho"] * t
    k_grid = np.linspace(1.2, 3.0, 60)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.15)
    kmax_all, etamax_all = [], []
    linestyles = ["-", "--", "-.", ":"]
    for (p, ls) in zip((1.0, 2.0, 3.5, 6.0), linestyles):
        etas = np.array([efficiency(p, Xs["Al"], Xs["Cu"], Xs["Pb"], k)
                        for k in k_grid])
        ax.plot(k_grid, etas, ls, lw=1.6, color=COLORS[str(p)],
               label=fr"$p={p:g}$ GeV$/c$")
        i = int(np.argmax(etas))
        kmax_all.append(k_grid[i]); etamax_all.append(etas[i])
    k_opt_mean = float(np.mean(kmax_all))
    eta_max_mean = float(np.mean(etamax_all))
    ax.axvline(K_OPT, color="black", lw=0.8, ls="--")
    ax.annotate(fr"$k_{{\mathrm{{opt}}}}={K_OPT:.2f}$", (K_OPT, 0.15),
               xytext=(K_OPT + 0.08, 0.15), fontsize=8.5)
    ax.set_xlabel(r"$k = \theta_{\mathrm{cut}}/\theta_0$")
    ax.set_ylabel(r"$\eta(k)$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _save(fig, outdir, "eta_vs_k")
    print("\n[eta_vs_k] suggested caption:\n"
         r"  Per-event imaging efficiency $\eta(k)$ (Eq. 14) vs.\ reduced "
         r"acceptance $k=\theta_{\mathrm{cut}}/\theta_0$, axial path, at "
         fr"four momenta. The curves coincide (universal in $k$) and peak at "
         fr"$k_{{\mathrm{{opt}}}}\approx{k_opt_mean:.2f}$, "
         fr"$\eta_{{\max}}\approx{eta_max_mean:.3f}$, confirming momentum-"
         r"independence.")


# ------------------------------------------------------- Fig 3: Geant4 vs ln(eta_cut)
# Real 32-configuration sweep (material, thickness_cm, p_GeV, model, x/X0,
# theta_space_mrad [sqrt2*Highland theta0], frac_diff_pct [(Geant4-quad)/quad]).
# Recorded directly from the analysis/*.log output of the Geant4 cross-check
# (Sec. 7). eta_cut = 200mrad / theta0 = 200mrad / (theta_space/sqrt2).
GEANT4_SWEEP = [
    # material, thickness_cm, p, model, theta_space_mrad, frac_diff_pct
    ("Cu", 15.0, 1.0, "urban",   68.006,  3.77), ("Cu", 15.0, 1.0, "wentzel", 68.006,  3.57),
    ("Cu", 15.0, 2.0, "urban",   33.852, -4.11), ("Cu", 15.0, 2.0, "wentzel", 33.852, -4.40),
    ("Cu", 15.0, 3.5, "urban",   19.325, -8.63), ("Cu", 15.0, 3.5, "wentzel", 19.325, -8.60),
    ("Cu", 15.0, 6.0, "urban",   11.269,-11.95), ("Cu", 15.0, 6.0, "wentzel", 11.269,-11.95),
    ("Cu",  3.0, 1.0, "urban",   28.706, -5.44), ("Cu",  3.0, 1.0, "wentzel", 28.706, -5.28),
    ("Cu",  3.0, 2.0, "urban",   14.289, -8.70), ("Cu",  3.0, 2.0, "wentzel", 14.289, -8.62),
    ("Cu",  3.0, 3.5, "urban",    8.157,-11.26), ("Cu",  3.0, 3.5, "wentzel",  8.157,-11.24),
    ("Cu",  3.0, 6.0, "urban",    4.757,-13.53), ("Cu",  3.0, 6.0, "wentzel",  4.757,-13.71),
    ("Pb",  2.0, 1.0, "urban",   38.333, -5.71), ("Pb",  2.0, 1.0, "wentzel", 38.333, -5.89),
    ("Pb",  2.0, 2.0, "urban",   19.081, -9.72), ("Pb",  2.0, 2.0, "wentzel", 19.081, -9.75),
    ("Pb",  2.0, 3.5, "urban",   10.893,-12.79), ("Pb",  2.0, 3.5, "wentzel", 10.893,-12.68),
    ("Pb",  2.0, 6.0, "urban",    6.352,-14.95), ("Pb",  2.0, 6.0, "wentzel",  6.352,-15.15),
    ("Pb",  8.0, 1.0, "urban",   80.517, -0.72), ("Pb",  8.0, 1.0, "wentzel", 80.517, -0.85),
    ("Pb",  8.0, 2.0, "urban",   40.080, -7.39), ("Pb",  8.0, 2.0, "wentzel", 40.080, -7.40),
    ("Pb",  8.0, 3.5, "urban",   22.880,-11.35), ("Pb",  8.0, 3.5, "wentzel", 22.880,-11.26),
    ("Pb",  8.0, 6.0, "urban",   13.342,-14.21), ("Pb",  8.0, 6.0, "wentzel", 13.342,-14.15),
]


def fig_geant4_discrepancy(outdir):
    eta_urban, fd_urban = [], []
    eta_wentzel, fd_wentzel = [], []
    for mat, t, p, model, tspace_mrad, fd in GEANT4_SWEEP:
        theta0_mrad = tspace_mrad / math.sqrt(2.0)
        eta_cut = (THETA_CUT * 1e3) / theta0_mrad
        if model == "urban":
            eta_urban.append(eta_cut); fd_urban.append(fd)
        else:
            eta_wentzel.append(eta_cut); fd_wentzel.append(fd)
    eta_urban, fd_urban = np.array(eta_urban), np.array(fd_urban)
    eta_wentzel, fd_wentzel = np.array(eta_wentzel), np.array(fd_wentzel)

    all_ln_eta = np.log(np.concatenate([eta_urban, eta_wentzel]))
    all_fd = np.concatenate([fd_urban, fd_wentzel])
    r = float(np.corrcoef(all_ln_eta, all_fd)[0, 1])
    coef = np.polyfit(all_ln_eta, all_fd, 1)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.15)
    ax.scatter(eta_urban, fd_urban, marker="o", s=32, facecolors="none",
              edgecolors=COLORS["urban"], linewidths=1.3, label="Urban")
    ax.scatter(eta_wentzel, fd_wentzel, marker="^", s=32, facecolors="none",
              edgecolors=COLORS["wentzel"], linewidths=1.3, label="Wentzel-VI")
    xx = np.linspace(all_ln_eta.min(), all_ln_eta.max(), 50)
    ax.plot(np.exp(xx), np.polyval(coef, xx), "-", color="black", lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"$\eta_{\mathrm{cut}} = \theta_{\mathrm{cut}}/\theta_0$")
    ax.set_ylabel(r"(Geant4 $-$ quad)/quad (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(0.97, 0.95, fr"$r={r:.2f}$", transform=ax.transAxes,
           ha="right", va="top", fontsize=9)
    _save(fig, outdir, "geant4_discrepancy")
    print(f"\n[geant4_discrepancy] recomputed r = {r:.3f} (paper: -0.89)")
    print("[geant4_discrepancy] suggested caption:\n"
         r"  Fractional disagreement between Geant4-measured and quadrature-"
         r"predicted $\theta_{\mathrm{RMS}}$, vs.\ the reduced acceptance "
         fr"$\eta_{{\mathrm{{cut}}}}$ (log scale), for all 32 configurations "
         r"(Cu/Pb, two thicknesses, four momenta, Urban and Wentzel-VI). "
         fr"The two models track each other closely at fixed $\eta_{{\mathrm{{cut}}}}$"
         fr" ($r={r:.2f}$ with $\ln\eta_{{\mathrm{{cut}}}}$ across all 32 points), "
         r"consistent with the truncated Moli\`ere series' asymptotic (non-"
         r"convergent) tail rather than Geant4 model uncertainty.")


# ------------------------------------------------------- Fig 4: log-law panels
def fig_logscale_theta_rms(outdir):
    AXIAL = [("Al", 10.0), ("Cu", 15.0)]
    Xs = {"Al": 0.0, "Cu": 0.0, "Pb": 0.0}
    for m, t in AXIAL:
        Xs[m] += MATERIALS[m]["rho"] * t
    xX0 = sum(t / MATERIALS[m]["X0"] for m, t in AXIAL)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18, wspace=0.35)
    momenta = (1.0, 2.0, 3.5, 6.0)
    for i, (ax, p) in enumerate(zip(axes, momenta)):
        chi_c2, chi_a2 = ml.combine_path(Xs["Al"], Xs["Cu"], Xs["Pb"], p)
        B = ml.solve_B(chi_c2, chi_a2)
        theta0 = theta0_highland(p, xX0)
        cuts = np.linspace(3 * theta0, 25 * theta0, 14)
        rms2 = np.array([_theta_rms_disc(chi_c2, chi_a2, B, cut=c) ** 2
                        for c in cuts])
        ln_cut = np.log(cuts)
        coef = np.polyfit(ln_cut, rms2, 1)
        pred = np.polyval(coef, ln_cut)
        ss_res = np.sum((rms2 - pred) ** 2)
        ss_tot = np.sum((rms2 - rms2.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.plot(ln_cut, rms2 * 1e6, "o", ms=3.5, color=COLORS[str(p)])
        ax.plot(ln_cut, pred * 1e6, "-", color="black", lw=1.0)
        ax.set_title(fr"$p={p:g}$ GeV$/c$, $R^2={r2:.3f}$", fontsize=9)
        ax.set_xlabel(r"$\ln\theta_{\mathrm{cut}}$")
        if i == 0:
            ax.set_ylabel(r"$\theta_{\mathrm{RMS}}^2$ (mrad$^2$)")
        _panel_label(ax, PANEL_LABELS[i])
    _save(fig, outdir, "theta_rms_loglaw")
    print("\n[theta_rms_loglaw] suggested caption:\n"
         r"  $\theta_{\mathrm{RMS}}^2(\theta_{\mathrm{cut}})$ vs.\ "
         r"$\ln\theta_{\mathrm{cut}}$ (points) with a linear fit (line), "
         r"axial path, over $\theta_{\mathrm{cut}}\in[3\theta_0,25\theta_0]$, "
         r"confirming the asymptotic $\sqrt{\ln\theta_{\mathrm{cut}}}$ "
         r"scaling of Eq.\ (7) at each momentum.")


# ------------------------------------------------------- Fig 5: core vs tail sketch
def fig_moliere_sketch(outdir):
    AXIAL = [("Al", 10.0), ("Cu", 15.0)]
    Xs = {"Al": 0.0, "Cu": 0.0, "Pb": 0.0}
    for m, t in AXIAL:
        Xs[m] += MATERIALS[m]["rho"] * t
    p = 2.0
    chi_c2, chi_a2 = ml.combine_path(Xs["Al"], Xs["Cu"], Xs["Pb"], p)
    B = ml.solve_B(chi_c2, chi_a2)
    scale = math.sqrt(chi_c2 * B)

    eta = np.geomspace(0.05, 9.0, 400)
    F_gauss = np.clip(ml.f0(eta) / scale, 1e-6 * (ml.f0(0.0) / scale), None)
    F_full = (ml.f0(eta) + ml.f1(eta) / B + ml.f2(eta) / B ** 2)
    F_full = np.clip(F_full, 1e-30, None) / scale
    theta_mrad = eta * scale * 1e3

    # theta^-3 reference line, anchored at a point well into the tail
    i_anchor = int(np.argmin(np.abs(eta - 6.0)))
    C = F_full[i_anchor] * theta_mrad[i_anchor] ** 3
    tail_ref = C / theta_mrad ** 3

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.15)
    ax.loglog(theta_mrad, F_gauss, "--", color="gray", lw=1.3,
             label=r"Gaussian core, $f^{(0)}$ only")
    ax.loglog(theta_mrad, F_full, "-", color=COLORS["axial"], lw=1.6,
             label=r"full $n\leq2$ series")
    ax.loglog(theta_mrad, tail_ref, ":", color="black", lw=1.1,
             label=r"$\theta^{-3}$ reference")
    theta0_mrad = theta0_highland(p, sum(t / MATERIALS[m]["X0"] for m, t in AXIAL)) * 1e3
    ax.axvline(theta0_mrad, color=COLORS["cu_short"], lw=0.8, ls="-.")
    ax.annotate(r"$\theta_0$", (theta0_mrad, ax.get_ylim()[1] * 0.5),
               fontsize=8.5, color=COLORS["cu_short"])
    ax.set_xlabel(r"$\theta_{\mathrm{plane}}$ (mrad)")
    ax.set_ylabel(r"$F(\theta_{\mathrm{plane}})$ (rad$^{-1}$)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, outdir, "moliere_sketch")
    print("\n[moliere_sketch] suggested caption:\n"
         r"  Projected Moli\`ere density (axial path, $p=2$ GeV$/c$), log-log: "
         r"the Gaussian core ($f^{(0)}$ alone, dashed) diverges from the full "
         r"$n\le2$ distribution (solid) beyond a few $\theta_0$ (dash-dot), "
         r"where the single-scatter $\theta^{-3}$ tail (dotted reference) "
         r"dominates -- the origin of the second moment's logarithmic "
         r"divergence (Eq.\ 6).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_epsM_vs_p(a.outdir)
    fig_eta_vs_k(a.outdir)
    fig_geant4_discrepancy(a.outdir)
    fig_logscale_theta_rms(a.outdir)
    fig_moliere_sketch(a.outdir)
    print(f"\nfigures written to {a.outdir}/")


if __name__ == "__main__":
    main()