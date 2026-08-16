"""Focused figures for the revised manuscript.

The legacy five-near-identical-panel figure and unresolved PSF bar chart are
intentionally omitted.  These plots expose differences and reduced variables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_theory(outdir):
    out = Path(outdir)
    d = pd.read_csv(out/"theory_collapse.csv")
    fig, ax = plt.subplots(figsize=(4.3,3.3))
    # Al+Cu fixed-path curve from its p=1 reduced parameters; the four momentum
    # points should lie on it because R and B are nearly momentum invariant.
    from analysis import PATHS
    from physics import reduced_parameters, mu2_eta
    rp = reduced_parameters(PATHS["AlCu"], 1.0)
    eta = np.geomspace(2.0, 30.0, 150)
    eps = [np.sqrt(rp["R"]*rp["B"]*mu2_eta(float(e),rp["B"],2))-1 for e in eta]
    ax.plot(eta, 100*np.asarray(eps), "-")
    ax.plot(d.eta_cut, 100*d.epsilon, "o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_{\rm cut}=\theta_{\rm cut}/(\chi_c\sqrt{B})$")
    ax.set_ylabel(r"$\epsilon_M$ (%)")
    fig.tight_layout(); fig.savefig(out/"collapse_eta.pdf"); fig.savefig(out/"collapse_eta.png", dpi=300); plt.close(fig)

    e = pd.read_csv(out/"eta1_protocol.csv")
    # Deep analytic-tail windows are used only for eta1 stabilization.  Slope
    # validation is plotted only for windows at or below the numerical-table
    # boundary, with eta0 marked explicitly.
    deep = e[e.window_role == "eta1_asymptote"].copy()
    deepest_hi = deep.eta_max.max()
    deepest_lo = deep.loc[deep.eta_max == deepest_hi, "eta_min"].max()
    deep = deep[(deep.eta_min == deepest_lo) & (deep.eta_max == deepest_hi)]
    diag = e[(e.window_role == "slope_diagnostic") & (e.nmax == 2)].copy()
    diag["eta_center"] = np.sqrt(diag.eta_min * diag.eta_max)

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))
    ax = axes[0]
    for nmax, g in deep.groupby("nmax"):
        ax.plot(g.path, g.eta1_joint, "o-", label=fr"$n\leq {nmax}$")
    ax.axhline(1.0, lw=0.8, color="k")
    ax.set_ylabel(r"joint-fit $\eta_1$")
    ax.set_xlabel("path")
    ax.legend(frameon=False)

    ax = axes[1]
    for path, g in diag.groupby("path"):
        g = g.sort_values("eta_center")
        x = g.eta_center.to_numpy(float)
        lo = g.eta_min.to_numpy(float)
        hi = g.eta_max.to_numpy(float)
        y = 100.0*(g.slope_ratio.to_numpy(float)-1.0)
        ax.errorbar(x, y, xerr=np.vstack([x-lo, hi-x]), marker="o", capsize=2, label=path)
    eta0 = float(e.eta_table_max.iloc[0])
    ax.axvline(eta0, lw=0.9, ls="--", color="k", label=fr"$\eta_0={eta0:g}$")
    ax.axhline(0.0, lw=0.8, color="k")
    ax.set_xscale("log")
    ax.set_xlim(7.0, eta0*1.08)
    ax.set_ylabel(r"$(m/2R-1)$ (%)")
    ax.set_xlabel(r"fit-window $\eta$ (center and span)")
    ax.legend(frameon=False, fontsize=6.8)
    fig.tight_layout(); fig.savefig(out/"eta1_paths.pdf"); fig.savefig(out/"eta1_paths.png", dpi=300); plt.close(fig)


def _central(arr):
    return arr[:,:,arr.shape[2]//2]


def plot_images(outdir):
    out=Path(outdir); z=np.load(out/"images.npz")
    In, Ip, IQ = z["I_nom"], z["I_p"], z["I_Q"]
    maps=[In-IQ, Ip-IQ]
    labels=[r"$I_{\rm nom}-I_Q$", r"$I_p-I_Q$"]
    fig, axes=plt.subplots(1,2,figsize=(7.0,3.2))
    vmax=np.nanpercentile(np.abs(np.stack([_central(x) for x in maps])),99)
    for ax,m,l in zip(axes,maps,labels):
        im=ax.imshow(_central(m).T,origin="lower",extent=[-15,15,-15,15],vmin=-vmax,vmax=vmax,cmap="RdBu_r")
        ax.set_xlabel("x (cm)"); ax.set_title(l)
    axes[0].set_ylabel("y (cm)")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="weight difference",shrink=.85)
    fig.subplots_adjust(wspace=.15,right=.88)
    fig.savefig(out/"difference_maps.pdf"); fig.savefig(out/"difference_maps.png",dpi=300); plt.close(fig)


def plot_gradient(outdir):
    out=Path(outdir); z=np.load(out/"gradient_maps.npz")
    names=["observed","predicted_unweighted","residual_unweighted","residual_weighted"]
    labels=[r"observed $I_{\rm nom}-I_Q$",
            "mean normalization-field predictor",
            "mechanism-fit residual",
            r"$w_Q$-weighted closure residual"]
    fig,axes=plt.subplots(1,4,figsize=(13.0,3.1))
    vals=[_central(z[n]) for n in names]
    for ax,v,l in zip(axes,vals,labels):
        finite=np.abs(v[np.isfinite(v)])
        vmax=np.nanpercentile(finite,99) if finite.size else 1.0
        vmax=max(float(vmax),1e-12)
        im=ax.imshow(v.T,origin="lower",extent=[-15,15,-15,15],
                     vmin=-vmax,vmax=vmax,cmap="RdBu_r")
        ax.set_xlabel("x (cm)"); ax.set_title(l,fontsize=8.5)
        fig.colorbar(im,ax=ax,shrink=.76)
    axes[0].set_ylabel("y (cm)")
    fig.subplots_adjust(wspace=.38)
    fig.savefig(out/"gradient_causal_maps.pdf",bbox_inches="tight")
    fig.savefig(out/"gradient_causal_maps.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("outdir"); ap.add_argument("--kind",choices=["theory","images","gradient","all"],default="all")
    a=ap.parse_args()
    if a.kind in ("theory","all") and (Path(a.outdir)/"theory_collapse.csv").exists(): plot_theory(a.outdir)
    if a.kind in ("images","all") and (Path(a.outdir)/"images.npz").exists(): plot_images(a.outdir)
    if a.kind in ("gradient","all") and (Path(a.outdir)/"gradient_maps.npz").exists(): plot_gradient(a.outdir)

if __name__=="__main__": main()
