#!/usr/bin/env python3
"""Independent validation and sensitivity studies requested for revision 15."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

from analysis import (
    PATHS,
    _rms,
    build_weights,
    fiducial_voxel_mask,
    image_from_events,
    optimal_global_scale,
    path_length_error_diagnostics,
    valid_voxel_mask,
)
from config import (
    CUT_CACHE_STEP,
    MIN_VOX_COUNT,
    MOMENTA,
    P_CACHE_STEP,
    SEG_CACHE_STEP,
    THETA_CUT,
)
from physics import (
    Layer,
    PofxCache,
    calibrate_pofx,
    constant_calibration,
    dimensionless_moments_quad,
    mu2_eta,
    radial_eta_from_uniform,
    reduced_parameters,
)
from simulation import load_events, segment_matrix, simulate_fixed_node


AXIAL_SEGMENT = np.array([5.0, 15.0, 0.0, 0.0, 5.0])
AXIAL_PATH = (Layer("Al", 5.0), Layer("Cu", 15.0), Layer("Al", 5.0))
PB_CROSSING_PATH = (Layer("Al", 5.0), Layer("Pb", 15.0), Layer("Al", 5.0))


def _save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def quadrature_validation(outdir, n_mc=10_000_000, seed=20260823, chunk=1_000_000):
    """Compare production-grid, adaptive-quadrature, and direct-MC moments."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(int(seed))
    for p in MOMENTA:
        r = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=2)
        cuts = {"eta10": 10.0, "200mrad": r["eta_cut"]}
        acc = {name: [0, 0.0, 0.0] for name in cuts}
        remaining = int(n_mc)
        while remaining:
            n = min(int(chunk), remaining)
            eta = radial_eta_from_uniform(r["B"], rng.random(n), nmax=2)
            eta2 = eta * eta
            for name, eta_cut in cuts.items():
                keep = eta < eta_cut
                q = eta2[keep]
                acc[name][0] += int(q.size)
                acc[name][1] += float(np.sum(q))
                acc[name][2] += float(np.sum(q * q))
            remaining -= n
        for name, eta_cut in cuts.items():
            mass, n2, n4, emass, en2, en4 = dimensionless_moments_quad(
                eta_cut, r["B"], nmax=2
            )
            mu_quad = n2 / mass
            mu_prod = mu2_eta(eta_cut, r["B"], nmax=2)
            n_acc, s2, s4 = acc[name]
            mu_mc = s2 / n_acc
            mu4_mc = s4 / n_acc
            se_mc = math.sqrt(max(mu4_mc - mu_mc * mu_mc, 0.0) / n_acc)
            rows.append(
                dict(
                    p_GeV=p,
                    cut_case=name,
                    eta_cut=eta_cut,
                    B=r["B"],
                    n_mc=int(n_mc),
                    n_accepted=n_acc,
                    mu2_production=mu_prod,
                    mu2_quad=mu_quad,
                    mu2_mc=mu_mc,
                    mc_se=se_mc,
                    production_vs_quad_rel=mu_prod / mu_quad - 1.0,
                    mc_vs_quad_rel=mu_mc / mu_quad - 1.0,
                    mc_z=(mu_mc - mu_quad) / se_mc if se_mc else np.nan,
                    quad_reported_error=math.hypot(en2 / mass, n2 * emass / mass**2),
                    pass_production_quad=abs(mu_prod / mu_quad - 1.0) < 0.001,
                    pass_mc_quad=abs(mu_mc / mu_quad - 1.0) < 0.001,
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "quadrature_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(result))
    ax.axhspan(-0.1, 0.1, color="0.92", label="0.1% criterion")
    ax.plot(x, 100 * result.production_vs_quad_rel, "o", label="grid vs adaptive")
    ax.errorbar(
        x,
        100 * result.mc_vs_quad_rel,
        yerr=100 * result.mc_se / result.mu2_quad,
        fmt="s",
        capsize=2,
        label="direct MC vs adaptive",
    )
    ax.set_xticks(
        x,
        [f"{p:g}\n{c}" for p, c in zip(result.p_GeV, result.cut_case)],
    )
    ax.set_ylabel("relative discrepancy (%)")
    ax.set_xlabel("momentum (GeV/c) and cut")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out / "quadrature_validation_panel_c")
    return result


def raw_sampler_overlay(outdir, n_mc=2_000_000, seed=20260824):
    """Independent pre-reconstruction sampler points for the collapse curve."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(MOMENTA):
        r = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=2)
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        eta = radial_eta_from_uniform(r["B"], rng.random(int(n_mc)), nmax=2)
        theta = math.sqrt(r["chi_c2"] * r["B"]) * eta
        accepted = theta < THETA_CUT
        q = theta[accepted] ** 2
        theta_rms = math.sqrt(float(np.mean(q)))
        epsilon_mc = theta_rms / r["theta_space"] - 1.0
        epsilon_se = (1.0 + epsilon_mc) / (2.0 * math.sqrt(q.size))
        rows.append(
            dict(
                p_GeV=p,
                n_generated=int(n_mc),
                n_accepted=int(q.size),
                eta_cut=r["eta_cut"],
                theta_rms_mc=theta_rms,
                theta_rms_analytic=r["theta_rms"],
                epsilon_mc=epsilon_mc,
                epsilon_analytic=r["epsilon"],
                epsilon_se_approx=epsilon_se,
                relative_rms_offset=theta_rms / r["theta_rms"] - 1.0,
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "raw_sampler_overlay.csv", index=False)

    rp = reduced_parameters(PATHS["AlCu"], 1.0)
    eta_grid = np.geomspace(2.0, 30.0, 240)
    curve = np.array(
        [math.sqrt(rp["R"] * rp["B"] * mu2_eta(e, rp["B"], 2)) - 1 for e in eta_grid]
    )
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.plot(eta_grid, 100 * curve, label="analytic reduced curve")
    ax.errorbar(
        result.eta_cut,
        100 * result.epsilon_mc,
        yerr=100 * result.epsilon_se_approx,
        fmt="o",
        capsize=2,
        label="raw sampled true angles",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_{\rm cut}$")
    ax.set_ylabel(r"$\epsilon_M$ (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, out / "theory_collapse_mc_overlay")
    return result


def _central_artifact(df, cache, theta_cut=THETA_CUT, min_count=MIN_VOX_COUNT):
    use = df[df.dth_reco.to_numpy(float) <= theta_cut].reset_index(drop=True)
    weights, _ = build_weights(use, cache=cache, theta_cut=theta_cut)
    images = {}
    counts = None
    for name in ("I_nom", "I_p", "I_Q"):
        images[name], counts = image_from_events(use, weights[name])
    valid = valid_voxel_mask(
        counts, images["I_nom"], images["I_Q"], min_count=min_count
    )
    c_opt = optimal_global_scale(images["I_nom"], images["I_Q"], valid)
    nom = _rms((images["I_nom"] - images["I_Q"])[valid])
    scale = _rms((images["I_nom"] / c_opt - images["I_Q"])[valid])
    pres = _rms((images["I_p"] - images["I_Q"])[valid])
    return dict(
        artifact_rms=nom,
        c_opt=c_opt,
        scale_opt_residual_rms=scale,
        p_residual_rms=pres,
        scale_opt_reduction=1.0 - scale / nom,
        p_reduction=1.0 - pres / nom,
        post_scalar_p_reduction=1.0 - pres / scale,
        n_valid_voxels=int(np.sum(valid)),
        n_accepted=len(use),
    )


def cache_sensitivity(outdir, n_events=500_000, seed=0):
    """Common-random-number coarse/fine cache comparison at 1 GeV/c."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "production": dict(
            p_step=P_CACHE_STEP,
            segment_step=SEG_CACHE_STEP,
            cut_step=CUT_CACHE_STEP,
        ),
        "fine": dict(
            p_step=P_CACHE_STEP / 5.0,
            segment_step=SEG_CACHE_STEP / 5.0,
            cut_step=CUT_CACHE_STEP / 4.0,
        ),
    }
    rows = []
    for label, kwargs in settings.items():
        cache = PofxCache(nmax=2, **kwargs)
        df = simulate_fixed_node(
            1.0,
            int(n_events),
            center_xy=(0.0, 0.0),
            seed=int(seed),
            calibrator=cache,
            reference_target=True,
            theta_cut=THETA_CUT,
            n_kinks=1,
        )
        art = _central_artifact(df, cache)
        central = cache.calibration(1.0, AXIAL_SEGMENT, THETA_CUT)
        rows.append(
            dict(
                cache=label,
                n_generated=len(df),
                epsilon_matched=central["epsilon_matched"],
                p_step=cache.p_step,
                segment_step=cache.segment_step,
                cut_step=cache.cut_step,
                **art,
            )
        )
    result = pd.DataFrame(rows)
    coarse, fine = result.set_index("cache").loc[["production", "fine"]].to_dict("index").values()
    comparison = pd.DataFrame(
        [
            dict(
                metric=metric,
                production=coarse[metric],
                fine=fine[metric],
                relative_shift=fine[metric] / coarse[metric] - 1.0,
                pass_0p4pct=abs(fine[metric] / coarse[metric] - 1.0) < 0.004,
            )
            for metric in ("artifact_rms", "epsilon_matched")
        ]
    )
    result.to_csv(out / "cache_sensitivity_runs.csv", index=False)
    comparison.to_csv(out / "cache_sensitivity.csv", index=False)
    return result, comparison


def occupancy_threshold(theta_cut, baseline_cut=THETA_CUT, baseline_count=MIN_VOX_COUNT):
    def worst_variance(cut):
        vals = []
        for p in MOMENTA:
            r = calibrate_pofx(AXIAL_PATH, p, cut, nmax=2)
            vals.append(r["M4"] / r["M2"] ** 2 - 1.0)
        return max(vals), vals

    target, _ = worst_variance(baseline_cut)
    current, values = worst_variance(theta_cut)
    needed = max(MIN_VOX_COUNT, int(math.ceil(baseline_count * current / target)))
    return needed, values, target


def cut_sweep(event_files, outdir):
    """Re-analyse stored raw events at 100, 150, 200, and 300 mrad."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cuts = (0.100, 0.150, 0.200, 0.300)
    rows = []
    occ_rows = []
    cut_settings = {}
    for cut in cuts:
        min_count, ratios, baseline_var = occupancy_threshold(cut)
        cut_settings[cut] = min_count
        for p, ratio in zip(MOMENTA, ratios):
            occ_rows.append(
                dict(
                    theta_cut_mrad=1000 * cut,
                    p_GeV=p,
                    M4_over_M2sq=ratio + 1.0,
                    sigma_w_over_sqrtN_at_20=math.sqrt(ratio / 20.0),
                    selected_min_count=min_count,
                    baseline_worst_variance=baseline_var,
                )
            )
    # Reuse each seed's cut-independent p(X) path cache across all four cuts.
    for file in event_files:
        df = load_events(file)
        cache = PofxCache(nmax=2)
        for cut in cuts:
            min_count = cut_settings[cut]
            row = _central_artifact(df, cache, theta_cut=cut, min_count=min_count)
            match = re.search(r"seed(\d+)", str(file))
            rows.append(
                dict(
                    seed=int(match.group(1)) if match else len(rows),
                    theta_cut_mrad=1000 * cut,
                    min_vox_count=min_count,
                    **row,
                )
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "cut_sweep_raw.csv", index=False)
    metrics = [
        "artifact_rms",
        "c_opt",
        "scale_opt_residual_rms",
        "p_residual_rms",
        "scale_opt_reduction",
        "p_reduction",
        "post_scalar_p_reduction",
    ]
    summary = raw.groupby("theta_cut_mrad", as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_sd": (m, "std") for m in metrics},
        n_seeds=("seed", "count"),
        min_vox_count=("min_vox_count", "first"),
    )
    summary.to_csv(out / "cut_sweep_summary.csv", index=False)
    pd.DataFrame(occ_rows).to_csv(out / "cut_occupancy_check.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.errorbar(
        summary.theta_cut_mrad,
        100 * summary.post_scalar_p_reduction_mean,
        yerr=100 * summary.post_scalar_p_reduction_sd,
        fmt="o-",
        capsize=2,
    )
    ax.set_xlabel(r"$\theta_{\rm cut}$ (mrad)")
    ax.set_ylabel("incremental post-scalar reduction (%)")
    fig.tight_layout()
    _save_figure(fig, out / "cut_dependence_incremental")
    return raw, summary


def combine_cut_sweeps(indirs, outdir):
    """Combine independently run per-seed cut sweeps."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(
        [pd.read_csv(Path(d) / "cut_sweep_raw.csv") for d in indirs],
        ignore_index=True,
    )
    raw.to_csv(out / "cut_sweep_raw.csv", index=False)
    metrics = [
        "artifact_rms",
        "c_opt",
        "scale_opt_residual_rms",
        "p_residual_rms",
        "scale_opt_reduction",
        "p_reduction",
        "post_scalar_p_reduction",
    ]
    summary = raw.groupby("theta_cut_mrad", as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_sd": (m, "std") for m in metrics},
        n_seeds=("seed", "count"),
        min_vox_count=("min_vox_count", "first"),
    )
    summary.to_csv(out / "cut_sweep_summary.csv", index=False)
    occ = pd.read_csv(Path(indirs[0]) / "cut_occupancy_check.csv")
    occ.to_csv(out / "cut_occupancy_check.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.errorbar(
        summary.theta_cut_mrad,
        100 * summary.post_scalar_p_reduction_mean,
        yerr=100 * summary.post_scalar_p_reduction_sd,
        fmt="o-",
        capsize=2,
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\theta_{\rm cut}$ (mrad)")
    ax.set_ylabel("incremental post-scalar reduction (%)")
    fig.tight_layout()
    _save_figure(fig, out / "cut_dependence_incremental")
    return raw, summary


def path_length_study(event_files, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, file in enumerate(event_files):
        d = path_length_error_diagnostics(load_events(file))
        match = re.search(r"seed(\d+)", str(file))
        d["seed"] = int(match.group(1)) if match else i
        rows.append(d)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(out / "path_length_error_raw.csv", index=False)
    summary = (
        raw.groupby(["comparison", "bin"], as_index=False)
        .agg(
            x_true_mean=("x_true_mean", "mean"),
            rms_x_error_mean=("rms_x_error", "mean"),
            rms_x_error_sd=("rms_x_error", "std"),
            epsilon_path_rms_mean=("epsilon_path_rms", "mean"),
            epsilon_path_rms_sd=("epsilon_path_rms", "std"),
            n_seeds=("seed", "count"),
        )
    )
    summary.to_csv(out / "path_length_error_summary.csv", index=False)
    return raw, summary


def screening_pb(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in (("AlCu", AXIAL_PATH), ("AlPb", PB_CROSSING_PATH)):
        for p in MOMENTA:
            d = calibrate_pofx(path, p, THETA_CUT, screening_weight="dchi_c2")
            s = calibrate_pofx(path, p, THETA_CUT, screening_weight="serial")
            rows.append(
                dict(
                    path=label,
                    p_GeV=p,
                    epsilon_matched_dchi=d["epsilon_matched"],
                    epsilon_matched_serial=s["epsilon_matched"],
                    epsilon_matched_spread_pp=100
                    * abs(d["epsilon_matched"] - s["epsilon_matched"]),
                    epsilon_mixed_dchi=d["epsilon_mixed"],
                    epsilon_mixed_serial=s["epsilon_mixed"],
                    epsilon_mixed_spread_pp=100
                    * abs(d["epsilon_mixed"] - s["epsilon_mixed"]),
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "screening_log_pb.csv", index=False)
    return result


def summarize_kinks(result_dirs, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in result_dirs:
        directory = Path(directory)
        a = pd.read_csv(directory / "artifact_summary.csv").iloc[0]
        cal = pd.read_csv(directory / "calibration_summary.csv").iloc[0]
        match_seed = re.search(r"seed(\d+)", str(directory))
        match_kink = re.search(r"kink(\d+)", str(directory))
        rows.append(
            dict(
                directory=str(directory),
                seed=int(match_seed.group(1)) if match_seed else -1,
                n_kinks=int(match_kink.group(1)) if match_kink else 1,
                local_kink_fallbacks=int(cal.get("local_kink_fallbacks", 0)),
                **a.to_dict(),
            )
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "kink_sensitivity_raw.csv", index=False)
    metrics = [
        "artifact_rms",
        "c_opt",
        "scale_opt_residual_rms",
        "p_residual_rms",
        "post_scalar_p_reduction",
    ]
    summary = raw.groupby("n_kinks", as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_sd": (m, "std") for m in metrics},
        n_seeds=("seed", "count"),
        local_kink_fallbacks=("local_kink_fallbacks", "sum"),
    )
    summary.to_csv(out / "kink_sensitivity_summary.csv", index=False)
    if {5, 25}.issubset(set(summary.n_kinks)):
        a = summary.set_index("n_kinks")
        checks = []
        for metric in ("artifact_rms", "scale_opt_residual_rms", "post_scalar_p_reduction"):
            mean5 = a.loc[5, f"{metric}_mean"]
            mean25 = a.loc[25, f"{metric}_mean"]
            se = math.sqrt(
                a.loc[5, f"{metric}_sd"] ** 2 / a.loc[5, "n_seeds"]
                + a.loc[25, f"{metric}_sd"] ** 2 / a.loc[25, "n_seeds"]
            )
            checks.append(
                dict(
                    metric=metric,
                    n5=mean5,
                    n25=mean25,
                    absolute_difference=abs(mean25 - mean5),
                    combined_seed_standard_error=se,
                    pass_within_seed_error=abs(mean25 - mean5) <= se,
                )
            )
        c5 = a.loc[5, "c_opt_mean"]
        c25 = a.loc[25, "c_opt_mean"]
        checks.append(
            dict(
                metric="c_opt_relative_shift",
                n5=c5,
                n25=c25,
                absolute_difference=abs(c25 / c5 - 1.0),
                combined_seed_standard_error=np.nan,
                pass_within_seed_error=abs(c25 / c5 - 1.0) <= 0.005,
            )
        )
        pd.DataFrame(checks).to_csv(out / "kink_convergence.csv", index=False)
    return raw, summary


def kink_composition_check(
    outdir,
    n_events=200_000,
    seed=20260824,
    kink_counts=(1, 2, 5, 25),
    form_factor="gaussian",
):
    """Check whether sliced-kink total angles reproduce the N=1 generator."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in MOMENTA:
        samples = {}
        for n_kinks in kink_counts:
            cache = PofxCache(nmax=2, form_factor=form_factor)
            p_values = np.full(int(n_events), p)
            segments = np.tile(AXIAL_SEGMENT, (int(n_events), 1))
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(round(1000 * p)), int(n_kinks)])
            )
            tx, ty, _ = cache.sample_kinks(
                p_values,
                segments,
                rng,
                n_kinks=int(n_kinks),
                cut=THETA_CUT,
            )
            samples[int(n_kinks)] = np.hypot(tx.sum(axis=1), ty.sum(axis=1))
        base = samples[1]
        for n_kinks in kink_counts:
            theta = samples[int(n_kinks)]
            keep0 = base <= THETA_CUT
            keep = theta <= THETA_CUT
            q0 = base[keep0] ** 2
            q = theta[keep] ** 2
            m20, m2 = float(q0.mean()), float(q.mean())
            se0 = math.sqrt(float(q0.var(ddof=1)) / q0.size)
            se = math.sqrt(float(q.var(ddof=1)) / q.size)
            combined = math.hypot(se0, se)
            ks = ks_2samp(base, theta, method="asymp").statistic
            ks99 = 1.63 * math.sqrt(2.0 / len(theta))
            rows.append(
                dict(
                    p_GeV=p,
                    n_kinks=int(n_kinks),
                    n_events=int(n_events),
                    acceptance=float(keep.mean()),
                    acceptance_n1=float(keep0.mean()),
                    accepted_M2=m2,
                    accepted_M2_n1=m20,
                    M2_difference=m2 - m20,
                    M2_combined_se=combined,
                    M2_z=(m2 - m20) / combined if combined else np.nan,
                    ks_statistic=ks,
                    ks_99pct_limit=ks99,
                    pass_M2_sampling_error=abs(m2 - m20) <= 3.0 * combined,
                    pass_KS_sampling_error=ks <= ks99,
                    form_factor=form_factor,
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "kink_composition_check.csv", index=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("quadrature")
    p.add_argument("--out", default="out/validation/quadrature")
    p.add_argument("--n-mc", type=int, default=10_000_000)
    p = sub.add_parser("raw-mc")
    p.add_argument("--out", default="out/validation/raw_mc")
    p.add_argument("--n-mc", type=int, default=2_000_000)
    p = sub.add_parser("cache")
    p.add_argument("--out", default="out/validation/cache")
    p.add_argument("--n-events", type=int, default=500_000)
    p = sub.add_parser("cut-sweep")
    p.add_argument("events", nargs="+")
    p.add_argument("--out", default="out/validation/cuts")
    p = sub.add_parser("path-length")
    p.add_argument("events", nargs="+")
    p.add_argument("--out", default="out/validation/path_length")
    p = sub.add_parser("combine-cuts")
    p.add_argument("indirs", nargs="+")
    p.add_argument("--out", default="out/validation/cuts")
    p = sub.add_parser("screening-pb")
    p.add_argument("--out", default="out/validation/screening")
    p = sub.add_parser("summarize-kinks")
    p.add_argument("result_dirs", nargs="+")
    p.add_argument("--out", default="out/validation/kinks")
    p = sub.add_parser("kink-composition")
    p.add_argument("--out", default="out/validation/kink_composition")
    p.add_argument("--n-events", type=int, default=200_000)
    p.add_argument("--form-factor", choices=("none", "gaussian", "uniform_sphere"), default="gaussian")
    a = ap.parse_args()
    if a.cmd == "quadrature":
        print(quadrature_validation(a.out, n_mc=a.n_mc).to_string(index=False))
    elif a.cmd == "raw-mc":
        print(raw_sampler_overlay(a.out, n_mc=a.n_mc).to_string(index=False))
    elif a.cmd == "cache":
        _, c = cache_sensitivity(a.out, n_events=a.n_events)
        print(c.to_string(index=False))
    elif a.cmd == "cut-sweep":
        _, s = cut_sweep(a.events, a.out)
        print(s.to_string(index=False))
    elif a.cmd == "path-length":
        _, s = path_length_study(a.events, a.out)
        print(s.to_string(index=False))
    elif a.cmd == "combine-cuts":
        _, s = combine_cut_sweeps(a.indirs, a.out)
        print(s.to_string(index=False))
    elif a.cmd == "screening-pb":
        print(screening_pb(a.out).to_string(index=False))
    elif a.cmd == "summarize-kinks":
        _, s = summarize_kinks(a.result_dirs, a.out)
        print(s.to_string(index=False))
    elif a.cmd == "kink-composition":
        print(
            kink_composition_check(
                a.out, n_events=a.n_events, form_factor=a.form_factor
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
