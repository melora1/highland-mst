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
from matplotlib.ticker import NullFormatter
from scipy.stats import ks_2samp, kstest
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
    MATERIALS,
    MIN_VOX_COUNT,
    MOMENTA,
    P_CACHE_STEP,
    SEG_CACHE_STEP,
    THETA_CUT,
)
from physics import (
    Layer,
    PofxCache,
    beta_of,
    calibrate_pofx,
    calibrate_pofx_transform,
    composition_scan,
    constant_calibration,
    dimensionless_moments_quad,
    efficiency_scan,
    finite_size_rho,
    finite_size_eta_from_uniform,
    finite_size_kernel,
    mu2_eta,
    radial_eta_from_uniform,
    reduced_parameters,
    _rms_untruncated_diagnostics,
    tail_ratio_scan,
    transform_moments_g1,
    transform_radial_density,
    untruncated_finite_size_moments,
)
from sampling import TransformSampler
from simulation import load_events, segment_matrix, simulate_fixed_node


AXIAL_SEGMENT = np.array([5.0, 15.0, 0.0, 0.0, 5.0])
AXIAL_PATH = (Layer("Al", 5.0), Layer("Cu", 15.0), Layer("Al", 5.0))
PB_CROSSING_PATH = (Layer("Al", 5.0), Layer("Pb", 15.0), Layer("Al", 5.0))
SCAN_PATHS = ("Al25", "Cu15", "AlCu", "Pb15")
SCAN_FF_MODELS = ("gauss", "sphere")
STANDARD_COLUMNS = ("ff_model", "floor", "path", "p_GeV", "theta_cut_mrad")


def _ordered_columns(frame):
    """Keep the mandatory merge keys first in every new validation CSV."""
    return frame.loc[:, [*STANDARD_COLUMNS, *[c for c in frame if c not in STANDARD_COLUMNS]]]


def task1_untruncated_rms(outdir):
    """Run Task 1 for central Al+Cu and enforce the 3-to-10 rad gate."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p_GeV in MOMENTA:
        for ff_model in SCAN_FF_MODELS:
            for floor in (True, False):
                rows.append(
                    _rms_untruncated_diagnostics(
                        "AlCu", p_GeV, ff_model, floor, 3.0
                    )
                )
    result = _ordered_columns(pd.DataFrame(rows))
    result["pass_gate_A"] = result.convergence_3_to_10 < 1.0e-3
    result.to_csv(out / "untruncated_rms.csv", index=False)
    assert bool(result.pass_gate_A.all()), "Gate A failed"
    return result


def _point_m4_gate():
    rows = efficiency_scan("AlCu", 1.0, "point", False, [200.0])
    rows += efficiency_scan("AlCu", 6.0, "point", False, [200.0])
    got = {row["p_GeV"]: row["M4_over_M2_sq"] for row in rows}
    assert abs(got[1.0] / 1.99 - 1.0) < 0.01, got
    assert abs(got[6.0] / 11.05 - 1.0) < 0.01, got
    return got


def task2_efficiency(outdir):
    """Run the registered coarse/fine efficiency scan and report true optima."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    point = _point_m4_gate()
    cuts = np.arange(50.0, 300.0 + 0.1, 5.0)
    rows = []
    for path in SCAN_PATHS:
        for p_GeV in MOMENTA:
            for ff_model in SCAN_FF_MODELS:
                for floor in (True, False):
                    rows.extend(
                        efficiency_scan(path, p_GeV, ff_model, floor, cuts)
                    )
    scan = _ordered_columns(pd.DataFrame(rows))
    scan.to_csv(out / "efficiency_scan.csv", index=False)

    optima = []
    keys = ["path", "p_GeV", "ff_model", "floor"]
    for key, group in scan.groupby(keys, sort=False):
        group = group.sort_values("theta_cut_mrad")
        values = group.efficiency.to_numpy(float)
        delta = np.diff(values)
        maxima = np.flatnonzero((delta[:-1] > 0.0) & (delta[1:] < 0.0)) + 1
        best = int(np.nanargmax(values))
        optima.append(
            dict(
                path=key[0],
                p_GeV=key[1],
                ff_model=key[2],
                floor=key[3],
                theta_cut_mrad=float(group.iloc[best].theta_cut_mrad),
                interior_maximum=bool(maxima.size),
                n_sign_change_maxima=int(maxima.size),
                maximum_locations_mrad=";".join(
                    f"{group.iloc[i].theta_cut_mrad:g}" for i in maxima
                ),
                point_M4_gate_1GeV=point[1.0],
                point_M4_gate_6GeV=point[6.0],
            )
        )
    optimum = _ordered_columns(pd.DataFrame(optima))
    optimum.to_csv(out / "efficiency_optima.csv", index=False)
    assert len(optimum) == len(SCAN_PATHS) * len(MOMENTA) * len(SCAN_FF_MODELS) * 2
    return scan, optimum


def task3_composition(outdir):
    """Write the composition scan and the weighted Al+Cu rho diagnostic."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ff_model in SCAN_FF_MODELS:
        for floor in (True, False):
            rows.extend(
                composition_scan(
                    SCAN_PATHS,
                    MOMENTA,
                    [50, 75, 100, 150, 200, 250, 300],
                    ff_model,
                    floor,
                )
            )
    result = _ordered_columns(pd.DataFrame(rows))
    result.to_csv(out / "composition_scan.csv", index=False)
    rho = result[
        (result.path == "AlCu")
        & (result.p_GeV == 1.0)
        & (result.ff_model == "gauss")
        & result.floor
        & (result.theta_cut_mrad == 200.0)
    ].iloc[0].rho
    summary = pd.DataFrame(
        [
            dict(
                ff_model="gauss",
                floor=True,
                path="AlCu",
                p_GeV=1.0,
                theta_cut_mrad=200.0,
                rho_weighted=rho,
                rho_manuscript=0.562,
                relative_difference=rho / 0.562 - 1.0,
                manuscript_update_needed=abs(rho / 0.562 - 1.0) > 0.01,
            )
        ]
    )
    summary = _ordered_columns(summary)
    summary.to_csv(out / "rho_check.csv", index=False)

    matched_rows = []
    for ff_model, internal in (("gauss", "gaussian"), ("sphere", "uniform_sphere")):
        for p_GeV in MOMENTA:
            reference = calibrate_pofx_transform(
                AXIAL_PATH, p_GeV, THETA_CUT,
                form_factor=internal, include_incoherent=True,
            )
            k_ref = THETA_CUT / (reference["theta_space_pofx"] / math.sqrt(2.0))
            eta_ref = reference["eta_cut"]
            for path in SCAN_PATHS:
                ordered = {
                    "Al25": (Layer("Al", 25.0),),
                    "Cu15": (Layer("Cu", 15.0),),
                    "AlCu": AXIAL_PATH,
                    "Pb15": (Layer("Pb", 15.0),),
                }[path]
                base = calibrate_pofx_transform(
                    ordered, p_GeV, THETA_CUT,
                    form_factor=internal, include_incoherent=True,
                )
                cuts = {
                    "matched_k": k_ref * base["theta_space_pofx"] / math.sqrt(2.0),
                    "matched_eta_cut": eta_ref * math.sqrt(base["chi_c2"] * base["B"]),
                }
                for matching, cut in cuts.items():
                    q = calibrate_pofx_transform(
                        ordered, p_GeV, cut,
                        form_factor=internal, include_incoherent=True,
                    )
                    matched_rows.append(dict(
                        ff_model=ff_model, floor=True, path=path, p_GeV=p_GeV,
                        theta_cut_mrad=1000.0*cut, matching=matching,
                        eps_M=q["epsilon_matched"], k_reference=k_ref,
                        eta_cut_reference=eta_ref,
                    ))
    matched = _ordered_columns(pd.DataFrame(matched_rows))
    matched.to_csv(out / "fig2_matched_composition.csv", index=False)

    spread_rows = []
    point_spreads = {"matched_k": 13.2, "matched_eta_cut": 19.2}
    for (ff_model, p_GeV, matching), group in matched.groupby(
        ["ff_model", "p_GeV", "matching"]
    ):
        values = group.set_index("path").eps_M
        spread_pp = 100.0 * (values["Pb15"] - values["AlCu"])
        baseline_pp = point_spreads[matching]
        if np.sign(spread_pp) != np.sign(baseline_pp):
            change = "inverts"
        elif abs(spread_pp) > abs(baseline_pp):
            change = "grows"
        else:
            change = "shrinks"
        spread_rows.append(dict(
            ff_model=ff_model, floor=True, path="Pb15/AlCu", p_GeV=p_GeV,
            theta_cut_mrad=np.nan, matching=matching, spread_pp=spread_pp,
            point_nucleus_spread_pp=baseline_pp, change=change,
        ))
    spread = _ordered_columns(pd.DataFrame(spread_rows))
    spread.to_csv(out / "fig2_spread_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, matching in zip(axes, ("matched_k", "matched_eta_cut")):
        for ff_model, style in (("gauss", "-"), ("sphere", "--")):
            for p_GeV, marker in zip(MOMENTA, ("o", "s", "^", "D")):
                q = matched[(matched.matching == matching)
                            & (matched.ff_model == ff_model)
                            & (matched.p_GeV == p_GeV)].set_index("path")
                ax.plot(
                    SCAN_PATHS, 100.0*q.loc[list(SCAN_PATHS)].eps_M,
                    linestyle=style, marker=marker, ms=3,
                    label=f"{ff_model}, {p_GeV:g} GeV/c",
                )
        ax.set_title(matching.replace("_", " "))
        ax.set_xlabel("path")
    axes[0].set_ylabel(r"$\epsilon_M$ (%)")
    axes[1].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    _save_figure(fig, out / "fig2_composition_transform")
    return result, summary


def task4_tail(outdir):
    """Generate the three-regime tail table and enforce the Cu floor gate."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    angles = np.geomspace(10.0, 300.0, 180)
    rows = []
    gates = []
    for ff_model in SCAN_FF_MODELS:
        rows.extend(tail_ratio_scan("Cu15", 6.0, ff_model, True, angles))
    result = _ordered_columns(pd.DataFrame(rows))
    result.to_csv(out / "tail_ratio_scan.csv", index=False)
    for ff_model, group in result.groupby("ff_model"):
        plateau = group[
            (group.theta_mrad >= 1.2 * group.theta_nuc_mrad)
            & (group.theta_mrad <= 1.4 * group.theta_nuc_mrad)
        ]
        measured = float(plateau.tail_ratio.median())
        expected = float(plateau.expected_floor.iloc[0])
        relative = abs(measured / expected - 1.0)
        gates.append(
            dict(
                ff_model=ff_model,
                floor=True,
                path="Cu15",
                p_GeV=6.0,
                theta_cut_mrad=np.nan,
                plateau_ratio=measured,
                expected_floor=expected,
                relative_difference=relative,
                pass_gate=relative <= 0.20,
            )
        )
    gate = _ordered_columns(pd.DataFrame(gates))
    gate.to_csv(out / "tail_plateau_gate.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for ff_model, group in result.groupby("ff_model"):
        ax.plot(group.theta_mrad, group.tail_ratio, label=ff_model)
    first = result.iloc[0]
    ax.axvline(first.theta_FF_mrad, color="0.4", ls="--", label=r"$\theta_{FF}$")
    ax.axvline(first.theta_nuc_mrad, color="0.4", ls=":", label=r"$\theta_{nuc}$")
    ax.axhline(first.expected_floor, color="0.6", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel=r"$\Theta$ (mrad)",
           ylabel=r"$h(\Theta)\Theta^3/(2\chi_c^2)$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out / "three_regime_tail")
    assert bool(gate.pass_gate.all()), "Task 4 incoherent-floor plateau gate failed"
    return result, gate


def transform_g1_closure(outdir, threshold_pp=0.05):
    """Step A.4 gate: exact screened transform versus radial n<=2 Moliere."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in MOMENTA:
        reference = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=2)
        Fc, M2, M4 = transform_moments_g1(
            reference["chi_c2"], reference["B"], THETA_CUT
        )
        epsilon = math.sqrt(M2) / reference["theta_space"] - 1.0
        delta_pp = 100.0 * (epsilon - reference["epsilon"])
        rows.append(
            dict(
                p_GeV=p,
                theta_cut_mrad=1000.0 * THETA_CUT,
                B=reference["B"],
                epsilon_moliere_n2=reference["epsilon"],
                epsilon_transform_g1=epsilon,
                epsilon_difference_pp=delta_pp,
                absolute_difference_pp=abs(delta_pp),
                Fc_moliere_n2=reference["Fc"],
                Fc_transform_g1=Fc,
                M2_moliere_n2=reference["M2"],
                M2_transform_g1=M2,
                M4_transform_g1=M4,
                threshold_pp=float(threshold_pp),
                pass_gate=abs(delta_pp) < float(threshold_pp),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "transform_g1_closure.csv", index=False)
    if not bool(result.pass_gate.all()):
        raise RuntimeError(
            "Step A.4 transform closure failed; finite-size regeneration is blocked"
        )
    return result


def finite_size_transform_regeneration(outdir):
    """Regenerate the Step A.5/B transform tables after enforcing A.4."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    closure = transform_g1_closure(out / "closure")
    if not bool(closure.pass_gate.all()):
        raise RuntimeError("A.4 closure did not pass")

    paths = {
        "Al25": (Layer("Al", 25.0),),
        "Cu15": (Layer("Cu", 15.0),),
        "AlCu": AXIAL_PATH,
        "Pb15": (Layer("Pb", 15.0),),
    }
    cuts = (0.050, 0.100, 0.150, 0.200, 0.300)
    rows = []
    for path_name, path in paths.items():
        for p in MOMENTA:
            for cut in cuts:
                point = calibrate_pofx_transform(
                    path, p, cut, form_factor="none"
                )
                rows.append(dict(path=path_name, p_GeV=p, theta_cut_mrad=1000*cut, **{
                    k: point[k] for k in (
                        "form_factor", "include_incoherent", "Fc", "M2", "M4",
                        "mu2", "epsilon_matched", "epsilon_mixed",
                        "ratio2_matched", "ratio2_mixed", "R_matched", "R_mixed",
                        "B", "eta_cut",
                        "p_out", "dp_over_p",
                    )
                }))
                for model in ("gaussian", "uniform_sphere"):
                    for floor in (True, False):
                        value = calibrate_pofx_transform(
                            path,
                            p,
                            cut,
                            form_factor=model,
                            include_incoherent=floor,
                        )
                        rows.append(dict(path=path_name, p_GeV=p, theta_cut_mrad=1000*cut, **{
                            k: value[k] for k in (
                                "form_factor", "include_incoherent", "Fc", "M2", "M4",
                                "mu2", "epsilon_matched", "epsilon_mixed",
                                "ratio2_matched", "ratio2_mixed", "R_matched", "R_mixed",
                                "B", "eta_cut",
                                "p_out", "dp_over_p",
                            )
                        }))
    result = pd.DataFrame(rows)
    result["reduced_identity_error_matched"] = (
        result.ratio2_matched - result.R_matched * result.B * result.mu2
    )
    result["reduced_identity_error_mixed"] = (
        result.ratio2_mixed - result.R_mixed * result.B * result.mu2
    )
    result.to_csv(out / "finite_size_transform_scan.csv", index=False)

    finite = result[result.form_factor != "none"].copy()
    idx = ["path", "p_GeV", "theta_cut_mrad", "form_factor"]
    inc = finite[finite.include_incoherent].set_index(idx)
    omit = finite[~finite.include_incoherent].set_index(idx)
    systematic = inc[["epsilon_matched", "epsilon_mixed"]].join(
        omit[["epsilon_matched", "epsilon_mixed"]],
        lsuffix="_with_floor",
        rsuffix="_floor_omitted",
    ).reset_index()
    for mismatch in ("matched", "mixed"):
        systematic[f"floor_systematic_{mismatch}_pp"] = 100.0 * (
            systematic[f"epsilon_{mismatch}_with_floor"]
            - systematic[f"epsilon_{mismatch}_floor_omitted"]
        )
    systematic.to_csv(out / "incoherent_floor_systematic.csv", index=False)

    angle_rows = []
    for cut in cuts:
        angle_rows.append(dict(
            theta_mrad=1000.0 * cut,
            tan_over_theta=math.tan(cut) / cut,
            tan_relative_error=math.tan(cut) / cut - 1.0,
            exact_q_over_p_theta=2.0 * math.sin(0.5 * cut) / cut,
            q_relative_difference=2.0 * math.sin(0.5 * cut) / cut - 1.0,
            Cu_qR_over_hbarc_at_6GeV=(
                6000.0 * cut * 1.2 * MATERIALS["Cu"].A ** (1.0 / 3.0) / 197.3
            ),
            transform_q_convention="q=p*theta",
        ))
    pd.DataFrame(angle_rows).to_csv(out / "small_angle_diagnostics.csv", index=False)
    return result, systematic


def finite_size_decision_gates(outdir):
    """Evaluate the untruncated-RMS and efficiency-scan decision gates."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    gate_a_rows = []
    gate_b_rows = []
    cuts = np.linspace(0.050, 0.300, 101)
    for p in MOMENTA:
        for model in ("gaussian", "uniform_sphere"):
            base = calibrate_pofx_transform(
                AXIAL_PATH, p, THETA_CUT, form_factor=model
            )
            M2_inf, M4_inf = untruncated_finite_size_moments(
                base["chi_c2"], base["B"], base["tail_components"], model
            )
            gate_a_rows.append(dict(
                p_GeV=p,
                form_factor=model,
                theta_rms_over_theta_space=math.sqrt(M2_inf) / base["theta_space_pofx"],
                epsilon_untruncated=math.sqrt(M2_inf) / base["theta_space_pofx"] - 1.0,
                M2_untruncated=M2_inf,
                M4_untruncated=M4_inf,
                M4_over_M2_sq=M4_inf / M2_inf**2,
                mechanism_gate=abs(math.sqrt(M2_inf) / base["theta_space_pofx"] - 1.0) < 0.01,
            ))
            for cut in cuts:
                value = calibrate_pofx_transform(
                    AXIAL_PATH, p, float(cut), form_factor=model
                )
                variance = value["M4"] - value["M2"] ** 2
                efficiency = (
                    math.sqrt(value["Fc"]) * value["M2"] / math.sqrt(variance)
                    if variance > 0.0 else np.nan
                )
                gate_b_rows.append(dict(
                    p_GeV=p,
                    form_factor=model,
                    theta_cut_mrad=1000.0 * cut,
                    efficiency=efficiency,
                    Fc=value["Fc"],
                    M2=value["M2"],
                    M4=value["M4"],
                    M4_over_M2_sq=value["M4"] / value["M2"] ** 2,
                    sigma_wQ=math.sqrt(max(value["M4"] / value["M2"] ** 2 - 1.0, 0.0)),
                ))
    gate_a = pd.DataFrame(gate_a_rows)
    gate_b = pd.DataFrame(gate_b_rows)
    gate_a.to_csv(out / "gate_a_untruncated_rms.csv", index=False)
    gate_b.to_csv(out / "gate_b_efficiency_scan.csv", index=False)

    opt_rows = []
    for (p, model), group in gate_b.groupby(["p_GeV", "form_factor"], sort=False):
        group = group.sort_values("theta_cut_mrad")
        j = int(np.nanargmax(group.efficiency.to_numpy()))
        row = group.iloc[j]
        diff = np.diff(group.efficiency.to_numpy())
        opt_rows.append(dict(
            p_GeV=p,
            form_factor=model,
            theta_cut_opt_mrad=row.theta_cut_mrad,
            efficiency_opt=row.efficiency,
            interior_50_300=0 < j < len(group) - 1,
            monotonic_50_300=bool(np.all(diff >= 0.0) or np.all(diff <= 0.0)),
        ))
    optimum = pd.DataFrame(opt_rows)
    optimum.to_csv(out / "gate_b_optima.csv", index=False)

    global_rows = []
    global_cuts = np.geomspace(0.003, 0.300, 220)
    for p in MOMENTA:
        for model in ("gaussian", "uniform_sphere"):
            values = [
                calibrate_pofx_transform(AXIAL_PATH, p, float(c), form_factor=model)
                for c in global_cuts
            ]
            efficiencies = np.asarray([
                math.sqrt(v["Fc"])*v["M2"] / math.sqrt(v["M4"]-v["M2"]**2)
                for v in values
            ])
            j = int(np.argmax(efficiencies))
            global_rows.append(dict(
                p_GeV=p, form_factor=model,
                theta_cut_opt_mrad=1000.0*global_cuts[j],
                efficiency_opt=efficiencies[j],
                M4_over_M2_sq_at_opt=values[j]["M4"]/values[j]["M2"]**2,
                interior_3_300=0 < j < len(global_cuts)-1,
            ))
    pd.DataFrame(global_rows).to_csv(out / "gate_b_global_optima.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    styles = {"gaussian": "-", "uniform_sphere": "--"}
    for (p, model), group in gate_b.groupby(["p_GeV", "form_factor"]):
        label = f"{p:g} GeV/c, {model.replace('_', ' ')}"
        axes[0].plot(group.theta_cut_mrad, group.efficiency, styles[model], label=label)
        axes[1].plot(group.theta_cut_mrad, group.M4_over_M2_sq, styles[model], label=label)
    axes[0].set(xlabel=r"$\theta_{\rm cut}$ (mrad)", ylabel=r"$\mathcal{E}$")
    axes[1].set(xlabel=r"$\theta_{\rm cut}$ (mrad)", ylabel=r"$M_4/M_2^2$")
    axes[1].axhline(2.0, color="0.6", lw=0.8)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    _save_figure(fig, out / "decision_gate_efficiency")
    return gate_a, gate_b, optimum


def finite_size_analytic_completion(outdir):
    """Composition, transform collapse, and three-regime tail deliverables."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "Al25": (Layer("Al", 25.0),),
        "Cu15": (Layer("Cu", 15.0),),
        "AlCu": AXIAL_PATH,
        "Pb15": (Layer("Pb", 15.0),),
    }
    cuts = (0.050, 0.100, 0.150, 0.200, 0.300)
    rows = []
    for path_name, path in paths.items():
        for p in MOMENTA:
            for model in ("gaussian", "uniform_sphere"):
                for cut in cuts:
                    value = calibrate_pofx_transform(path, p, cut, form_factor=model)
                    rows.append(dict(
                        path=path_name, p_GeV=p, form_factor=model,
                        theta_cut_mrad=1000.0 * cut,
                        rho=finite_size_rho(value["chi_c2"], value["B"], value["tail_components"]),
                        eta_cut=value["eta_cut"], mu2=value["mu2"],
                        epsilon_matched=value["epsilon_matched"],
                        epsilon_mixed=value["epsilon_mixed"], Fc=value["Fc"],
                        M4_over_M2_sq=value["M4"] / value["M2"]**2,
                    ))
    composition = pd.DataFrame(rows)
    composition.to_csv(out / "finite_size_composition.csv", index=False)

    colors = {"Al25": "tab:blue", "Cu15": "tab:orange", "AlCu": "tab:green", "Pb15": "tab:red"}
    markers = {1.0: "o", 2.0: "s", 3.5: "^", 6.0: "D"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, model in zip(axes, ("gaussian", "uniform_sphere")):
        q = composition[composition.form_factor == model]
        for path_name in paths:
            for p in MOMENTA:
                g = q[(q.path == path_name) & (q.p_GeV == p)].sort_values("eta_cut")
                ax.plot(g.eta_cut, 100.0*g.epsilon_matched, color=colors[path_name], alpha=0.5)
                ax.scatter(g.eta_cut, 100.0*g.epsilon_matched, color=colors[path_name], marker=markers[p], s=16)
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 10, 20, 30])
        ax.set_xticklabels(["2", "3", "5", "10", "20", "30"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$\eta_{\rm cut}$")
        ax.set_title(model.replace("_", " "))
    axes[0].set_ylabel(r"self-consistent $\epsilon_M$ (%)")
    axes[1].legend(
        handles=[plt.Line2D([], [], color=colors[k], label=k) for k in paths],
        frameon=False, fontsize=8,
    )
    fig.tight_layout()
    _save_figure(fig, out / "fig2_finite_size_composition")

    eta_grid = np.geomspace(1.5, 35.0, 60)
    collapse_rows = []
    for p in MOMENTA:
        for model in ("gaussian", "uniform_sphere"):
            base = calibrate_pofx_transform(AXIAL_PATH, p, THETA_CUT, form_factor=model)
            scale = math.sqrt(base["chi_c2"] * base["B"])
            for eta in eta_grid:
                value = calibrate_pofx_transform(AXIAL_PATH, p, float(eta*scale), form_factor=model)
                collapse_rows.append(dict(
                    p_GeV=p, form_factor=model, eta_cut=eta,
                    rho=finite_size_rho(value["chi_c2"], value["B"], value["tail_components"]),
                    epsilon_matched=value["epsilon_matched"], mu2=value["mu2"],
                ))
    collapse = pd.DataFrame(collapse_rows)
    collapse.to_csv(out / "finite_size_collapse.csv", index=False)
    summary_rows = []
    beta_values = np.asarray([beta_of(p) for p in MOMENTA], float)
    for model, q in collapse.groupby("form_factor"):
        pivot = q.pivot(index="eta_cut", columns="p_GeV", values="epsilon_matched")
        rho_by_p = q.groupby("p_GeV").rho.first()
        summary_rows.append(dict(
            form_factor=model, rho_mean=rho_by_p.mean(),
            rho_peak_to_peak_fraction=(rho_by_p.max()-rho_by_p.min())/rho_by_p.mean(),
            epsilon_max_peak_to_peak_pp=100.0*(pivot.max(axis=1)-pivot.min(axis=1)).max(),
            beta_floor_fraction=(beta_values.max()-beta_values.min())/beta_values.mean(),
        ))
    collapse_summary = pd.DataFrame(summary_rows)
    collapse_summary.to_csv(out / "finite_size_collapse_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, model in zip(axes, ("gaussian", "uniform_sphere")):
        q = collapse[collapse.form_factor == model]
        for p in MOMENTA:
            g = q[q.p_GeV == p]
            ax.plot(g.eta_cut, 100.0*g.epsilon_matched, label=f"{p:g} GeV/c")
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 10, 20, 30])
        ax.set_xticklabels(["2", "3", "5", "10", "20", "30"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$\eta_{\rm cut}$")
        ax.set_title(model.replace("_", " "))
    axes[0].set_ylabel(r"self-consistent $\epsilon_M$ (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out / "fig1_transform_collapse")

    theta = np.geomspace(0.010, 0.300, 180)
    tail_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    colors_tail = {"gaussian": "tab:blue", "uniform_sphere": "tab:orange"}
    for ax, p_tail in zip(axes, (1.0, 6.0)):
        for model in ("gaussian", "uniform_sphere"):
            value = calibrate_pofx_transform(AXIAL_PATH, p_tail, THETA_CUT, form_factor=model)
            h = transform_radial_density(theta, value["chi_c2"], value["B"], value["tail_components"], model)
            ratio = h * theta**3 / (2.0 * value["chi_c2"])
            kernel = finite_size_kernel(theta, value["tail_components"], model)
            for angle, density, rr, gg in zip(theta, h, ratio, kernel):
                tail_rows.append(dict(
                    p_GeV=p_tail, form_factor=model, theta_mrad=1000.0*angle,
                    h=density, h_theta3_over_2chic2=rr, G=gg,
                ))
            ax.plot(1000.0*theta, ratio, color=colors_tail[model], label=model.replace("_", " "))
            ax.plot(1000.0*theta, kernel, color=colors_tail[model], ls="--", alpha=0.75)
        theta_ff = 1000.0 * math.exp(np.mean([
            math.log(197.3/(1000.0*p_tail*1.2*MATERIALS[m].A**(1.0/3.0)))
            for m in ("Al", "Cu")
        ]))
        theta_nuc = 1000.0 * 197.3 / (1000.0*p_tail*0.84)
        ax.axvline(theta_ff, color="0.35", ls="--", lw=0.9)
        ax.axvline(theta_nuc, color="0.35", ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Theta$ (mrad)")
        ax.set_title(f"{p_tail:g} GeV/c")
    axes[0].set_ylabel(r"$h(\Theta)\Theta^3/(2\chi_c^2)$")
    axes[0].legend(frameon=False, fontsize=8, title="solid: h ratio\ndashed: G")
    fig.tight_layout()
    _save_figure(fig, out / "three_regime_tail")
    tail = pd.DataFrame(tail_rows)
    tail.to_csv(out / "three_regime_tail.csv", index=False)
    return composition, collapse, collapse_summary, tail


def finite_size_sampler_validation(outdir, n_mc, seed, ff_model, floor):
    """Enforce the accepted-angle M2 and one-sample KS production gates."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(MOMENTA):
        sampler = TransformSampler("AlCu", p, ff_model, floor, THETA_CUT)
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        sum2 = 0.0
        sum4 = 0.0
        ks_sample = []
        remaining = int(n_mc)
        while remaining:
            n = min(1_000_000, remaining)
            theta = sampler.sample(n, rng)
            q = theta**2
            sum2 += float(np.sum(q))
            sum4 += float(np.sum(q*q))
            if sum(map(len, ks_sample)) < 200_000:
                ks_sample.append(theta[: min(n, 200_000 - sum(map(len, ks_sample)))])
            remaining -= n
        M2_mc = sum2 / n_mc
        M4_mc = sum4 / n_mc
        se = math.sqrt(max(M4_mc-M2_mc**2, 0.0)/n_mc)
        relative = M2_mc/sampler.M2 - 1.0
        ks = kstest(np.concatenate(ks_sample), sampler.cdf, method="asymp")
        rows.append(dict(
            ff_model=ff_model, floor=bool(floor), path="AlCu", p_GeV=p,
            theta_cut_mrad=1000.0*THETA_CUT, n_mc=int(n_mc),
            n_accepted=int(n_mc), accepted_only=True, Fc_transform=sampler.Fc,
            M2_transform=sampler.M2, M2_mc=M2_mc,
            M2_relative_difference=relative, M2_mc_se=se,
            M2_z=(M2_mc-sampler.M2)/se,
            ks_n=sum(map(len, ks_sample)), ks_statistic=ks.statistic,
            ks_pvalue=ks.pvalue, pass_ks=ks.pvalue > 0.01,
            target_relative=3.5e-4,
            pass_3p5e4=abs(relative) <= 3.5e-4,
        ))
    result = _ordered_columns(pd.DataFrame(rows))
    result.to_csv(out / "finite_size_sampler_validation.csv", index=False)
    assert bool(result.pass_3p5e4.all()), "Task 5 M2 sampler gate failed"
    assert bool(result.pass_ks.all()), "Task 5 KS sampler gate failed"
    return result


def geant4_finite_size_benchmark(outdir, rawdir="out/geant4/raw"):
    """Compare existing Cu/Pb transport dumps with both finite-size kernels."""
    from geant4_compare import load_angles, sample_moments

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(rawdir)
    summary_rows = []
    band_rows = []
    slabs = (("Cu", 15.0), ("Pb", 8.0))
    for material, thickness in slabs:
        X = {m: 0.0 for m in MATERIALS}
        X[material] = thickness * MATERIALS[material].rho
        radius = 1.2 * MATERIALS[material].A ** (1.0/3.0)
        for p in MOMENTA:
            theta_ff = 197.3 / (1000.0*p*radius)
            theta_nuc = 197.3 / (1000.0*p*0.84)
            edges = sorted(set(
                [0.0, 0.5*theta_ff, theta_ff, 2.0*theta_ff,
                 0.5*theta_nuc, theta_nuc, 2.0*theta_nuc, 0.300]
            ))
            edges = [x for x in edges if 0.0 <= x <= 0.300]
            if edges[-1] < 0.300:
                edges.append(0.300)
            pattern = f"{material}_t{thickness}_p{p}_*_s*.txt"
            for file in sorted(raw.glob(pattern)):
                name = file.stem
                transport = "ftfp_bert_wvi" if "ftfp_bert_wvi" in name else "ftfp_bert"
                angles = load_angles(file)
                g4 = sample_moments(angles, 0.200, n_generated=1_000_000)
                for model in ("gaussian", "uniform_sphere"):
                    q200 = constant_calibration(X, p, 0.200, form_factor=model)
                    summary_rows.append(dict(
                        material=material, thickness_cm=thickness, p_GeV=p,
                        transport=transport, form_factor=model,
                        theta_FF_mrad=1000.0*theta_ff,
                        theta_nuc_mrad=1000.0*theta_nuc,
                        n_exit=len(angles), Fc_g4=g4["Fc"], Fc_model=q200["Fc"],
                        theta_rms_g4=g4["theta_rms"], theta_rms_model=q200["theta_rms"],
                        rms_model_over_g4_minus1=q200["theta_rms"]/g4["theta_rms"]-1.0,
                        quadratic_weight_bias_if_g4_true=(g4["theta_rms"]/q200["theta_rms"])**2-1.0,
                    ))
                    cumulative = {}
                    for edge in edges:
                        if edge == 0.0:
                            cumulative[edge] = (0.0, 0.0)
                        else:
                            q = constant_calibration(X, p, edge, form_factor=model)
                            cumulative[edge] = (q["Fc"], q["Fc"]*q["M2"])
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        mask = (angles >= lo) & (angles < hi)
                        Fc_lo, n2_lo = cumulative[lo]
                        Fc_hi, n2_hi = cumulative[hi]
                        band_rows.append(dict(
                            material=material, thickness_cm=thickness, p_GeV=p,
                            transport=transport, form_factor=model,
                            theta_lo_mrad=1000.0*lo, theta_hi_mrad=1000.0*hi,
                            straddles_theta_FF=lo <= theta_ff <= hi,
                            straddles_theta_nuc=lo <= theta_nuc <= hi,
                            probability_g4=float(mask.sum()/1_000_000),
                            probability_model=Fc_hi-Fc_lo,
                            M2_numerator_g4=float(np.sum(angles[mask]**2)/1_000_000),
                            M2_numerator_model=n2_hi-n2_lo,
                        ))
    summary = pd.DataFrame(summary_rows)
    bands = pd.DataFrame(band_rows)
    summary.to_csv(out / "geant4_finite_size_benchmark.csv", index=False)
    bands.to_csv(out / "geant4_finite_size_bands.csv", index=False)
    return summary, bands


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


def cache_sensitivity(outdir, n_events=500_000, seed=0, form_factor="none"):
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
        cache = PofxCache(nmax=2, form_factor=form_factor, **kwargs)
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
                form_factor=form_factor,
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
    p = sub.add_parser("transform-g1")
    p.add_argument("--out", default="out/validation/transform_g1")
    p.add_argument("--threshold-pp", type=float, default=0.05)
    p = sub.add_parser("finite-size-transform")
    p.add_argument("--out", default="out/validation/finite_size_transform")
    p = sub.add_parser("decision-gates")
    p.add_argument("--out", default="out/validation/decision_gates")
    p = sub.add_parser("analytic-completion")
    p.add_argument("--out", default="out/validation/analytic_completion")
    p = sub.add_parser("finite-size-sampler")
    p.add_argument("--out", default="out/validation/finite_size_sampler")
    p.add_argument("--n-mc", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--ff-model", choices=("point", "gauss", "sphere"), required=True)
    p.add_argument("--floor", choices=("on", "off"), required=True)
    p = sub.add_parser("task1")
    p.add_argument("--out", default="out/validation/task1")
    p = sub.add_parser("task2")
    p.add_argument("--out", default="out/validation/task2")
    p = sub.add_parser("task3")
    p.add_argument("--out", default="out/validation/task3")
    p = sub.add_parser("task4")
    p.add_argument("--out", default="out/validation/task4")
    p = sub.add_parser("geant4-finite")
    p.add_argument("--out", default="out/validation/geant4_finite")
    p.add_argument("--rawdir", default="out/geant4/raw")
    p = sub.add_parser("raw-mc")
    p.add_argument("--out", default="out/validation/raw_mc")
    p.add_argument("--n-mc", type=int, default=2_000_000)
    p = sub.add_parser("cache")
    p.add_argument("--out", default="out/validation/cache")
    p.add_argument("--n-events", type=int, default=500_000)
    p.add_argument("--form-factor", choices=("none", "gaussian", "uniform_sphere"), default="none")
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
    p.add_argument("--form-factor", choices=("none", "gaussian", "uniform_sphere"), default="none")
    a = ap.parse_args()
    if a.cmd == "quadrature":
        print(quadrature_validation(a.out, n_mc=a.n_mc).to_string(index=False))
    elif a.cmd == "transform-g1":
        print(
            transform_g1_closure(a.out, threshold_pp=a.threshold_pp).to_string(
                index=False
            )
        )
    elif a.cmd == "finite-size-transform":
        result, systematic = finite_size_transform_regeneration(a.out)
        print(result.to_string(index=False))
        print("\nIncoherent-floor systematic:\n", systematic.to_string(index=False))
    elif a.cmd == "decision-gates":
        gate_a, _, optimum = finite_size_decision_gates(a.out)
        print("Gate A:\n", gate_a.to_string(index=False))
        print("\nGate B optima:\n", optimum.to_string(index=False))
    elif a.cmd == "analytic-completion":
        _, _, summary, _ = finite_size_analytic_completion(a.out)
        print(summary.to_string(index=False))
    elif a.cmd == "finite-size-sampler":
        print(
            finite_size_sampler_validation(
                a.out, a.n_mc, a.seed, a.ff_model, a.floor == "on"
            ).to_string(index=False)
        )
    elif a.cmd == "task1":
        print(task1_untruncated_rms(a.out).to_string(index=False))
    elif a.cmd == "task2":
        _, optimum = task2_efficiency(a.out)
        print(optimum.to_string(index=False))
    elif a.cmd == "task3":
        _, summary = task3_composition(a.out)
        print(summary.to_string(index=False))
    elif a.cmd == "task4":
        _, gate = task4_tail(a.out)
        print(gate.to_string(index=False))
    elif a.cmd == "geant4-finite":
        summary, _ = geant4_finite_size_benchmark(a.out, rawdir=a.rawdir)
        print(summary.to_string(index=False))
    elif a.cmd == "raw-mc":
        print(raw_sampler_overlay(a.out, n_mc=a.n_mc).to_string(index=False))
    elif a.cmd == "cache":
        _, c = cache_sensitivity(a.out, n_events=a.n_events, form_factor=a.form_factor)
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
