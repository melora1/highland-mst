"""Theory tables and detector-level analysis for the revised study."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AL_HALF,
    CU_HALF,
    CU_ROI_R,
    CU_ROI_ZHALF,
    MATERIALS,
    MIN_VOX_COUNT,
    MOMENTA,
    N_VOX,
    PB_CX,
    PB_CY,
    PB_ROI_R,
    PB_ROI_ZHALF,
    RADIAL_ETA_MAX,
    ROI_GUARD_GAPS_CM,
    SPLIT_SEED,
    THETA_CUT,
    VOX_HALF,
    VOX_SIZE,
)
from physics import (
    Layer,
    PofxCache,
    beta_of,
    calibrate_pofx,
    constant_calibration,
    epsilon_asymptotic,
    fit_eta1,
    fit_log_asymptote,
    mu2_eta,
    optimal_k_constant,
    radial_tail_ratio,
    reduced_parameters,
    theta0_highland,
    theta_space_highland,
)
from simulation import SEG_NAMES, segment_matrix

EDGES = np.linspace(-VOX_HALF, VOX_HALF, N_VOX + 1)
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])


def fiducial_voxel_mask():
    """Voxels fully contained in the 25 cm outer Al cube.

    Quantitative map statistics exclude boundary voxels that straddle the
    physical target edge.  This prevents grazing reconstructed reference paths
    from producing arbitrarily small denominators in partially outside voxels.
    """
    X, Y, Z = np.meshgrid(CENTERS, CENTERS, CENTERS, indexing="ij")
    half = 0.5 * VOX_SIZE
    tol = 1e-12
    return (
        (np.abs(X) + half <= AL_HALF + tol)
        & (np.abs(Y) + half <= AL_HALF + tol)
        & (np.abs(Z) + half <= AL_HALF + tol)
    )


def valid_voxel_mask(counts, *arrays, min_count=MIN_VOX_COUNT):
    """Common quantitative voxel mask: occupancy, fiducial containment, finite data."""
    valid = (np.asarray(counts) >= min_count) & fiducial_voxel_mask()
    for a in arrays:
        valid &= np.isfinite(np.asarray(a))
    return valid


def path_X(thicknesses):
    """thicknesses dict material->cm -> areal densities dict."""
    return {m: float(thicknesses.get(m, 0.0)) * MATERIALS[m].rho for m in MATERIALS}


PATHS = {
    "Al25": path_X({"Al": 25.0}),
    "Cu15": path_X({"Cu": 15.0}),
    "AlCu": path_X({"Al": 10.0, "Cu": 15.0}),
    "Pb15": path_X({"Pb": 15.0}),
}

AXIAL_ORDERED = (Layer("Al", 5.0), Layer("Cu", 15.0), Layer("Al", 5.0))
PBCROSSING_ORDERED = (Layer("Al", 5.0), Layer("Pb", 15.0), Layer("Al", 5.0))
OFFCU_ORDERED = (Layer("Al", 25.0),)


def run_theory(outdir):
    """Produce Steps 1,2,4,6.2,6.3 tables without detector simulation."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Constant-p fixed-path collapse and exact reduced identity.
    rows = []
    for p in MOMENTA:
        r = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=2)
        kopt, eff = optimal_k_constant(PATHS["AlCu"], p, nmax=2)
        rows.append(
            dict(
                p=p,
                R=r["R"],
                B=r["B"],
                k=r["k"],
                eta_cut=r["eta_cut"],
                epsilon=r["epsilon"],
                k_opt=kopt,
                eta_eff=eff,
                exact_error=(1 + r["epsilon"]) ** 2 - r["exact_ratio2"],
                clipped_fraction=r["clipped_fraction"],
            )
        )
    collapse = pd.DataFrame(rows)
    collapse.to_csv(out / "theory_collapse.csv", index=False)

    # Fixed-path momentum-invariance summary.  R and B vary at the 1e-3 level
    # individually but anticorrelate, leaving RB substantially more invariant.
    rb = collapse["R"] * collapse["B"]
    pd.DataFrame([dict(
        R_mean=float(collapse["R"].mean()),
        R_peak_to_peak_fraction=float((collapse["R"].max() - collapse["R"].min()) / collapse["R"].mean()),
        B_mean=float(collapse["B"].mean()),
        B_peak_to_peak_fraction=float((collapse["B"].max() - collapse["B"].min()) / collapse["B"].mean()),
        RB_mean=float(rb.mean()),
        RB_peak_to_peak_fraction=float((rb.max() - rb.min()) / rb.mean()),
    )]).to_csv(out / "rb_invariance_summary.csv", index=False)

    # Usable reduced-moment table for assessing the weak B dependence directly.
    mu_rows = []
    for B in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0):
        for eta in (2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0):
            mu_rows.append(dict(B=B, eta_cut=eta, mu2=mu2_eta(eta, B, nmax=2)))
    pd.DataFrame(mu_rows).to_csv(out / "mu2_grid.csv", index=False)

    # p(X) energy-loss table.  Two explicit screening-log continuations are
    # carried so their finite-loss spread is a visible model systematic rather
    # than an arbitrary hidden convention.
    rows = []
    for label, path in (
        ("AlCu", AXIAL_ORDERED),
        ("AlPb", PBCROSSING_ORDERED),
        ("Al25", OFFCU_ORDERED),
    ):
        for p in MOMENTA:
            rd = calibrate_pofx(path, p, THETA_CUT, nmax=2, screening_weight="dchi_c2")
            rs = calibrate_pofx(path, p, THETA_CUT, nmax=2, screening_weight="serial")
            rows.append(
                dict(
                    path=label,
                    p=p,
                    x_over_x0=rd["x_over_x0"],
                    dp_over_p=rd["dp_over_p"],
                    delta_E_MeV=1000 * rd["delta_E"],
                    n_slices=rd["n_slices"],
                    eta_cut=rd["eta_cut"],
                    Fc=rd["Fc"],
                    B_dchi=rd["B"],
                    B_serial=rs["B"],
                    epsilon_matched_dchi=rd["epsilon_matched"],
                    epsilon_matched_serial=rs["epsilon_matched"],
                    epsilon_matched_spread_pp=100
                    * abs(rd["epsilon_matched"] - rs["epsilon_matched"]),
                    epsilon_mixed_dchi=rd["epsilon_mixed"],
                    epsilon_mixed_serial=rs["epsilon_mixed"],
                    epsilon_mixed_spread_pp=100
                    * abs(rd["epsilon_mixed"] - rs["epsilon_mixed"]),
                    E_w_nominal_dchi=(1 + rd["epsilon_mixed"]) ** 2,
                    E_w_nominal_serial=(1 + rs["epsilon_mixed"]) ** 2,
                    clipped_fraction=max(
                        rd["clipped_fraction"], rs["clipped_fraction"]
                    ),
                )
            )
    pofx = pd.DataFrame(rows)
    pofx.to_csv(out / "energy_loss_calibration.csv", index=False)

    # Log-law protocol with the numerical-table and analytic-tail roles kept
    # separate.  Slope diagnostics are restricted to eta <= RADIAL_ETA_MAX so
    # the fitted coefficient is actually tested against Phi^(1)/Phi^(2), not
    # imposed by the analytic continuation.  Deep windows are retained only to
    # stabilize the asymptotic intercept eta1.
    slope_windows = ((8.0, 20.0, 41), (10.0, 30.0, 41), (15.0, 30.0, 41))
    intercept_windows = ((30.0, 100.0, 41), (50.0, 500.0, 61))
    eta_rows = []
    for name, X in PATHS.items():
        rp = reduced_parameters(X, 1.0)
        for nmax in (1, 2):
            for role, windows in (
                ("slope_diagnostic", slope_windows),
                ("eta1_asymptote", intercept_windows),
            ):
                for eta_lo, eta_hi, n_eta in windows:
                    eta_grid = np.geomspace(eta_lo, eta_hi, n_eta)
                    eps = []
                    for eta in eta_grid:
                        cut = eta * math.sqrt(rp["chi_c2"] * rp["B"])
                        eps.append(
                            constant_calibration(X, 1.0, cut, nmax=nmax)["epsilon"]
                        )
                    joint = fit_log_asymptote(eta_grid, eps, R=rp["R"])
                    eta1_fixed = fit_eta1(eta_grid, eps, rp["R"])
                    tail_mask = eta_grid > RADIAL_ETA_MAX
                    eta_rows.append(
                        dict(
                            path=name,
                            nmax=nmax,
                            window_role=role,
                            eta_min=eta_lo,
                            eta_max=eta_hi,
                            eta_table_max=RADIAL_ETA_MAX,
                            analytic_tail_point_fraction=float(np.mean(tail_mask)),
                            R=rp["R"],
                            B=rp["B"],
                            sqrt2RB=rp["sqrt2RB"],
                            slope_fit=joint["slope"],
                            slope_expected=joint["slope_expected"],
                            slope_ratio=joint["slope_ratio"],
                            slope_rel_error=joint["slope_rel_error"],
                            intercept_fit=joint["intercept"],
                            eta1_joint=joint["eta1"],
                            eta1_fixed_2R=eta1_fixed,
                            rms_residual=joint["rms_residual"],
                            max_abs_residual=joint["max_abs_residual"],
                        )
                    )
    eta1_df = pd.DataFrame(eta_rows)
    eta1_df.to_csv(out / "eta1_protocol.csv", index=False)

    # General dimensionless design table: epsilon_M(eta_cut; R, B).
    curve_rows = []
    for B in (14.0, 16.0, 18.0, 20.0):
        for R in (0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.20):
            for eta in np.geomspace(2.0, 50.0, 80):
                eps = math.sqrt(R * B * mu2_eta(float(eta), B, nmax=2)) - 1.0
                curve_rows.append(dict(B=B, R=R, eta_cut=eta, epsilon=eps))
    pd.DataFrame(curve_rows).to_csv(out / "dimensionless_curves.csv", index=False)

    # Matched k and matched eta composition comparisons.
    comp = []
    for name, X in PATHS.items():
        rp = reduced_parameters(X, 1.0)
        for k in (4.0, 8.0, 16.0, 32.0):
            cut = k * rp["theta_space"] / math.sqrt(2.0)
            r = constant_calibration(X, 1.0, cut, nmax=2)
            comp.append(
                dict(
                    match="k",
                    value=k,
                    path=name,
                    R=r["R"],
                    B=r["B"],
                    eta_cut=r["eta_cut"],
                    epsilon=r["epsilon"],
                )
            )
        for eta in (3.0, 5.5, 10.0, 20.0):
            cut = eta * math.sqrt(rp["chi_c2"] * rp["B"])
            r = constant_calibration(X, 1.0, cut, nmax=2)
            comp.append(
                dict(
                    match="eta",
                    value=eta,
                    path=name,
                    R=r["R"],
                    B=r["B"],
                    eta_cut=r["eta_cut"],
                    epsilon=r["epsilon"],
                )
            )
    pd.DataFrame(comp).to_csv(out / "composition_matched.csv", index=False)

    # n<=1 vs n<=2 convergence and tail check.
    conv = []
    for p in MOMENTA:
        r1 = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=1)
        r2 = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=2)
        conv.append(
            dict(
                p=p,
                eps_n1=r1["epsilon"],
                eps_n2=r2["epsilon"],
                shift_abs=r2["epsilon"] - r1["epsilon"],
                shift_rel=(r2["epsilon"] / r1["epsilon"] - 1.0)
                if r1["epsilon"]
                else np.nan,
                clip_n1=r1["clipped_fraction"],
                clip_n2=r2["clipped_fraction"],
            )
        )
    pd.DataFrame(conv).to_csv(out / "truncation_convergence.csv", index=False)

    # Tail check at 6 GeV/c so 50--200 mrad is actually in the Rutherford
    # approach region; at 1 GeV/c these angles still overlap the core.
    r = constant_calibration(PATHS["AlCu"], 6.0, THETA_CUT, nmax=2)
    s_mol = math.sqrt(r["chi_c2"] * r["B"])
    tail = pd.DataFrame(
        [
            dict(
                p_GeV=6.0,
                theta_mrad=th * 1000,
                eta=th / s_mol,
                ratio=radial_tail_ratio(th, r["chi_c2"], r["B"], nmax=2),
            )
            for th in (0.050, 0.100, 0.200)
        ]
    )
    tail.to_csv(out / "tail_check.csv", index=False)
    return dict(collapse=collapse, pofx=pofx, eta1=eta1_df)


def _hist3(values, weights=None):
    H, _ = np.histogramdd(values, bins=(EDGES, EDGES, EDGES), weights=weights)
    return H


def image_from_events(df, weights):
    xyz = df[["poca_x", "poca_y", "poca_z"]].to_numpy(float)
    count = _hist3(xyz)
    sums = _hist3(xyz, np.asarray(weights, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        image = np.where(count > 0, sums / count, np.nan)
    return image, count


def _central_p_only_eps(p, cache, theta_cut=THETA_CUT):
    p = np.asarray(p, float)
    seg = np.tile(np.array([5.0, 15.0, 0.0, 0.0, 5.0]), (p.size, 1))
    return cache.arrays(p, seg, theta_cut)["epsilon_mixed"]


def build_weights(df, cache=None, theta_cut=THETA_CUT):
    """Fixed-cut weights with all energy-loss quantities kept distinct."""
    cache = cache or PofxCache(nmax=2)
    p = df.p_meas.to_numpy(float)
    dth = df.dth_reco.to_numpy(float)
    xx0 = df.xx0_ref_reco.to_numpy(float)
    tspace_incident = theta_space_highland(p, xx0)
    q_reco = cache.arrays(p, segment_matrix(df, "ref_reco"), theta_cut)
    q_true = cache.arrays(
        df.p_true.to_numpy(float), segment_matrix(df, "ref_true"), theta_cut
    )

    w_nom = (
        np.divide(
            dth, tspace_incident, out=np.zeros_like(dth), where=tspace_incident > 0
        )
        ** 2
    )
    eps_p = _central_p_only_eps(p, cache, theta_cut=theta_cut)
    den_p = (1.0 + eps_p) * tspace_incident
    w_p = np.divide(dth, den_p, out=np.zeros_like(dth), where=den_p > 0) ** 2
    w_Q = (
        np.divide(
            dth,
            q_reco["theta_rms"],
            out=np.zeros_like(dth),
            where=q_reco["theta_rms"] > 0,
        )
        ** 2
    )
    w_ideal = (
        np.divide(
            df.dth_true.to_numpy(float),
            q_true["theta_rms"],
            out=np.zeros(len(df)),
            where=q_true["theta_rms"] > 0,
        )
        ** 2
    )

    # Two means are reported because the published 12.18% used a weighted
    # average, whereas an unweighted event-count mean is the cleaner scale-null
    # control.  I_const deliberately uses the event-count mean.
    # A mismatch is defined only when the reconstructed reference path has a
    # nonzero Highland and accepted-RMS denominator.  Do not let NumPy's divide
    # fallback turn an empty reference path into epsilon=-1 and contaminate the
    # event-count scalar control.
    valid_reference = (
        np.isfinite(tspace_incident)
        & np.isfinite(q_reco["theta_rms"])
        & (tspace_incident > 0)
        & (q_reco["theta_rms"] > 0)
    )
    ratio_event = np.full_like(dth, np.nan)
    np.divide(
        q_reco["theta_rms"],
        tspace_incident,
        out=ratio_event,
        where=valid_reference,
    )
    eps_event = ratio_event - 1.0
    finite = valid_reference & np.isfinite(eps_event)
    eps_bar_event = float(np.mean(eps_event[finite]))
    pos = finite & np.isfinite(w_nom) & (w_nom > 0)
    eps_bar_weighted = (
        float(np.average(eps_event[pos], weights=w_nom[pos])) if np.any(pos) else np.nan
    )
    den_const = (1.0 + eps_bar_event) * tspace_incident
    w_const = (
        np.divide(dth, den_const, out=np.zeros_like(dth), where=den_const > 0) ** 2
    )

    return dict(
        I_nom=w_nom,
        I_p=w_p,
        I_Q=w_Q,
        I_ideal=w_ideal,
        I_const=w_const,
        eps_event=eps_event,
        eps_matched_event=q_reco["epsilon_matched"],
        valid_reference=valid_reference,
        eps_p=eps_p,
        eps_bar_event_mean=eps_bar_event,
        eps_bar_nominal_weighted=eps_bar_weighted,
        theta_rms_reco=q_reco["theta_rms"],
        p_out_reco=q_reco["p_out"],
    ), cache


def roi_masks(guard_gap_cm=0.0):
    """Return Pb and Cu comparison masks with an optional guard gap around Pb.

    The Pb ROI is unchanged.  The Cu ROI excludes all voxels whose centers lie
    within ``PB_ROI_R + guard_gap_cm`` of the Pb axis.  This permits a direct
    sensitivity check for PoCA spill between the contiguous nominal ROIs.
    """
    guard_gap_cm = float(guard_gap_cm)
    if guard_gap_cm < 0:
        raise ValueError("guard_gap_cm must be non-negative")
    X, Y, Z = np.meshgrid(CENTERS, CENTERS, CENTERS, indexing="ij")
    pb_r2 = (X - PB_CX) ** 2 + (Y - PB_CY) ** 2
    pb = (pb_r2 <= PB_ROI_R**2) & (np.abs(Z) <= PB_ROI_ZHALF)
    cu = (X**2 + Y**2 <= CU_ROI_R**2) & (np.abs(Z) <= CU_ROI_ZHALF)
    cu &= pb_r2 > (PB_ROI_R + guard_gap_cm) ** 2
    return pb, cu


def image_metrics(img, counts, guard_gap_cm=0.0, min_count=MIN_VOX_COUNT):
    pb, cu = roi_masks(guard_gap_cm=guard_gap_cm)
    valid = valid_voxel_mask(counts, img, min_count=min_count)
    pb &= valid
    cu &= valid
    a, b = img[pb], img[cu]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return dict(
            SNR_Pb=np.nan,
            CNR=np.nan,
            mean_Pb=np.nan,
            mean_Cu=np.nan,
            sd_Pb=np.nan,
            sd_Cu=np.nan,
            n_pb=len(a),
            n_cu=len(b),
            guard_gap_cm=float(guard_gap_cm),
        )
    return dict(
        SNR_Pb=float(a.mean() / a.std(ddof=1)),
        CNR=float((a.mean() - b.mean()) / b.std(ddof=1)),
        mean_Pb=float(a.mean()),
        mean_Cu=float(b.mean()),
        sd_Pb=float(a.std(ddof=1)),
        sd_Cu=float(b.std(ddof=1)),
        n_pb=len(a),
        n_cu=len(b),
        guard_gap_cm=float(guard_gap_cm),
    )


def guard_gap_sensitivity(
    images, counts, gaps=ROI_GUARD_GAPS_CM, min_count=MIN_VOX_COUNT
):
    """ROI-metric sensitivity to a finite gap between Pb and Cu comparison ROIs."""
    rows = []
    for gap in gaps:
        for name, img in images.items():
            if name not in ("I_nom", "I_p", "I_Q", "I_ideal", "I_const", "I_scale_opt"):
                continue
            rows.append(
                dict(
                    image=name,
                    **image_metrics(
                        img, counts, guard_gap_cm=gap, min_count=min_count
                    ),
                )
            )
    return pd.DataFrame(rows)


def _event_roi_masks(df, guard_gap_cm=0.0):
    """Event-level reconstructed Pb/Cu ROI masks for PoCA spill diagnostics."""
    x = df.poca_x.to_numpy(float)
    y = df.poca_y.to_numpy(float)
    z = df.poca_z.to_numpy(float)
    pb_r2 = (x - PB_CX) ** 2 + (y - PB_CY) ** 2
    pb = (pb_r2 <= PB_ROI_R**2) & (np.abs(z) <= PB_ROI_ZHALF)
    cu = (x * x + y * y <= CU_ROI_R**2) & (np.abs(z) <= CU_ROI_ZHALF)
    cu &= pb_r2 > (PB_ROI_R + float(guard_gap_cm)) ** 2
    return pb, cu


def roi_spill_diagnostics(df, gaps=ROI_GUARD_GAPS_CM):
    """Truth-class migration into the reconstructed Pb/Cu ROIs."""
    rows = []
    truth_groups = {
        "Pb-crossing": df.true_pb.to_numpy(bool),
        "Cu-only": df.true_cu_only.to_numpy(bool),
        "Al-only": df.true_al_only.to_numpy(bool),
    }
    for gap in gaps:
        pb, cu = _event_roi_masks(df, guard_gap_cm=gap)
        for truth_name, mask in truth_groups.items():
            n = int(mask.sum())
            rows.append(
                dict(
                    guard_gap_cm=float(gap),
                    truth_group=truth_name,
                    n_events=n,
                    frac_in_pb_roi=float(np.mean(pb[mask])) if n else np.nan,
                    frac_in_cu_roi=float(np.mean(cu[mask])) if n else np.nan,
                    frac_in_neither=float(np.mean((~pb & ~cu)[mask])) if n else np.nan,
                )
            )
    return pd.DataFrame(rows)


def response_closure_by_momentum(
    df, weights, cache, generated=None, theta_cut=THETA_CUT
):
    """Per-setting closure and detector-response decomposition on accepted events.

    All rows use the same reconstructed-cut-selected event sample so differences
    isolate numerator/path/momentum substitutions rather than acceptance changes.
    """
    p_true = df.p_true.to_numpy(float)
    p_meas = df.p_meas.to_numpy(float)
    d_true = df.dth_true.to_numpy(float)
    d_reco = df.dth_reco.to_numpy(float)
    seg_true = segment_matrix(df, "ref_true")
    seg_reco = segment_matrix(df, "ref_reco")

    q_true = cache.arrays(p_true, seg_true, theta_cut)["theta_rms"]
    q_reco_truep = cache.arrays(p_true, seg_reco, theta_cut)["theta_rms"]
    q_true_measp = cache.arrays(p_meas, seg_true, theta_cut)["theta_rms"]

    def sq(num, den):
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0) ** 2

    variants = {
        "w_ideal_trueangle_truep_truepath": sq(d_true, q_true),
        "w_angle_only_recoangle_truep_truepath": sq(d_reco, q_true),
        "w_path_only_trueangle_truep_recopath": sq(d_true, q_reco_truep),
        "w_momentum_only_trueangle_measp_truepath": sq(d_true, q_true_measp),
        "w_angle_path_recoangle_truep_recopath": sq(d_reco, q_reco_truep),
        "w_angle_momentum_recoangle_measp_truepath": sq(d_reco, q_true_measp),
        "w_Q_full": np.asarray(weights["I_Q"], float),
        "w_nom": np.asarray(weights["I_nom"], float),
        "w_p": np.asarray(weights["I_p"], float),
    }

    rows = []
    pset = df.p_set.to_numpy(float)
    for p0 in sorted(np.unique(pset)):
        m = np.isclose(pset, p0)
        n_generated = int(np.sum(np.isclose(generated.p_set.to_numpy(float), p0))) if generated is not None else int(m.sum())
        row = dict(
            p_set=float(p0),
            n_generated=n_generated,
            n_accepted=int(m.sum()),
            acceptance_fraction=float(m.sum() / n_generated) if n_generated else np.nan,
            mean_p_true=float(np.mean(p_true[m])),
            mean_p_meas=float(np.mean(p_meas[m])),
            mean_p_ratio=float(np.mean(p_meas[m] / p_true[m])),
            mean_p2_ratio=float(np.mean((p_meas[m] / p_true[m]) ** 2)),
            sd_fractional_p_error=float(np.std(p_meas[m] / p_true[m] - 1.0, ddof=1)),
            mean_true_angle_mrad=1000.0 * float(np.mean(d_true[m])),
            mean_reco_angle_mrad=1000.0 * float(np.mean(d_reco[m])),
        )
        for key, arr in variants.items():
            row[f"mean_{key}"] = float(np.mean(arr[m]))
        ideal = row["mean_w_ideal_trueangle_truep_truepath"]
        for key in variants:
            row[f"delta_{key}_minus_ideal"] = row[f"mean_{key}"] - ideal
        rows.append(row)
    return pd.DataFrame(rows)


def roi_split_half_diagnostics(df, weights, seed=SPLIT_SEED):
    """Split-half stability of ROI means and descriptive SNR/CNR metrics.

    The single-split quantity |m0-m1|/2 is an empirical full-sample noise-scale
    estimate for a scalar metric under equal independent halves; it is a
    diagnostic, not a formal confidence interval.  Paired estimator differences
    are evaluated on the same half assignment.
    """
    rng = np.random.default_rng(int(seed))
    half = rng.integers(0, 2, size=len(df), dtype=np.int8)
    image_names = ("I_nom", "I_p", "I_Q", "I_ideal", "I_const")
    rows = []
    full_metrics = {}
    half_metrics = {0: {}, 1: {}}

    for name in image_names:
        full_img, full_count = image_from_events(df, weights[name])
        fm = image_metrics(full_img, full_count)
        full_metrics[name] = fm
        for h in (0, 1):
            m = half == h
            img, count = image_from_events(df.loc[m], np.asarray(weights[name])[m])
            half_metrics[h][name] = image_metrics(img, count)
        h0, h1 = half_metrics[0][name], half_metrics[1][name]
        rows.append(
            dict(
                kind="image",
                comparison=name,
                SNR_full=fm["SNR_Pb"],
                SNR_half0=h0["SNR_Pb"],
                SNR_half1=h1["SNR_Pb"],
                SNR_noise_full_est=0.5 * abs(h0["SNR_Pb"] - h1["SNR_Pb"]),
                CNR_full=fm["CNR"],
                CNR_half0=h0["CNR"],
                CNR_half1=h1["CNR"],
                CNR_noise_full_est=0.5 * abs(h0["CNR"] - h1["CNR"]),
                split_seed=int(seed),
            )
        )

    for a, b in (("I_nom", "I_Q"), ("I_p", "I_Q")):
        full_dsnr = full_metrics[a]["SNR_Pb"] - full_metrics[b]["SNR_Pb"]
        full_dcnr = full_metrics[a]["CNR"] - full_metrics[b]["CNR"]
        dsnr0 = half_metrics[0][a]["SNR_Pb"] - half_metrics[0][b]["SNR_Pb"]
        dsnr1 = half_metrics[1][a]["SNR_Pb"] - half_metrics[1][b]["SNR_Pb"]
        dcnr0 = half_metrics[0][a]["CNR"] - half_metrics[0][b]["CNR"]
        dcnr1 = half_metrics[1][a]["CNR"] - half_metrics[1][b]["CNR"]
        rows.append(
            dict(
                kind="paired_difference",
                comparison=f"{a}-{b}",
                SNR_full=full_dsnr,
                SNR_half0=dsnr0,
                SNR_half1=dsnr1,
                SNR_noise_full_est=0.5 * abs(dsnr0 - dsnr1),
                CNR_full=full_dcnr,
                CNR_half0=dcnr0,
                CNR_half1=dcnr1,
                CNR_noise_full_est=0.5 * abs(dcnr0 - dcnr1),
                split_seed=int(seed),
            )
        )
    return pd.DataFrame(rows)

def _reference_class_masks(df, prefix):
    """Return Al-only and Cu-bearing masks for a stored reference-path trace."""
    seg = segment_matrix(df, prefix)
    al_only = (seg[:, 1] + seg[:, 2] + seg[:, 3]) <= 1e-8
    cu_bearing = (seg[:, 1] + seg[:, 3]) > 1e-8
    return {"Al-only": al_only, "Cu-bearing": cu_bearing}


def path_residual_diagnostics(df, weights):
    """Compare I_p-I_Q for both truth and reconstructed reference-path classes."""
    out = []
    for classification, prefix in (("truth", "ref_true"), ("reconstructed", "ref_reco")):
        for name, mask in _reference_class_masks(df, prefix).items():
            d = df.loc[mask]
            if d.empty:
                continue
            residual_event = weights["I_p"][mask] - weights["I_Q"][mask]
            residual_img, c = image_from_events(d, residual_event)
            valid = valid_voxel_mask(c, residual_img)
            out.append(
                dict(
                    classification=classification,
                    region=name,
                    n_events=int(mask.sum()),
                    event_rms=float(np.sqrt(np.mean(residual_event**2))),
                    image_rms=float(np.sqrt(np.mean(residual_img[valid] ** 2)))
                    if np.any(valid)
                    else np.nan,
                    n_voxels=int(np.sum(valid)),
                )
            )
    return pd.DataFrame(out)


def split_half_noise_diagnostics(df, weights, seed=SPLIT_SEED, min_half_count=None):
    """Estimate the map-RMS noise floor for I_p-I_Q with an independent split.

    Events are assigned randomly to two halves with a fixed seed.  For a voxel
    with half counts n0,n1, the half-difference of the mean event residuals is
    scaled by sqrt(n0*n1)/(n0+n1), which converts its variance to the variance
    of the full-sample mean under the usual independent-identically-distributed
    approximation.  The reported ``physical_rms_quadrature`` is therefore a
    diagnostic, not a deconvolution theorem.
    """
    if min_half_count is None:
        min_half_count = max(5, int(math.ceil(MIN_VOX_COUNT / 2)))
    rng = np.random.default_rng(int(seed))
    half = rng.integers(0, 2, size=len(df), dtype=np.int8)
    residual = np.asarray(weights["I_p"] - weights["I_Q"], float)

    groups = [("all", "all", np.ones(len(df), dtype=bool))]
    for classification, prefix in (("truth", "ref_true"), ("reconstructed", "ref_reco")):
        for region, mask in _reference_class_masks(df, prefix).items():
            groups.append((classification, region, mask))

    rows = []
    for classification, region, class_mask in groups:
        d_full = df.loc[class_mask]
        if d_full.empty:
            continue
        full_img, full_count = image_from_events(d_full, residual[class_mask])

        half_imgs = []
        half_counts = []
        for h in (0, 1):
            m = class_mask & (half == h)
            d = df.loc[m]
            if d.empty:
                half_imgs.append(np.full_like(full_img, np.nan))
                half_counts.append(np.zeros_like(full_count))
                continue
            img, count = image_from_events(d, residual[m])
            half_imgs.append(img)
            half_counts.append(count)

        n0, n1 = half_counts
        valid = (
            fiducial_voxel_mask()
            & (full_count >= MIN_VOX_COUNT)
            & (n0 >= min_half_count)
            & (n1 >= min_half_count)
            & np.isfinite(full_img)
            & np.isfinite(half_imgs[0])
            & np.isfinite(half_imgs[1])
        )
        if not np.any(valid):
            rows.append(
                dict(
                    classification=classification,
                    region=region,
                    n_events=int(class_mask.sum()),
                    n_voxels=0,
                    observed_rms=np.nan,
                    split_difference_rms=np.nan,
                    noise_rms_full_est=np.nan,
                    physical_rms_quadrature=np.nan,
                    min_half_count=min_half_count,
                    split_seed=int(seed),
                )
            )
            continue

        observed = full_img[valid]
        split_diff = half_imgs[0][valid] - half_imgs[1][valid]
        scale = np.sqrt(n0[valid] * n1[valid]) / (n0[valid] + n1[valid])
        noise_realization = split_diff * scale
        observed_rms = float(np.sqrt(np.mean(observed**2)))
        split_diff_rms = float(np.sqrt(np.mean(split_diff**2)))
        noise_rms = float(np.sqrt(np.mean(noise_realization**2)))
        physical = math.sqrt(max(observed_rms**2 - noise_rms**2, 0.0))
        rows.append(
            dict(
                classification=classification,
                region=region,
                n_events=int(class_mask.sum()),
                n_voxels=int(np.sum(valid)),
                observed_rms=observed_rms,
                split_difference_rms=split_diff_rms,
                noise_rms_full_est=noise_rms,
                physical_rms_quadrature=physical,
                noise_fraction=noise_rms / observed_rms if observed_rms else np.nan,
                min_half_count=min_half_count,
                split_seed=int(seed),
            )
        )
    return pd.DataFrame(rows)


def path_class_migration(df):
    """Truth-vs-reconstructed Al-only classification for the reference geometry."""
    t = _reference_class_masks(df, "ref_true")["Al-only"]
    r = _reference_class_masks(df, "ref_reco")["Al-only"]
    return pd.DataFrame(
        [
            dict(
                truth_al_only=bool(tv),
                reco_al_only=bool(rv),
                n=int(np.sum((t == tv) & (r == rv))),
            )
            for tv in (False, True)
            for rv in (False, True)
        ]
    )


def adaptive_retention(df, k_opt=1.800):
    """Adaptive-selection diagnostic with explicit denominators.

    Primary retention is defined relative to *all generated events* in each
    truth/reconstructed class.  A second conditional row reports retention
    among events that also satisfy the fixed 200 mrad cut, solely to make the
    denominator dependence transparent.
    """
    cut = k_opt * theta0_highland(
        df.p_meas.to_numpy(float), df.xx0_ref_reco.to_numpy(float)
    )
    keep = df.dth_reco.to_numpy(float) < cut
    fixed = df.pass_reco.to_numpy(bool)
    pb_truth = df.true_pb.to_numpy(bool)
    cu_truth = df.true_cu_only.to_numpy(bool)
    rpb = (df.poca_x.to_numpy() - PB_CX) ** 2 + (
        df.poca_y.to_numpy() - PB_CY
    ) ** 2 <= PB_ROI_R**2
    rpb &= np.abs(df.poca_z.to_numpy()) <= PB_ROI_ZHALF
    rcu = df.poca_x.to_numpy() ** 2 + df.poca_y.to_numpy() ** 2 <= CU_ROI_R**2
    rcu &= np.abs(df.poca_z.to_numpy()) <= CU_ROI_ZHALF
    rcu &= ~rpb

    rows = []
    for classification, groups in (
        ("truth", (("Pb-crossing", pb_truth), ("Cu-only", cu_truth))),
        ("reconstructed", (("Pb ROI", rpb), ("Cu ROI", rcu))),
    ):
        for group, mask in groups:
            n_all = int(mask.sum())
            rows.append(
                dict(
                    classification=classification,
                    group=group,
                    denominator="all generated",
                    n=n_all,
                    retention=float(np.mean(keep[mask])) if n_all else np.nan,
                )
            )
            cond = mask & fixed
            n_cond = int(cond.sum())
            rows.append(
                dict(
                    classification=classification,
                    group=group,
                    denominator="fixed-cut accepted",
                    n=n_cond,
                    retention=float(np.mean(keep[cond])) if n_cond else np.nan,
                )
            )
    return pd.DataFrame(rows)

def _rms(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else np.nan


def optimal_global_scale(I_nom, I_Q, valid):
    """Return c minimizing ||I_nom/c - I_Q||_2 over ``valid`` voxels."""
    x = np.asarray(I_nom, float)[valid]
    y = np.asarray(I_Q, float)[valid]
    den = float(x @ y)
    num = float(x @ x)
    if x.size == 0 or not np.isfinite(den) or den <= 0.0 or num <= 0.0:
        return np.nan
    return num / den


def highland_path_derivative(p_gev, x_over_x0):
    """Analytic d(theta0)/d(x/X0) for manuscript Eq. (1)."""
    p = np.asarray(p_gev, float)
    x = np.asarray(x_over_x0, float)
    beta = beta_of(p)
    theta0 = theta0_highland(p, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_factor = 1.0 + 0.038 * np.log(x / (beta * beta))
        relative = (0.5 * log_factor + 0.038) / (x * log_factor)
        derivative = theta0 * relative
    return np.where(x > 0.0, derivative, np.nan)


def path_length_error_diagnostics(df, n_bins=10):
    """Bin true/reconstructed path error and propagate it through Eq. (1).

    The primary ``reference_geometry`` rows compare exact and reconstructed
    rays under the same Al+Cu material hypothesis, isolating path geometry.
    ``physical_vs_reference`` is retained as a visibly composition-confounded
    diagnostic for Pb-crossing events.
    """
    p = df.p_true.to_numpy(float)
    reco = df.xx0_ref_reco.to_numpy(float)
    rows = []
    comparisons = (
        ("reference_geometry", df.xx0_ref_true.to_numpy(float)),
        ("physical_vs_reference", df.xx0_true.to_numpy(float)),
    )
    for label, true in comparisons:
        finite = np.isfinite(true) & np.isfinite(reco) & (true > 0.0)
        if not np.any(finite):
            continue
        # Quantile bins keep the requested roughly ten-bin diagnostic populated
        # despite the strongly bimodal Al-only/Al+Cu path distribution.
        edges = np.unique(np.quantile(true[finite], np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            continue
        which = np.clip(np.digitize(true, edges[1:-1], right=False), 0, edges.size - 2)
        delta = reco - true
        deriv = highland_path_derivative(p, true)
        theta0 = theta0_highland(p, true)
        for j in range(edges.size - 1):
            m = finite & (which == j)
            if not np.any(m):
                continue
            propagated = deriv[m] * delta[m]
            relative = np.divide(
                propagated,
                theta0[m],
                out=np.full(np.sum(m), np.nan),
                where=theta0[m] > 0,
            )
            x_mean = float(np.mean(true[m]))
            p_mean = float(np.mean(p[m]))
            rms_x = float(np.sqrt(np.mean(delta[m] ** 2)))
            derivative_at_mean = float(highland_path_derivative(p_mean, x_mean))
            theta0_at_mean = float(theta0_highland(p_mean, x_mean))
            propagated_bin = abs(derivative_at_mean) * rms_x
            rows.append(
                dict(
                    comparison=label,
                    bin=j,
                    x_true_min=float(edges[j]),
                    x_true_max=float(edges[j + 1]),
                    x_true_mean=x_mean,
                    p_true_mean=p_mean,
                    n_events=int(np.sum(m)),
                    rms_x_error=rms_x,
                    bias_x_error=float(np.mean(delta[m])),
                    dtheta0_dx_at_bin_mean=derivative_at_mean,
                    rms_theta0_error_rad=propagated_bin,
                    epsilon_path_rms=propagated_bin / theta0_at_mean,
                    eventwise_epsilon_path_rms=float(
                        np.sqrt(np.nanmean(relative * relative))
                    ),
                )
            )
    return pd.DataFrame(rows)


def analyze_events(
    df,
    outdir,
    cache=None,
    theta_cut=THETA_CUT,
    min_count=MIN_VOX_COUNT,
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    use = df[df.dth_reco.to_numpy(float) <= float(theta_cut)].reset_index(drop=True)
    weights, cache = build_weights(use, cache=cache, theta_cut=theta_cut)
    images = {}
    counts = None

    # First construct all event-defined image estimators.
    for name in ("I_nom", "I_p", "I_Q", "I_ideal", "I_const"):
        img, c = image_from_events(use, weights[name])
        images[name] = img
        counts = c

    # Exact one-parameter scale-null control requested in the manuscript:
    # minimize RMS(I_nom/c - I_Q) over the same valid voxel population.
    valid_scale = valid_voxel_mask(
        counts, images["I_nom"], images["I_Q"], min_count=min_count
    )
    c_opt = optimal_global_scale(images["I_nom"], images["I_Q"], valid_scale)
    images["I_scale_opt"] = (
        images["I_nom"] / c_opt if np.isfinite(c_opt) and c_opt > 0 else np.full_like(images["I_nom"], np.nan)
    )

    metric_rows = []
    for name in ("I_nom", "I_const", "I_scale_opt", "I_p", "I_Q", "I_ideal"):
        metric_rows.append(
            dict(
                image=name,
                **image_metrics(images[name], counts, min_count=min_count),
            )
        )
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)

    path_residual_diagnostics(use, weights).to_csv(
        out / "path_residuals.csv", index=False
    )
    split_half_noise_diagnostics(use, weights).to_csv(
        out / "split_half_noise.csv", index=False
    )
    path_class_migration(use).to_csv(out / "path_class_migration.csv", index=False)
    path_length_error_diagnostics(df).to_csv(
        out / "path_length_error.csv", index=False
    )
    adaptive_retention(df).to_csv(out / "adaptive_retention.csv", index=False)
    guard_gap_sensitivity(images, counts, min_count=min_count).to_csv(
        out / "roi_guard_gap_sensitivity.csv", index=False
    )
    roi_spill_diagnostics(use).to_csv(out / "roi_spill.csv", index=False)
    response_closure_by_momentum(
        use, weights, cache, generated=df, theta_cut=theta_cut
    ).to_csv(
        out / "weight_closure_by_momentum.csv", index=False
    )
    roi_split_half_diagnostics(use, weights).to_csv(
        out / "roi_split_half_metrics.csv", index=False
    )

    valid = valid_scale
    A = images["I_nom"] - images["I_Q"]
    C = images["I_const"] - images["I_Q"]
    S = images["I_scale_opt"] - images["I_Q"]
    R = images["I_p"] - images["I_Q"]
    artifact_rms = _rms(A[valid])
    const_rms = _rms(C[valid])
    scale_rms = _rms(S[valid])
    p_rms = _rms(R[valid])
    artifact = dict(
        artifact_rms=artifact_rms,
        const_residual_rms=const_rms,
        scale_opt_residual_rms=scale_rms,
        p_residual_rms=p_rms,
        const_reduction=1.0 - const_rms / artifact_rms if artifact_rms else np.nan,
        scale_opt_reduction=1.0 - scale_rms / artifact_rms if artifact_rms else np.nan,
        p_reduction=1.0 - p_rms / artifact_rms if artifact_rms else np.nan,
        c_opt=c_opt,
        denominator_scale_opt=math.sqrt(c_opt) if np.isfinite(c_opt) and c_opt > 0 else np.nan,
        n_valid_voxels=int(np.sum(valid)),
        n_fiducial_voxels=int(np.sum(fiducial_voxel_mask())),
        post_scalar_p_reduction=(1.0 - p_rms / scale_rms) if scale_rms else np.nan,
    )
    pd.DataFrame([artifact]).to_csv(out / "artifact_summary.csv", index=False)

    np.savez_compressed(out / "images.npz", centers=CENTERS, counts=counts, **images)
    pd.DataFrame(
        [
            dict(
                eps_bar_event_mean=weights["eps_bar_event_mean"],
                eps_bar_nominal_weighted=weights["eps_bar_nominal_weighted"],
                I_const_uses="event-count mean epsilon",
                I_scale_opt_uses="voxel RMS optimum of I_nom/c against I_Q",
                c_opt=c_opt,
                mean_dp_over_p=float(
                    np.mean(weights["p_out_reco"] / use.p_meas.to_numpy() - 1.0)
                ),
                screening_weight=cache.screening_weight,
                max_clipped=cache.max_clipped,
                local_kink_fallbacks=getattr(cache, "local_kink_fallbacks", 0),
                n_kinks=int(use.n_kinks.iloc[0]) if "n_kinks" in use else 1,
                theta_cut=float(theta_cut),
                min_vox_count=int(min_count),
                fiducial_rule=f"voxel fully inside outer Al cube: |coord|+{0.5 * VOX_SIZE:.3f} <= {AL_HALF:.3f} cm",
            )
        ]
    ).to_csv(out / "calibration_summary.csv", index=False)
    return images, counts, weights, cache

def analyze_gradient(df, outdir, cache=None, theta_cut=THETA_CUT):
    """Spatial-gradient test with self-consistent and closure predictors."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    use = df[df.dth_reco.to_numpy(float) <= float(theta_cut)].reset_index(drop=True)
    weights, cache = build_weights(use, cache=cache, theta_cut=theta_cut)
    I_nom, counts = image_from_events(use, weights["I_nom"])
    I_Q, _ = image_from_events(use, weights["I_Q"])
    observed = I_nom - I_Q

    # Redesigned mechanism predictor: use the self-consistent p(X) mismatch,
    # whose four-setting span is materially larger than the upstream-tagged
    # mismatch used in the original gradient diagnostic.
    valid_reference = np.asarray(weights["valid_reference"], bool)
    excess_matched = np.zeros(len(use), float)
    excess_matched[valid_reference] = (
        (1.0 + weights["eps_matched_event"][valid_reference]) ** 2 - 1.0
    )
    predicted_unweighted, _ = image_from_events(
        use.loc[valid_reference], excess_matched[valid_reference]
    )

    # Algebraic closure predictor.  Since w_nom-w_Q = w_Q*((1+eps)^2-1) for
    # the same event, this should reproduce the observed map up to roundoff and
    # serves as a code/weighting closure check, not as independent evidence.
    excess_mixed = np.zeros(len(use), float)
    excess_mixed[valid_reference] = (
        (1.0 + weights["eps_event"][valid_reference]) ** 2 - 1.0
    )
    predicted_weighted, _ = image_from_events(use, weights["I_Q"] * excess_mixed)

    summaries = []
    residuals = {}
    for label, predicted in (
        ("self_consistent_normalization_field", predicted_unweighted),
        ("wQ_weighted_closure", predicted_weighted),
    ):
        valid = valid_voxel_mask(counts, observed, predicted)
        x = predicted[valid]
        y = observed[valid]
        amp = float((x @ y) / (x @ x)) if x.size and (x @ x) > 0 else np.nan
        residual = observed - amp * predicted
        residuals[label] = residual
        yrms = float(np.sqrt(np.mean(y * y))) if y.size else np.nan
        rrms = float(np.sqrt(np.mean(residual[valid] ** 2))) if y.size else np.nan
        summaries.append(
            dict(
                predictor=label,
                amplitude=amp,
                correlation=float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan,
                observed_rms=yrms,
                residual_rms=rrms,
                residual_fraction=rrms / yrms if y.size and yrms else np.nan,
            )
        )
    pd.DataFrame(summaries).to_csv(out / "gradient_summary.csv", index=False)
    np.savez_compressed(
        out / "gradient_maps.npz",
        centers=CENTERS,
        counts=counts,
        observed=observed,
        predicted_unweighted=predicted_unweighted,
        predicted_weighted=predicted_weighted,
        residual_unweighted=residuals["self_consistent_normalization_field"],
        residual_weighted=residuals["wQ_weighted_closure"],
    )
    return summaries


def refresh_image_summaries(outdir):
    """Recompute fiducial image-only summaries from an existing images.npz.

    This does not replace event-level path-class or split-half diagnostics.  It
    is useful when the event table is unavailable but the saved images are.
    """
    out = Path(outdir)
    path = out / "images.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    counts = data["counts"]
    valid = valid_voxel_mask(counts, data["I_nom"], data["I_Q"])
    c_opt = optimal_global_scale(data["I_nom"], data["I_Q"], valid)
    data["I_scale_opt"] = data["I_nom"] / c_opt
    data["fiducial_mask"] = fiducial_voxel_mask()
    np.savez_compressed(path, **data)

    A = data["I_nom"] - data["I_Q"]
    C = data["I_const"] - data["I_Q"]
    S = data["I_scale_opt"] - data["I_Q"]
    R = data["I_p"] - data["I_Q"]
    artifact_rms = _rms(A[valid])
    const_rms = _rms(C[valid])
    scale_rms = _rms(S[valid])
    p_rms = _rms(R[valid])
    row = dict(
        artifact_rms=artifact_rms,
        const_residual_rms=const_rms,
        scale_opt_residual_rms=scale_rms,
        p_residual_rms=p_rms,
        const_reduction=1.0 - const_rms / artifact_rms,
        scale_opt_reduction=1.0 - scale_rms / artifact_rms,
        p_reduction=1.0 - p_rms / artifact_rms,
        c_opt=c_opt,
        denominator_scale_opt=math.sqrt(c_opt),
        n_valid_voxels=int(np.sum(valid)),
        n_fiducial_voxels=int(np.sum(fiducial_voxel_mask())),
        post_scalar_p_reduction=1.0 - p_rms / scale_rms,
        image_only_refresh=True,
    )
    pd.DataFrame([row]).to_csv(out / "artifact_summary.csv", index=False)
    images_for_roi = {k: v for k, v in data.items() if k.startswith("I_")}
    guard_gap_sensitivity(images_for_roi, counts).to_csv(
        out / "roi_guard_gap_sensitivity.csv", index=False
    )
    return row


def ensemble_artifact_summary(artifact_files, out_csv=None):
    rows = []
    for seed, f in enumerate(artifact_files):
        d = pd.read_csv(f).iloc[0]
        rows.append(dict(seed=seed, source=str(f), **d.to_dict()))
    raw = pd.DataFrame(rows)
    cols = [
        "artifact_rms", "const_residual_rms", "scale_opt_residual_rms", "p_residual_rms",
        "const_reduction", "scale_opt_reduction", "p_reduction", "post_scalar_p_reduction", "c_opt"
    ]
    summary_rows = []
    for col in cols:
        if col in raw:
            summary_rows.append(dict(metric=col, mean=float(raw[col].mean()), sd=float(raw[col].std(ddof=1)), n=len(raw)))
    summary = pd.DataFrame(summary_rows)
    if out_csv is not None:
        out_csv = Path(out_csv)
        summary.to_csv(out_csv, index=False)
        raw.to_csv(out_csv.with_name(out_csv.stem + "_raw.csv"), index=False)
    return summary


def ensemble_adaptive_summary(adaptive_files, out_csv=None):
    rows = []
    for seed, f in enumerate(adaptive_files):
        d = pd.read_csv(f)
        d = d[d.denominator == "all generated"].copy()
        d["seed"] = seed
        rows.append(d)
    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if raw.empty:
        return raw
    summary = (
        raw.groupby(["classification", "group"], as_index=False)
        .agg(retention_mean=("retention", "mean"), retention_sd=("retention", "std"), n_seeds=("seed", "count"))
    )
    if out_csv is not None:
        out_csv = Path(out_csv)
        summary.to_csv(out_csv, index=False)
        raw.to_csv(out_csv.with_name(out_csv.stem + "_raw.csv"), index=False)
    return summary


def refresh_gradient_summary(outdir):
    """Recompute the gradient summary from saved maps with the fiducial mask."""
    out = Path(outdir)
    with np.load(out / "gradient_maps.npz") as z:
        counts = z["counts"]
        observed = z["observed"]
        preds = {
            "self_consistent_normalization_field": z["predicted_unweighted"],
            "wQ_weighted_closure": z["predicted_weighted"],
        }
    rows = []
    for label, predicted in preds.items():
        valid = valid_voxel_mask(counts, observed, predicted)
        x = predicted[valid]
        y = observed[valid]
        amp = float((x @ y) / (x @ x)) if x.size and (x @ x) > 0 else np.nan
        residual = y - amp * x
        yrms = _rms(y)
        rrms = _rms(residual)
        rows.append(dict(
            predictor=label,
            amplitude=amp,
            correlation=float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan,
            observed_rms=yrms,
            residual_rms=rrms,
            residual_fraction=rrms / yrms if yrms else np.nan,
            n_valid_voxels=int(np.sum(valid)),
            image_only_refresh=True,
        ))
    d = pd.DataFrame(rows)
    d.to_csv(out / "gradient_summary.csv", index=False)
    return d



def _ensemble_grouped_csv(files, group_cols, value_cols, out_csv=None):
    """Combine matched-seed diagnostic CSVs into mean/SD summaries."""
    frames = []
    for seed, f in enumerate(files):
        d = pd.read_csv(f).copy()
        d["seed"] = seed
        d["source"] = str(f)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw
    agg = {}
    for col in value_cols:
        if col in raw.columns:
            agg[f"{col}_mean"] = (col, "mean")
            agg[f"{col}_sd"] = (col, "std")
    summary = raw.groupby(group_cols, as_index=False).agg(**agg, n_seeds=("seed", "nunique"))
    if out_csv is not None:
        out_csv = Path(out_csv)
        summary.to_csv(out_csv, index=False)
        raw.to_csv(out_csv.with_name(out_csv.stem + "_raw.csv"), index=False)
    return summary


def ensemble_guard_gap_summary(files, out_csv=None):
    return _ensemble_grouped_csv(
        files, ["image", "guard_gap_cm"],
        ["SNR_Pb", "CNR", "mean_Pb", "mean_Cu", "sd_Pb", "sd_Cu", "n_pb", "n_cu"],
        out_csv,
    )


def ensemble_weight_closure_summary(files, out_csv=None):
    cols = [
        "acceptance_fraction", "mean_p_ratio", "mean_p2_ratio", "sd_fractional_p_error",
        "mean_w_ideal_trueangle_truep_truepath", "mean_w_angle_only_recoangle_truep_truepath",
        "mean_w_path_only_trueangle_truep_recopath", "mean_w_momentum_only_trueangle_measp_truepath",
        "mean_w_Q_full", "mean_w_nom", "mean_w_p",
        "delta_w_angle_only_recoangle_truep_truepath_minus_ideal",
        "delta_w_path_only_trueangle_truep_recopath_minus_ideal",
        "delta_w_momentum_only_trueangle_measp_truepath_minus_ideal",
        "delta_w_Q_full_minus_ideal",
    ]
    return _ensemble_grouped_csv(files, ["p_set"], cols, out_csv)


def ensemble_roi_split_summary(files, out_csv=None):
    return _ensemble_grouped_csv(
        files, ["kind", "comparison"],
        ["SNR_full", "SNR_noise_full_est", "CNR_full", "CNR_noise_full_est"], out_csv,
    )


def ensemble_roi_spill_summary(files, out_csv=None):
    return _ensemble_grouped_csv(
        files, ["guard_gap_cm", "truth_group"],
        ["frac_in_pb_roi", "frac_in_cu_roi", "frac_in_neither"], out_csv,
    )

def paired_seed_summary(metric_files, out_csv=None):
    """Paired seed differences plus absolute metric means for each comparison."""
    rows = []
    comparisons = (
        ("I_nom", "I_Q"),
        ("I_p", "I_Q"),
    )
    for seed, f in enumerate(metric_files):
        d = pd.read_csv(f).set_index("image")
        for a, b in comparisons:
            if a in d.index and b in d.index:
                rows.append(
                    dict(
                        seed=seed,
                        comparison=f"{a}-{b}",
                        SNR_a=d.loc[a, "SNR_Pb"],
                        SNR_b=d.loc[b, "SNR_Pb"],
                        CNR_a=d.loc[a, "CNR"],
                        CNR_b=d.loc[b, "CNR"],
                        dSNR=d.loc[a, "SNR_Pb"] - d.loc[b, "SNR_Pb"],
                        dCNR=d.loc[a, "CNR"] - d.loc[b, "CNR"],
                    )
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("comparison")
        .agg(
            SNR_a_mean=("SNR_a", "mean"),
            SNR_b_mean=("SNR_b", "mean"),
            CNR_a_mean=("CNR_a", "mean"),
            CNR_b_mean=("CNR_b", "mean"),
            dSNR_mean=("dSNR", "mean"),
            dSNR_sd=("dSNR", "std"),
            dCNR_mean=("dCNR", "mean"),
            dCNR_sd=("dCNR", "std"),
            n=("seed", "count"),
        )
        .reset_index()
        if not raw.empty
        else raw
    )
    if out_csv is not None:
        summary.to_csv(out_csv, index=False)
        raw_path = Path(out_csv).with_name(Path(out_csv).stem + "_raw.csv")
        raw.to_csv(raw_path, index=False)
    return summary
