"""Theory tables and detector-level analysis for the revised study."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CU_HALF, CU_ROI_R, CU_ROI_ZHALF, MATERIALS, MIN_VOX_COUNT, MOMENTA,
    N_VOX, PB_CX, PB_CY, PB_ROI_R, PB_ROI_ZHALF, RADIAL_ETA_MAX,
    THETA_CUT, VOX_HALF,
)
from physics import (
    Layer, PofxCache, calibrate_pofx, constant_calibration, epsilon_asymptotic,
    fit_eta1, fit_log_asymptote, mu2_eta, optimal_k_constant, radial_tail_ratio, reduced_parameters,
    theta0_highland, theta_space_highland,
)
from simulation import SEG_NAMES, segment_matrix

EDGES = np.linspace(-VOX_HALF, VOX_HALF, N_VOX + 1)
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])


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
        rows.append(dict(p=p, R=r["R"], B=r["B"], k=r["k"], eta_cut=r["eta_cut"],
                         epsilon=r["epsilon"], k_opt=kopt, eta_eff=eff,
                         exact_error=(1+r["epsilon"])**2-r["exact_ratio2"],
                         clipped_fraction=r["clipped_fraction"]))
    collapse = pd.DataFrame(rows)
    collapse.to_csv(out / "theory_collapse.csv", index=False)

    # p(X) energy-loss table.  Two explicit screening-log continuations are
    # carried so their finite-loss spread is a visible model systematic rather
    # than an arbitrary hidden convention.
    rows = []
    for label, path in (("AlCu", AXIAL_ORDERED), ("Al25", OFFCU_ORDERED)):
        for p in MOMENTA:
            rd = calibrate_pofx(path, p, THETA_CUT, nmax=2, screening_weight="dchi_c2")
            rs = calibrate_pofx(path, p, THETA_CUT, nmax=2, screening_weight="serial")
            rows.append(dict(
                path=label, p=p, x_over_x0=rd["x_over_x0"],
                dp_over_p=rd["dp_over_p"], delta_E_MeV=1000*rd["delta_E"],
                n_slices=rd["n_slices"], eta_cut=rd["eta_cut"], Fc=rd["Fc"],
                B_dchi=rd["B"], B_serial=rs["B"],
                epsilon_matched_dchi=rd["epsilon_matched"],
                epsilon_matched_serial=rs["epsilon_matched"],
                epsilon_matched_spread_pp=100*abs(rd["epsilon_matched"]-rs["epsilon_matched"]),
                epsilon_mixed_dchi=rd["epsilon_mixed"],
                epsilon_mixed_serial=rs["epsilon_mixed"],
                epsilon_mixed_spread_pp=100*abs(rd["epsilon_mixed"]-rs["epsilon_mixed"]),
                E_w_nominal_dchi=(1+rd["epsilon_mixed"])**2,
                E_w_nominal_serial=(1+rs["epsilon_mixed"])**2,
                clipped_fraction=max(rd["clipped_fraction"], rs["clipped_fraction"])))
    pofx = pd.DataFrame(rows)
    pofx.to_csv(out / "energy_loss_calibration.csv", index=False)

    # Log-law protocol with the numerical-table and analytic-tail roles kept
    # separate.  Slope diagnostics are restricted to eta <= RADIAL_ETA_MAX so
    # the fitted coefficient is actually tested against Phi^(1)/Phi^(2), not
    # imposed by the analytic continuation.  Deep windows are retained only to
    # stabilize the asymptotic intercept eta1.
    slope_windows = ((8.0, 20.0, 41),
                     (10.0, 30.0, 41),
                     (15.0, 30.0, 41))
    intercept_windows = ((30.0, 100.0, 41),
                         (50.0, 500.0, 61))
    eta_rows = []
    for name, X in PATHS.items():
        rp = reduced_parameters(X, 1.0)
        for nmax in (1, 2):
            for role, windows in (("slope_diagnostic", slope_windows),
                                  ("eta1_asymptote", intercept_windows)):
                for eta_lo, eta_hi, n_eta in windows:
                    eta_grid = np.geomspace(eta_lo, eta_hi, n_eta)
                    eps = []
                    for eta in eta_grid:
                        cut = eta * math.sqrt(rp["chi_c2"] * rp["B"])
                        eps.append(constant_calibration(X, 1.0, cut, nmax=nmax)["epsilon"])
                    joint = fit_log_asymptote(eta_grid, eps, R=rp["R"])
                    eta1_fixed = fit_eta1(eta_grid, eps, rp["R"])
                    tail_mask = eta_grid > RADIAL_ETA_MAX
                    eta_rows.append(dict(
                        path=name, nmax=nmax, window_role=role,
                        eta_min=eta_lo, eta_max=eta_hi,
                        eta_table_max=RADIAL_ETA_MAX,
                        analytic_tail_point_fraction=float(np.mean(tail_mask)),
                        R=rp["R"], B=rp["B"], sqrt2RB=rp["sqrt2RB"],
                        slope_fit=joint["slope"], slope_expected=joint["slope_expected"],
                        slope_ratio=joint["slope_ratio"],
                        slope_rel_error=joint["slope_rel_error"],
                        intercept_fit=joint["intercept"], eta1_joint=joint["eta1"],
                        eta1_fixed_2R=eta1_fixed,
                        rms_residual=joint["rms_residual"],
                        max_abs_residual=joint["max_abs_residual"],
                    ))
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
            comp.append(dict(match="k", value=k, path=name, R=r["R"], B=r["B"],
                             eta_cut=r["eta_cut"], epsilon=r["epsilon"]))
        for eta in (3.0, 5.5, 10.0, 20.0):
            cut = eta * math.sqrt(rp["chi_c2"] * rp["B"])
            r = constant_calibration(X, 1.0, cut, nmax=2)
            comp.append(dict(match="eta", value=eta, path=name, R=r["R"], B=r["B"],
                             eta_cut=r["eta_cut"], epsilon=r["epsilon"]))
    pd.DataFrame(comp).to_csv(out / "composition_matched.csv", index=False)

    # n<=1 vs n<=2 convergence and tail check.
    conv = []
    for p in MOMENTA:
        r1 = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=1)
        r2 = constant_calibration(PATHS["AlCu"], p, THETA_CUT, nmax=2)
        conv.append(dict(p=p, eps_n1=r1["epsilon"], eps_n2=r2["epsilon"],
                         shift_abs=r2["epsilon"]-r1["epsilon"],
                         shift_rel=(r2["epsilon"]/r1["epsilon"]-1.0) if r1["epsilon"] else np.nan,
                         clip_n1=r1["clipped_fraction"], clip_n2=r2["clipped_fraction"]))
    pd.DataFrame(conv).to_csv(out / "truncation_convergence.csv", index=False)

    # Tail check at 6 GeV/c so 50--200 mrad is actually in the Rutherford
    # approach region; at 1 GeV/c these angles still overlap the core.
    r = constant_calibration(PATHS["AlCu"], 6.0, THETA_CUT, nmax=2)
    s_mol = math.sqrt(r["chi_c2"] * r["B"])
    tail = pd.DataFrame([
        dict(p_GeV=6.0, theta_mrad=th*1000, eta=th/s_mol,
             ratio=radial_tail_ratio(th, r["chi_c2"], r["B"], nmax=2))
        for th in (0.050, 0.100, 0.200)
    ])
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


def _central_p_only_eps(p, cache):
    p = np.asarray(p, float)
    seg = np.tile(np.array([5.0, 15.0, 0.0, 0.0, 5.0]), (p.size, 1))
    return cache.arrays(p, seg, THETA_CUT)["epsilon_mixed"]


def build_weights(df, cache=None):
    """Fixed-cut weights with all energy-loss quantities kept distinct."""
    cache = cache or PofxCache(nmax=2)
    p = df.p_meas.to_numpy(float)
    dth = df.dth_reco.to_numpy(float)
    xx0 = df.xx0_ref_reco.to_numpy(float)
    tspace_incident = theta_space_highland(p, xx0)
    q_reco = cache.arrays(p, segment_matrix(df, "ref_reco"), THETA_CUT)
    q_true = cache.arrays(df.p_true.to_numpy(float), segment_matrix(df, "ref_true"), THETA_CUT)

    w_nom = np.divide(dth, tspace_incident, out=np.zeros_like(dth), where=tspace_incident>0)**2
    eps_p = _central_p_only_eps(p, cache)
    den_p = (1.0 + eps_p) * tspace_incident
    w_p = np.divide(dth, den_p, out=np.zeros_like(dth), where=den_p>0)**2
    w_Q = np.divide(dth, q_reco["theta_rms"], out=np.zeros_like(dth), where=q_reco["theta_rms"]>0)**2
    w_ideal = np.divide(df.dth_true.to_numpy(float), q_true["theta_rms"],
                        out=np.zeros(len(df)), where=q_true["theta_rms"]>0)**2

    # Two means are reported because the published 12.18% used a weighted
    # average, whereas an unweighted event-count mean is the cleaner scale-null
    # control.  I_const deliberately uses the event-count mean.
    eps_event = np.divide(q_reco["theta_rms"], tspace_incident,
                          out=np.zeros_like(dth), where=tspace_incident>0) - 1.0
    finite = np.isfinite(eps_event)
    eps_bar_event = float(np.mean(eps_event[finite]))
    pos = finite & np.isfinite(w_nom) & (w_nom > 0)
    eps_bar_weighted = float(np.average(eps_event[pos], weights=w_nom[pos])) if np.any(pos) else np.nan
    den_const = (1.0 + eps_bar_event) * tspace_incident
    w_const = np.divide(dth, den_const, out=np.zeros_like(dth), where=den_const>0)**2

    return dict(I_nom=w_nom, I_p=w_p, I_Q=w_Q, I_ideal=w_ideal, I_const=w_const,
                eps_event=eps_event, eps_p=eps_p,
                eps_bar_event_mean=eps_bar_event,
                eps_bar_nominal_weighted=eps_bar_weighted,
                theta_rms_reco=q_reco["theta_rms"], p_out_reco=q_reco["p_out"]), cache


def roi_masks():
    X, Y, Z = np.meshgrid(CENTERS, CENTERS, CENTERS, indexing="ij")
    pb = ((X-PB_CX)**2 + (Y-PB_CY)**2 <= PB_ROI_R**2) & (np.abs(Z) <= PB_ROI_ZHALF)
    cu = (X**2 + Y**2 <= CU_ROI_R**2) & (np.abs(Z) <= CU_ROI_ZHALF) & ~pb
    return pb, cu


def image_metrics(img, counts):
    pb, cu = roi_masks()
    pb &= counts >= MIN_VOX_COUNT
    cu &= counts >= MIN_VOX_COUNT
    a, b = img[pb], img[cu]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return dict(SNR_Pb=np.nan, CNR=np.nan, n_pb=len(a), n_cu=len(b))
    return dict(SNR_Pb=float(a.mean()/a.std(ddof=1)),
                CNR=float((a.mean()-b.mean())/b.std(ddof=1)),
                n_pb=len(a), n_cu=len(b))


def path_residual_diagnostics(df, weights):
    """Step 3: compare I_p-I_Q in truth Al-only and Cu-bearing event classes."""
    seg = segment_matrix(df, "ref_true")
    al_only = (seg[:,1] + seg[:,2] + seg[:,3]) <= 1e-8
    cu = (seg[:,1] + seg[:,3]) > 1e-8
    out = []
    for name, mask in (("Al-only", al_only), ("Cu-bearing", cu)):
        d = df.loc[mask]
        if d.empty:
            continue
        ip, c = image_from_events(d, weights["I_p"][mask])
        iq, _ = image_from_events(d, weights["I_Q"][mask])
        valid = (c >= MIN_VOX_COUNT) & np.isfinite(ip) & np.isfinite(iq)
        out.append(dict(region=name, n_events=int(mask.sum()),
                        event_rms=float(np.sqrt(np.mean((weights["I_p"][mask]-weights["I_Q"][mask])**2))),
                        image_rms=float(np.sqrt(np.mean((ip[valid]-iq[valid])**2))) if np.any(valid) else np.nan))
    return pd.DataFrame(out)



def path_class_migration(df):
    """Truth-vs-reconstructed Al-only classification for the off-Cu test."""
    st = segment_matrix(df, "ref_true")
    sr = segment_matrix(df, "ref_reco")
    t = (st[:,1] + st[:,2] + st[:,3]) <= 1e-8
    r = (sr[:,1] + sr[:,2] + sr[:,3]) <= 1e-8
    return pd.DataFrame([
        dict(truth_al_only=bool(tv), reco_al_only=bool(rv), n=int(np.sum((t==tv)&(r==rv))))
        for tv in (False,True) for rv in (False,True)
    ])

def adaptive_retention(df, k_opt=1.800):
    """Step 6.1 adaptive-selection diagnostic with explicit denominators.

    Primary retention is defined relative to *all generated events* in each
    truth/reconstructed class.  A second conditional row reports retention
    among events that also satisfy the fixed 200 mrad cut, solely to make the
    denominator dependence transparent.
    """
    cut = k_opt * theta0_highland(df.p_meas.to_numpy(float), df.xx0_ref_reco.to_numpy(float))
    keep = df.dth_reco.to_numpy(float) < cut
    fixed = df.pass_reco.to_numpy(bool)
    pb_truth = df.true_pb.to_numpy(bool)
    cu_truth = df.true_cu_only.to_numpy(bool)
    rpb = (df.poca_x.to_numpy()-PB_CX)**2 + (df.poca_y.to_numpy()-PB_CY)**2 <= PB_ROI_R**2
    rpb &= np.abs(df.poca_z.to_numpy()) <= PB_ROI_ZHALF
    rcu = df.poca_x.to_numpy()**2 + df.poca_y.to_numpy()**2 <= CU_ROI_R**2
    rcu &= np.abs(df.poca_z.to_numpy()) <= CU_ROI_ZHALF
    rcu &= ~rpb

    rows = []
    for classification, groups in (
        ("truth", (("Pb-crossing", pb_truth), ("Cu-only", cu_truth))),
        ("reconstructed", (("Pb ROI", rpb), ("Cu ROI", rcu))),
    ):
        for group, mask in groups:
            n_all = int(mask.sum())
            rows.append(dict(classification=classification, group=group,
                             denominator="all generated", n=n_all,
                             retention=float(np.mean(keep[mask])) if n_all else np.nan))
            cond = mask & fixed
            n_cond = int(cond.sum())
            rows.append(dict(classification=classification, group=group,
                             denominator="fixed-cut accepted", n=n_cond,
                             retention=float(np.mean(keep[cond])) if n_cond else np.nan))
    return pd.DataFrame(rows)


def analyze_events(df, outdir, cache=None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    use = df[df.pass_reco].reset_index(drop=True)
    weights, cache = build_weights(use, cache=cache)
    images = {}; counts = None; metric_rows=[]
    for name in ("I_nom","I_p","I_Q","I_ideal","I_const"):
        img, c = image_from_events(use, weights[name])
        images[name] = img; counts = c
        metric_rows.append(dict(image=name, **image_metrics(img, c)))
    pd.DataFrame(metric_rows).to_csv(out/"metrics.csv", index=False)
    path_residual_diagnostics(use, weights).to_csv(out/"path_residuals.csv", index=False)
    path_class_migration(use).to_csv(out/"path_class_migration.csv", index=False)
    adaptive_retention(df).to_csv(out/"adaptive_retention.csv", index=False)

    # Nominal-minus-Q artifact remains descriptive unless the gradient experiment is used.
    valid = (counts >= MIN_VOX_COUNT) & np.isfinite(images["I_nom"]) & np.isfinite(images["I_Q"])
    A = images["I_nom"] - images["I_Q"]
    R = images["I_p"] - images["I_Q"]
    artifact = dict(artifact_rms=float(np.sqrt(np.mean(A[valid]**2))),
                    p_residual_rms=float(np.sqrt(np.mean(R[valid]**2))))
    artifact["p_reduction"] = 1.0 - artifact["p_residual_rms"]/artifact["artifact_rms"]
    pd.DataFrame([artifact]).to_csv(out/"artifact_summary.csv", index=False)

    np.savez_compressed(out/"images.npz", centers=CENTERS, counts=counts, **images)
    pd.DataFrame([dict(
        eps_bar_event_mean=weights["eps_bar_event_mean"],
        eps_bar_nominal_weighted=weights["eps_bar_nominal_weighted"],
        I_const_uses="event-count mean",
        mean_dp_over_p=float(np.mean(weights["p_out_reco"]/use.p_meas.to_numpy()-1.0)),
        screening_weight=cache.screening_weight,
        max_clipped=cache.max_clipped)]).to_csv(out/"calibration_summary.csv", index=False)
    return images, counts, weights, cache


def analyze_gradient(df, outdir, cache=None):
    """Step 5 direct causal test: observed difference, predicted field, residual."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    use = df[df.pass_reco].reset_index(drop=True)
    weights, cache = build_weights(use, cache=cache)
    I_nom, counts = image_from_events(use, weights["I_nom"])
    I_Q, _ = image_from_events(use, weights["I_Q"])
    observed = I_nom - I_Q

    # Mechanism predictor: the local mean normalization factor alone.  This is
    # intentionally *not* an algebraic identity because it omits eventwise w_Q.
    excess = (1.0 + weights["eps_event"])**2 - 1.0
    predicted_unweighted, _ = image_from_events(use, excess)

    # Algebraic closure predictor.  Since w_nom-w_Q = w_Q*((1+eps)^2-1) for
    # the same event, this should reproduce the observed map up to roundoff and
    # serves as a code/weighting closure check, not as independent evidence.
    predicted_weighted, _ = image_from_events(use, weights["I_Q"] * excess)

    summaries = []
    residuals = {}
    for label, predicted in (("normalization_field", predicted_unweighted),
                             ("wQ_weighted_closure", predicted_weighted)):
        valid = (counts >= MIN_VOX_COUNT) & np.isfinite(observed) & np.isfinite(predicted)
        x = predicted[valid]; y = observed[valid]
        amp = float((x @ y)/(x @ x)) if x.size and (x@x)>0 else np.nan
        residual = observed - amp * predicted
        residuals[label] = residual
        yrms = float(np.sqrt(np.mean(y*y))) if y.size else np.nan
        rrms = float(np.sqrt(np.mean(residual[valid]**2))) if y.size else np.nan
        summaries.append(dict(
            predictor=label, amplitude=amp,
            correlation=float(np.corrcoef(x,y)[0,1]) if x.size > 1 else np.nan,
            observed_rms=yrms, residual_rms=rrms,
            residual_fraction=rrms/yrms if y.size and yrms else np.nan))
    pd.DataFrame(summaries).to_csv(out/"gradient_summary.csv", index=False)
    np.savez_compressed(
        out/"gradient_maps.npz", centers=CENTERS, counts=counts, observed=observed,
        predicted_unweighted=predicted_unweighted,
        predicted_weighted=predicted_weighted,
        residual_unweighted=residuals["normalization_field"],
        residual_weighted=residuals["wQ_weighted_closure"])
    return summaries


def paired_seed_summary(metric_files, out_csv=None):
    """Paired seed differences for schemes sharing the same event realizations."""
    rows=[]
    for seed, f in enumerate(metric_files):
        d=pd.read_csv(f).set_index("image")
        for a,b in (("I_nom","I_Q"),("I_p","I_Q")):
            if a in d.index and b in d.index:
                rows.append(dict(seed=seed, comparison=f"{a}-{b}",
                                 dSNR=d.loc[a,"SNR_Pb"]-d.loc[b,"SNR_Pb"],
                                 dCNR=d.loc[a,"CNR"]-d.loc[b,"CNR"]))
    raw=pd.DataFrame(rows)
    summary=(raw.groupby("comparison").agg(
        dSNR_mean=("dSNR","mean"), dSNR_sd=("dSNR","std"),
        dCNR_mean=("dCNR","mean"), dCNR_sd=("dCNR","std"), n=("seed","count")
    ).reset_index() if not raw.empty else raw)
    if out_csv is not None:
        summary.to_csv(out_csv,index=False)
    return summary
