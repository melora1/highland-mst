#!/usr/bin/env python3
"""Compare supplied Geant4 exit-angle dumps with the radial model.

The reported fractional RMS convention is the manuscript convention

    rms_frac = theta_rms_model / theta_rms_G4 - 1.

Single-material Cu/Pb slabs are supported directly so this script matches the
supplied Geant4 executable.  Reference-path AlCu/Al25 comparisons are retained
for future layered transport dumps.  The script also reports the implied bias
of a quadratic denominator, core-width comparison, delta-method sampling
intervals, and a finite-u band decomposition of the second-moment numerator.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from analysis import AXIAL_ORDERED, OFFCU_ORDERED, PATHS
from config import M_E, M_MU, MATERIALS
from physics import (
    HBARC_MEV_FM,
    Layer,
    calibrate_pofx,
    constant_calibration,
    nuclear_radius_fm,
    _constant_transform_calibration,
    _normalize_ff_model,
)

PATH_ORDERED = {"AlCu": AXIAL_ORDERED, "Al25": OFFCU_ORDERED}


def load_exit_angles(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        d = pd.read_csv(p)
        if "theta_space" not in d.columns:
            raise ValueError(f"{p}: CSV needs theta_space column")
        a = d.theta_space.to_numpy(float)
        tx = d.theta_x.to_numpy(float) if "theta_x" in d else None
        ty = d.theta_y.to_numpy(float) if "theta_y" in d else None
    else:
        raw = np.loadtxt(p)
        if raw.ndim > 1:
            a = raw[:, 0].astype(float)
            tx = raw[:, 1].astype(float) if raw.shape[1] >= 3 else None
            ty = raw[:, 2].astype(float) if raw.shape[1] >= 3 else None
        else:
            a = np.asarray(raw, float)
            tx = ty = None
    keep = np.isfinite(a) & (a >= 0.0)
    if tx is not None and ty is not None:
        keep &= np.isfinite(tx) & np.isfinite(ty)
        tx, ty = tx[keep], ty[keep]
    a = a[keep]
    if a.size == 0:
        raise ValueError(f"{p}: no finite non-negative exit angles")
    return a, tx, ty


def load_angles(path):
    """Backward-compatible radial-angle loader."""
    return load_exit_angles(path)[0]


def projected_gaussian_fit(theta_x, theta_y, central_fraction=0.98):
    """Truncated-normal MLE for the central projected-angle fraction."""
    if theta_x is None or theta_y is None:
        return dict(sigma=np.nan, cut=np.nan, n=0, central_fraction=central_fraction)
    values = np.concatenate((np.asarray(theta_x, float), np.asarray(theta_y, float)))
    cut = float(np.quantile(np.abs(values), float(central_fraction)))
    central = values[np.abs(values) <= cut]
    scale = float(np.std(central, ddof=0))
    if not (cut > 0.0 and scale > 0.0):
        return dict(sigma=np.nan, cut=cut, n=int(central.size), central_fraction=central_fraction)

    def objective(log_sigma):
        sigma = math.exp(float(log_sigma))
        mass = 2.0 * ndtr(cut / sigma) - 1.0
        return (
            central.size * math.log(sigma)
            + float(np.sum(central * central)) / (2.0 * sigma * sigma)
            + central.size * math.log(max(mass, np.finfo(float).tiny))
        )

    fit = minimize_scalar(
        objective,
        bounds=(math.log(0.25 * scale), math.log(4.0 * scale)),
        method="bounded",
    )
    return dict(
        sigma=math.exp(float(fit.x)), cut=cut, n=int(central.size),
        central_fraction=float(central_fraction), fit_success=bool(fit.success),
    )


def sample_moments(a, cut, n_generated=None):
    a = np.asarray(a, float)
    denom = int(n_generated) if n_generated is not None else a.size
    if denom < a.size:
        raise ValueError("n_generated cannot be smaller than the number of recorded exits")
    keep = a[a < cut]
    if keep.size == 0:
        return dict(Fc=0.0, theta_rms=np.nan, M2=np.nan, M4=np.nan, n=0, exit_fraction=a.size / denom)
    M2 = float(np.mean(keep**2))
    M4 = float(np.mean(keep**4))
    theta = math.sqrt(M2)
    var_m2 = max(M4 - M2 * M2, 0.0) / keep.size
    se_theta = math.sqrt(var_m2) / (2.0 * theta) if theta > 0 else np.nan
    return dict(
        Fc=keep.size / denom,
        theta_rms=theta,
        M2=M2,
        M4=M4,
        n=int(keep.size),
        exit_fraction=a.size / denom,
        se_theta_rms=se_theta,
    )


def make_model_spec(args):
    if args.material is not None:
        if args.thickness_cm is None:
            raise ValueError("--material requires --thickness-cm")
        X = {m: 0.0 for m in MATERIALS}
        X[args.material] = float(args.thickness_cm) * MATERIALS[args.material].rho
        ordered = (Layer(args.material, float(args.thickness_cm)),)
        label = f"{args.material}{args.thickness_cm:g}cm"
        return label, X, ordered
    if args.thickness_cm is not None:
        raise ValueError("--thickness-cm requires --material")
    path = args.path or "AlCu"
    return path, PATHS[path], PATH_ORDERED[path]


def calibrator(X, ordered, p, energy_loss, ff_model, floor):
    model = _normalize_ff_model(ff_model)
    if energy_loss:
        from physics import calibrate_pofx_transform
        return lambda cut: calibrate_pofx_transform(
            ordered, p, theta_cut=float(cut), form_factor=model,
            include_incoherent=bool(floor),
        )
    return lambda cut: _constant_transform_calibration(
        X, p, float(cut), model, bool(floor)
    )


def model_core_and_theta0(base, energy_loss):
    if energy_loss:
        return float(base["theta_space_pofx"]), float(base["theta0_pofx"])
    return float(base["theta_space"]), float(base["theta_space"] / math.sqrt(2.0))


def band_decomposition(angles, model_at, theta0, bands, n_generated=None):
    angles = np.asarray(angles, float)
    denom = int(n_generated) if n_generated is not None else angles.size
    rows = []
    cumulative = {}
    for b in bands:
        if b <= 0:
            cumulative[b] = (0.0, 0.0)
        else:
            q = model_at(b * theta0)
            cumulative[b] = (float(q["Fc"]), float(q["Fc"] * q["M2"]))
    for lo, hi in zip(bands[:-1], bands[1:]):
        low = lo * theta0
        high = hi * theta0
        mask = (angles >= low) & (angles < high)
        prob_g4 = float(np.sum(mask) / denom)
        m2_num_g4 = float(np.sum(angles[mask] ** 2) / denom)
        Fc_lo, num_lo = cumulative[lo]
        Fc_hi, num_hi = cumulative[hi]
        rows.append(
            dict(
                u_lo=lo,
                u_hi=hi,
                theta_lo=low,
                theta_hi=high,
                prob_g4=prob_g4,
                prob_model=Fc_hi - Fc_lo,
                m2_numerator_g4=m2_num_g4,
                m2_numerator_model=num_hi - num_lo,
                m2_numerator_diff_model_minus_g4=(num_hi - num_lo) - m2_num_g4,
            )
        )
    return rows


def band_decomposition_pairs(angles, model_at, theta0, band_pairs, n_generated=None):
    """Band decomposition for explicit pairs, permitting collapsed low-u bands."""
    angles = np.asarray(angles, float)
    denom = int(n_generated) if n_generated is not None else angles.size
    cumulative = {}
    for edge in sorted({float(x) for pair in band_pairs for x in pair}):
        if edge <= 0.0:
            cumulative[edge] = (0.0, 0.0)
        else:
            q = model_at(edge * theta0)
            cumulative[edge] = (float(q["Fc"]), float(q["Fc"] * q["M2"]))
    rows = []
    for lo, hi in band_pairs:
        if hi <= lo:
            continue
        low, high = lo * theta0, hi * theta0
        mask = (angles >= low) & (angles < high)
        Fc_lo, num_lo = cumulative[lo]
        Fc_hi, num_hi = cumulative[hi]
        rows.append(dict(
            u_lo=lo, u_hi=hi, theta_lo=low, theta_hi=high,
            prob_g4=float(np.sum(mask) / denom), prob_model=Fc_hi-Fc_lo,
            m2_numerator_g4=float(np.sum(angles[mask]**2) / denom),
            m2_numerator_model=num_hi-num_lo,
            m2_numerator_diff_model_minus_g4=(num_hi-num_lo)
            - float(np.sum(angles[mask]**2) / denom),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True, help="label=angles.txt; repeat for each reference-list/configuration label")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--path", choices=["AlCu", "Al25"], default=None)
    target.add_argument("--material", choices=["Cu", "Pb", "Al"], default=None)
    ap.add_argument("--thickness-cm", type=float, default=None)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--energy-loss", action="store_true", help="compare against segmented p(X); default is constant-p")
    ap.add_argument("--ff-model", choices=("point", "gauss", "sphere"), required=True)
    ap.add_argument("--floor", choices=("on", "off"), required=True)
    ap.add_argument("--k", type=float, nargs="+", default=[2, 5, 10, 20, 40])
    ap.add_argument("--theta-cut-mrad", type=float, nargs="+", default=[200.0], help="physical angular cuts to evaluate in addition to --k")
    ap.add_argument("--u-bands", type=float, nargs="+", default=None, help="optional reduced-angle edges; default uses theta_FF/theta_nuc per run")
    ap.add_argument("--n-generated", type=int, default=None, help="number of primaries; permits exit fraction and Fc relative to all generated events")
    ap.add_argument(
        "--n-generated-label", action="append", default=[],
        help="label=count; per-transport primary count (overrides --n-generated)",
    )
    ap.add_argument("--out", default=None, help="truncated-moment CSV")
    ap.add_argument("--core-out", default=None, help="core-width CSV; defaults beside --out")
    ap.add_argument("--bands-out", default=None, help="band-decomposition CSV; defaults beside --out")
    a = ap.parse_args()

    files = {}
    for spec in a.file:
        if "=" not in spec:
            raise ValueError("--file must be label=path")
        label, path = spec.split("=", 1)
        files[label] = path
    data = {lab: load_exit_angles(f) for lab, f in files.items()}
    generated_by_label = {}
    for spec in a.n_generated_label:
        if "=" not in spec:
            raise ValueError("--n-generated-label must be label=count")
        label, count = spec.split("=", 1)
        if label not in files:
            raise ValueError(f"generated-count label {label!r} has no --file")
        generated_by_label[label] = int(count)

    def generated_count(label, n_exit):
        value = generated_by_label.get(label, a.n_generated)
        return int(n_exit if value is None else value)

    target_label, X, ordered = make_model_spec(a)
    floor = a.floor == "on"
    model_at = calibrator(X, ordered, a.p, a.energy_loss, a.ff_model, floor)
    base = model_at(0.2)
    model_core, theta0 = model_core_and_theta0(base, a.energy_loss)

    core_rows = []
    for lab, (ang, theta_x, theta_y) in data.items():
        n_generated = generated_count(lab, ang.size)
        # For a Rayleigh radial core h(theta)=2 theta/s^2 exp(-theta^2/s^2),
        # median = s*sqrt(ln 2) and s is the radial Gaussian-core RMS.
        core_g4 = float(np.median(ang) / math.sqrt(math.log(2.0)))
        projected = projected_gaussian_fit(theta_x, theta_y, 0.98)
        core_rows.append(
            dict(
                transport=lab,
                ff_model=a.ff_model,
                floor=floor,
                target=target_label,
                p=a.p,
                n_exit=ang.size,
                n_generated=n_generated,
                exit_fraction=ang.size / n_generated,
                theta_space_model=model_core,
                theta_space_g4_median=core_g4,
                core_frac_model_over_g4=model_core / core_g4 - 1.0,
                projected_fit_fraction=projected["central_fraction"],
                projected_fit_n=projected["n"],
                projected_fit_cut=projected["cut"],
                theta0_model=theta0,
                theta0_g4_projected_fit=projected["sigma"],
                theta0_frac_model_over_g4=(
                    theta0 / projected["sigma"] - 1.0
                    if np.isfinite(projected["sigma"]) and projected["sigma"] > 0.0
                    else np.nan
                ),
            )
        )

    rows = []
    band_rows = []
    theta_e = theta_ff = theta_nuc = np.nan
    if a.u_bands is not None:
        bands = sorted(set(float(x) for x in a.u_bands))
        if len(bands) < 2 or bands[0] < 0:
            raise ValueError("--u-bands needs at least two non-negative edges")
        band_pairs = list(zip(bands[:-1], bands[1:]))
    else:
        if a.material is None:
            raise ValueError("automatic FF bands require --material")
        theta_ff = HBARC_MEV_FM / (
            1000.0 * a.p * nuclear_radius_fm(MATERIALS[a.material].A)
        )
        theta_nuc = HBARC_MEV_FM / (1000.0 * a.p * 0.84)
        theta_e = M_E / M_MU
        u_e, u_ff, u_nuc = (
            theta_e / theta0, theta_ff / theta0, theta_nuc / theta0
        )
        u_max = max(float(max(np.max(value[0]) for value in data.values()) / theta0), u_nuc)
        # Keep the diagnostic reduced-angle boundaries while making the three
        # physical regime boundaries exact bin edges.
        edges = sorted(set(
            [0.0, 1.0, 3.0, u_e, 0.5*u_ff, u_ff, 2.0*u_ff, u_nuc, u_max]
        ))
        band_pairs = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
                      if hi > lo]
    for lab, (ang, _theta_x, _theta_y) in data.items():
        n_generated = generated_count(lab, ang.size)
        band_rows.extend(
            dict(
                transport=lab, ff_model=a.ff_model, floor=floor,
                target=target_label, p=a.p, theta_e=theta_e,
                theta_FF=theta_ff, theta_nuc=theta_nuc, **r
            )
            for r in band_decomposition_pairs(
                ang, model_at, theta0, band_pairs, n_generated
            )
        )
        cuts = [("reduced_k", float(k) * theta0, float(k)) for k in a.k]
        cuts += [("physical", float(mrad) * 1e-3, float(mrad) * 1e-3 / theta0) for mrad in a.theta_cut_mrad]
        seen = set()
        for cut_kind, cut, k in cuts:
            key = round(float(cut), 14)
            if key in seen:
                continue
            seen.add(key)
            q = model_at(cut)
            g = sample_moments(ang, cut, n_generated)
            if np.isfinite(g["theta_rms"]) and g["theta_rms"] > 0:
                rms_frac = float(q["theta_rms"] / g["theta_rms"] - 1.0)
                weight_bias = float((g["theta_rms"] / q["theta_rms"]) ** 2 - 1.0)
                se_frac = float(abs(q["theta_rms"] / g["theta_rms"] ** 2) * g["se_theta_rms"])
            else:
                rms_frac = weight_bias = se_frac = np.nan
            rows.append(
                dict(
                    transport=lab,
                    ff_model=a.ff_model,
                    floor=floor,
                    target=target_label,
                    p=a.p,
                    energy_loss=bool(a.energy_loss),
                    cut_kind=cut_kind,
                    k=k,
                    theta_cut=cut,
                    theta_cut_mrad=1000.0 * cut,
                    n_accept=g["n"],
                    exit_fraction=g["exit_fraction"],
                    Fc_g4=g["Fc"],
                    Fc_model=q["Fc"],
                    Fc_diff_model_minus_g4=q["Fc"] - g["Fc"],
                    theta_rms_g4=g["theta_rms"],
                    theta_rms_model=q["theta_rms"],
                    rms_frac_model_over_g4=rms_frac,
                    rms_frac_se_delta=se_frac,
                    rms_frac_ci95_lo=rms_frac - 1.96 * se_frac if np.isfinite(se_frac) else np.nan,
                    rms_frac_ci95_hi=rms_frac + 1.96 * se_frac if np.isfinite(se_frac) else np.nan,
                    quadratic_weight_bias_if_g4_true=weight_bias,
                    M4_g4=g["M4"],
                    M4_model=q["M4"],
                    M4_frac_model_over_g4=(q["M4"] / g["M4"] - 1.0) if g["M4"] and np.isfinite(g["M4"]) else np.nan,
                )
            )

    d = pd.DataFrame(rows)
    core = pd.DataFrame(core_rows)
    bands_df = pd.DataFrame(band_rows)
    print(d.to_string(index=False))
    print("\nCore-width comparison:\n", core.to_string(index=False))
    if len(files) >= 2:
        dk = d[d.cut_kind == "reduced_k"]
        pivot = dk.pivot(index="k", columns="transport", values="theta_rms_g4")
        cols = list(pivot.columns)
        if len(cols) >= 2:
            spread = (pivot[cols[0]] / pivot[cols[1]] - 1.0).rename("transport_spread_rms")
            print("\nSupplied-transport RMS spread:\n", spread.to_string())

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, index=False)
        core_out = Path(a.core_out) if a.core_out else out.with_name(out.stem + "_core.csv")
        bands_out = Path(a.bands_out) if a.bands_out else out.with_name(out.stem + "_bands.csv")
        core.to_csv(core_out, index=False)
        bands_df.to_csv(bands_out, index=False)
    elif a.core_out or a.bands_out:
        if a.core_out:
            core.to_csv(a.core_out, index=False)
        if a.bands_out:
            bands_df.to_csv(a.bands_out, index=False)


if __name__ == "__main__":
    main()
