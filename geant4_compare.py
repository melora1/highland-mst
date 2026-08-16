#!/usr/bin/env python3
"""Step-8 Geant4 comparison without hard-coded model-error budgets.

Each input file supplies one space-angle magnitude per line or a CSV column
`theta_space`.  Urban/Wentzel disagreement is reported from the supplied runs;
it is not replaced by a literature-derived pass/fail tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from analysis import PATHS, AXIAL_ORDERED, OFFCU_ORDERED
from physics import calibrate_pofx, constant_calibration, theta0_highland

PATH_ORDERED = {"AlCu": AXIAL_ORDERED, "Al25": OFFCU_ORDERED}


def load_angles(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        d = pd.read_csv(p)
        if "theta_space" not in d.columns:
            raise ValueError(f"{p}: CSV needs theta_space column")
        return d.theta_space.to_numpy(float)
    a = np.loadtxt(p)
    return a[:, 0].astype(float) if a.ndim > 1 else a.astype(float)


def sample_moments(a, cut):
    keep = a[a < cut]
    if keep.size == 0:
        return dict(Fc=0.0, theta_rms=np.nan, M4=np.nan, n=0)
    return dict(
        Fc=keep.size / a.size,
        theta_rms=float(np.sqrt(np.mean(keep**2))),
        M4=float(np.mean(keep**4)),
        n=int(keep.size),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        action="append",
        required=True,
        help="label=angles.txt (repeat for Urban/Wentzel)",
    )
    ap.add_argument("--path", choices=["AlCu", "Al25"], default="AlCu")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument(
        "--energy-loss",
        action="store_true",
        help="compare to the segmented p(X) approximation; otherwise constant-p control",
    )
    ap.add_argument("--k", type=float, nargs="+", default=[2, 5, 10, 20, 40])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in a.file}
    data = {lab: load_angles(f) for lab, f in files.items()}

    rows = []
    if a.energy_loss:
        if a.path not in PATH_ORDERED:
            raise ValueError("energy-loss benchmark requires an ordered path")
        base = calibrate_pofx(PATH_ORDERED[a.path], a.p, theta_cut=0.2)
        theta0 = base["theta0_pofx"]
    else:
        # k is always theta_cut/theta0, not eta_cut.
        rp = constant_calibration(PATHS[a.path], a.p, theta_cut=0.2)
        theta0 = rp["theta_space"] / np.sqrt(2.0)

    for k in a.k:
        cut = k * theta0
        if a.energy_loss:
            q = calibrate_pofx(PATH_ORDERED[a.path], a.p, theta_cut=cut)
        else:
            q = constant_calibration(PATHS[a.path], a.p, theta_cut=cut)
        for lab, ang in data.items():
            g = sample_moments(ang, cut)
            rows.append(
                dict(
                    model=lab,
                    k=k,
                    theta_cut=cut,
                    Fc_g4=g["Fc"],
                    Fc_quad=q["Fc"],
                    theta_rms_g4=g["theta_rms"],
                    theta_rms_quad=q["theta_rms"],
                    M4_g4=g["M4"],
                    M4_quad=q["M4"],
                    rms_frac=(g["theta_rms"] / q["theta_rms"] - 1.0),
                    Fc_diff=g["Fc"] - q["Fc"],
                    M4_frac=(g["M4"] / q["M4"] - 1.0),
                )
            )
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    if len(files) >= 2:
        pivot = d.pivot(index="k", columns="model", values="theta_rms_g4")
        cols = list(pivot.columns)
        if len(cols) >= 2:
            spread = (pivot[cols[0]] / pivot[cols[1]] - 1.0).rename("model_spread_rms")
            print(
                "\nUrban/Wentzel (or supplied-model) RMS spread:\n", spread.to_string()
            )
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
