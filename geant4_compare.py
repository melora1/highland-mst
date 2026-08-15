#!/usr/bin/env python3
"""Compare a constant-momentum Geant4 control sample with radial quadrature.

The corrected Python reference is the manuscript's non-factorized radial
Moliere n<=2 calculation.  Direct comparison is valid here only for a Geant4
control in which momentum is held fixed through the slab (energy loss disabled).
An energy-loss-enabled transport run requires the separate p(X)-aware treatment
described in the manuscript and is intentionally rejected by this script.

Input: one space-angle magnitude per line, or CSV column ``theta_space``.
Output: accepted RMS, radial-quadrature RMS, Highland core RMS, and discrepancy.
"""
import argparse, math, sys
import numpy as np
from config import MATERIALS
from kinematics import theta_space_highland
from eps_quadrature import theta_RMS_at_cut

# published model spread (Makarova 2017, thickness-averaged), fractional
MODEL_SPREAD = {"urban": 0.08, "wentzel": 0.04}   # low-Z worst case; Pb ~4%

def load_angles(path):
    try:
        arr = np.loadtxt(path)
        if arr.ndim > 1:                            # took first column
            arr = arr[:, 0]
        return arr.astype(float)
    except Exception:
        # CSV with header 'theta_space'
        import csv
        vals = []
        with open(path) as f:
            for row in csv.DictReader(f):
                vals.append(float(row["theta_space"]))
        return np.array(vals)

def in_acceptance_rms(angles, theta_cut):
    keep = angles[angles < theta_cut]
    if keep.size == 0:
        raise SystemExit("no events inside acceptance")
    return math.sqrt(float(np.mean(keep**2))), keep.size, angles.size

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Geant4 theta_space dump (rad)")
    ap.add_argument("--material", required=True, choices=["Al", "Cu", "Pb"])
    ap.add_argument("--thickness_cm", type=float, required=True)
    ap.add_argument("--p", type=float, required=True, help="momentum GeV/c")
    ap.add_argument("--model", default="urban", choices=["urban", "wentzel"])
    ap.add_argument("--theta_cut", type=float, default=0.200)
    ap.add_argument(
        "--constant-momentum-control", action="store_true",
        help=("Assert that the supplied Geant4 sample was generated with "
              "energy loss disabled (or otherwise fixed momentum). The "
              "Python radial quadrature is the manuscript's constant-p "
              "limit and must not be compared directly to an energy-loss "
              "sample as though it implemented p(X)."))
    a = ap.parse_args(argv)

    if not a.constant_momentum_control:
        raise SystemExit(
            "Direct Geant4/quadrature comparison requires "
            "--constant-momentum-control. The current Python calculation is "
            "the manuscript's constant-p limit; an energy-loss-enabled "
            "Geant4 run requires a p(X)-aware transport calibration, not the "
            "incident momentum substituted throughout the slab.")

    angles = load_angles(a.file)
    g4_rms, n_keep, n_tot = in_acceptance_rms(angles, a.theta_cut)

    # single-material slab: areal density X (g/cm^2) = rho * thickness_cm;
    # the other two materials' X are zero (eps_quadrature.theta_RMS_at_cut
    # takes X_al, X_cu, X_pb explicitly -- matches the real repo's moliere.py
    # combine_path signature, which needs all three even for one material).
    rho = MATERIALS[a.material]["rho"]
    X = rho * a.thickness_cm
    X_al = X if a.material == "Al" else 0.0
    X_cu = X if a.material == "Cu" else 0.0
    X_pb = X if a.material == "Pb" else 0.0
    xX0 = a.thickness_cm / MATERIALS[a.material]["X0"]

    q_rms = float(theta_RMS_at_cut(a.p, X_al, X_cu, X_pb, a.theta_cut)[0])
    tspace = float(theta_space_highland(a.p, xX0))

    frac_diff = (g4_rms - q_rms)/q_rms
    tol = MODEL_SPREAD[a.model]
    verdict = "PASS" if abs(frac_diff) <= tol else "OUTSIDE BUDGET"

    print(f"material={a.material}  p={a.p} GeV/c  x/X0={xX0:.3f}  "
          f"cut={a.theta_cut*1e3:.0f} mrad  model={a.model}")
    print(f"  events: {n_keep}/{n_tot} inside acceptance "
          f"({100*n_keep/n_tot:.2f}%)")
    print(f"  Geant4  RMS   = {g4_rms*1e3:8.3f} mrad")
    print(f"  quad    RMS   = {q_rms*1e3:8.3f} mrad")
    print(f"  theta_space   = {tspace*1e3:8.3f} mrad  (sqrt2 * Highland theta0)")
    print(f"  eps_M (Geant4)= {(g4_rms-tspace)/tspace*100:+6.2f} %")
    print(f"  eps_M (quad)  = {(q_rms  -tspace)/tspace*100:+6.2f} %")
    print(f"  Geant4/quad frac diff = {frac_diff*100:+.2f} %   "
          f"budget +/-{tol*100:.1f}%  -> {verdict}")

if __name__ == "__main__":
    main()

# ==========================================================================
# GEANT4 SIDE (run separately; produces the theta_space dump this reads).
#
# Physics list:  FTFP_BERT + G4EmStandardPhysics_option4, OR select MSC model
#                explicitly per run:  Urban  vs  WentzelVI  (manuscript Sec. 3).
# Geometry:      single slab of {Cu,Pb} with thickness matching --thickness_cm;
#                monochromatic mu- beam at {1.0,2.0,3.5,6.0} GeV/c, normal
#                incidence. For THIS script's direct quadrature comparison,
#                energy loss must be DISABLED and --constant-momentum-control
#                supplied. Energy-loss-enabled runs require a p(X)-aware model.
# Scoring:       at the downstream face record the exit direction; compute
#                theta_space = angle to the incident axis; write one value/line.
#
# Example macro (models selected via the EmParameters UI):
#   /run/verbose 0
#   /control/verbose 0
#   # ---- choose ONE MSC model for the run ----
#   /process/msc/StepLimit UseSafety
#   /process/em/setMscModel <Urban|WentzelVI>        # user hook in your app
#   /gun/particle mu-
#   /gun/energy  <E_kin_for_p>                        # E_kin = sqrt(p^2+m^2)-m
#   /run/beamOn 500000
#
# Output convention expected here: plain text, one theta_space (rad) per line,
# already restricted to primary muons that exit the downstream face.
# ==========================================================================