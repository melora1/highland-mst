#!/usr/bin/env python3
"""
geant4_compare.py  --  cross-check the Moliere quadrature (moliere.py/
quadrature.py) against an independent Geant4 transport simulation, as in
manuscript Sec. 3.

Geant4 itself is a C++/macro toolkit and is NOT invoked from Python; you run it
separately (see the macro template + physics notes at the bottom) and dump, per
configuration, the per-event SPACE-angle scatter theta_space = sqrt(thx^2+thy^2)
in radians to a text/CSV file. This script does the ANALYSIS side:

  input file  : one float per line = theta_space (rad) for each surviving muon,
                OR a CSV with a 'theta_space' column.
  it computes : in-acceptance RMS within |theta| < theta_cut (200 mrad),
  compares to : quadrature.theta_rms for the SAME (material, p, x/X0, cut),
  and reports : Geant4 vs quadrature RMS, the implied eps_M, and whether the
                difference sits inside the published Urban/Wentzel-VI model
                spread (Makarova 2017: ~4% in Pb) plus the ~1% planar systematic.

Usage:
  python3 geant4_compare.py --file cu_p1.0_urban.txt \
        --material Cu --thickness_cm 15 --p 1.0 --model urban
"""
import argparse, math, sys
import numpy as np
from moliere import moliere_params, theta0_highland
from quadrature import theta_rms

# published model spread (Makarova 2017, thickness-averaged), fractional
MODEL_SPREAD = {"urban": 0.08, "wentzel": 0.04}   # low-Z worst case; Pb ~4%
PLANAR_SYS = 0.011                                 # ~1% independence approximation

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
    a = ap.parse_args(argv)

    angles = load_angles(a.file)
    g4_rms, n_keep, n_tot = in_acceptance_rms(angles, a.theta_cut)

    path = [(a.material, a.thickness_cm)]
    mp = moliere_params(path, a.p)
    q_rms = theta_rms(a.theta_cut, mp)
    t0 = theta0_highland(a.p, mp["xX0"])
    tspace = math.sqrt(2.0)*t0

    frac_diff = (g4_rms - q_rms)/q_rms
    tol = MODEL_SPREAD[a.model] + PLANAR_SYS
    verdict = "PASS" if abs(frac_diff) <= tol else "OUTSIDE BUDGET"

    print(f"material={a.material}  p={a.p} GeV/c  x/X0={mp['xX0']:.3f}  "
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
#                incidence; energy loss ENABLED but incident p used in compare
#                (a no-eloss control run isolates pure scattering at low p).
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
