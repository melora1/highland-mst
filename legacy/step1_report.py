#!/usr/bin/env python3
"""Step 1 deliverable: energy-loss treatment for the reference paths.

    python3 step1_report.py

Sections
  0  closure   -- stopping power vs PDG minima; constant-p reduction;
                  repo-baseline match
  1  profile   -- p(X), Delta p/p, slice counts, Omega_0                [1.1-1.2]
  2  converge  -- slicing tolerance scan
  3  Table I   -- regenerated, with a Delta p / p column                [1.6]
  4  deployed  -- eps_mix and E[w], what Eq. (18) actually produces
  5  sensitiv. -- S(E) scale factor scan, bounding the omitted radiative term
  6  off-Cu    -- 25 cm Al, input to Step 3.4
  6b log-factor-per-slice deficit, justifying the 1.5 construction  [1.4]
  7  beta_eff convention sensitivity                                    [1.5]
"""

import math

import numpy as np

from . import energy_loss as el
from . import stopping
from config import MATERIALS, MOMENTA, THETA_CUT
from .eps_quadrature import eps_M as eps_M_constp, optimal_k as optimal_k_constp
from .kinematics import theta0_highland

AXIAL = (10.0, 15.0, 0.0)  # t_Al, t_Cu, t_Pb  cm
OFFCU = (25.0, 0.0, 0.0)
PUBLISHED = {1.0: 4.068, 2.0: 9.771, 3.5: 13.146, 6.0: 16.150}
RULE = "-" * 78


def _X(t):
    return (
        MATERIALS["Al"]["rho"] * t[0],
        MATERIALS["Cu"]["rho"] * t[1],
        MATERIALS["Pb"]["rho"] * t[2],
    )


# ------------------------------------------------------------------ 0 closure
print(RULE)
print("0.  CLOSURE")
print(RULE)

print("\n0a. stopping power vs PDG minima (indirect Sternheimer check):")
print(f"    {'mat':>4} {'S_min':>8} {'PDG':>8} {'rel':>9} {'(bg)_min':>9}")
for n, (got, want, rel, bg) in stopping.validate().items():
    print(f"    {n:>4} {got:8.4f} {want:8.4f} {rel * 100:+8.3f}% {bg:9.3f}")

print("\n0b. constant-p reduction (stopping power switched off):")
_real = el.energy_after
el.energy_after = lambda E, m, X: E
for p in (1.0, 6.0):
    r = el.calibrate(*AXIAL, p)
    print(
        f"    p={p:4.1f}  eps_M[p(X), no loss]={100 * r['eps_M']:.9f}%  "
        f"eps_M[Eq.(A14)]={100 * r['eps_M_0']:.9f}%  "
        f"diff={100 * (r['eps_M'] - r['eps_M_0']):+.2e} pp"
    )
el.energy_after = _real

print("\n0c. constant-p baseline vs repo eps_quadrature, and vs published Table I:")
print(
    f"    {'p':>5} {'calibrate':>11} {'eps_quad':>11} {'Table I':>9} "
    f"{'code-paper (pp)':>16}"
)
Xax = _X(AXIAL)
for p in MOMENTA:
    mine = 100 * el.calibrate(*AXIAL, p)["eps_M_0"]
    repo = 100 * float(eps_M_constp(p, *Xax)[0])
    print(
        f"    {p:5.1f} {mine:11.4f} {repo:11.4f} {PUBLISHED[p]:9.3f} "
        f"{repo - PUBLISHED[p]:+16.4f}"
    )
print("    If column 4 is non-zero, Table I in the manuscript was produced by")
print("    a different code state than the current repo and must be reissued")
print("    before any p(X) shift is quoted against it.")

# ------------------------------------------------------------------ 1 profile
print()
print(RULE)
print("1.  MOMENTUM PROFILE AND ADAPTIVE SLICING  [1.1, 1.2]")
print(RULE)
print(
    f"axial path: Al {AXIAL[0]} cm + Cu {AXIAL[1]} cm, "
    f"mass {sum(Xax):.1f} g/cm^2, x/X0 = "
    f"{el.x_over_X0(el.ordered_path(*AXIAL)):.4f}"
)
print(
    f"\n{'p_in':>5} {'p_out':>8} {'Dp/p':>9} {'DE (MeV)':>10} "
    f"{'<S>':>8} {'N_slice':>8} {'Omega_0':>11}"
)
rows = []
for p in MOMENTA:
    r = el.calibrate(*AXIAL, p)
    rows.append(r)
    print(
        f"{p:5.1f} {r['p_out']:8.4f} {100 * r['dp_over_p']:+8.2f}% "
        f"{r['dE'] * 1e3:10.1f} {r['dE'] * 1e3 / r['mass']:8.4f} "
        f"{r['n_slices']:8d} {r['Omega0']:11.3e}"
    )
print("  <S> in MeV cm^2/g.  Omega_0 >> 20 everywhere: Moliere stays valid")
print("  on the degraded profile (App. A validity condition).")

# ------------------------------------------------------------------ 2 converge
print()
print(RULE)
print("2.  SLICING CONVERGENCE")
print(RULE)
print(f"{'tol':>7} {'N_slice':>8} {'eps_M[p(X)] %':>16} {'shift vs 1%':>14}")
ref = el.calibrate(*AXIAL, 1.0, tol=0.01)["eps_M"]
for tol in (0.02, 0.01, 0.005, 0.002, 0.001):
    r = el.calibrate(*AXIAL, 1.0, tol=tol)
    print(
        f"{tol:7.3f} {r['n_slices']:8d} {100 * r['eps_M']:16.6f} "
        f"{100 * (r['eps_M'] - ref):+14.2e}"
    )

# ------------------------------------------------------------------ 3 Table I
print()
print(RULE)
print("3.  REGENERATED TABLE I  [1.6]   axial reference path, 200 mrad")
print(RULE)
print(
    f"{'p_in':>5} {'Dp/p %':>8} {'k const-p':>10} {'k p(X)':>8} "
    f"{'F_c':>9} {'eps const-p %':>14} {'eps p(X) %':>12} {'shift pp':>10}"
)
for p, r in zip(MOMENTA, rows):
    print(
        f"{p:5.1f} {100 * r['dp_over_p']:+8.2f} {r['k_0']:10.3f} "
        f"{r['k']:8.3f} {r['Fc']:9.5f} {100 * r['eps_M_0']:14.3f} "
        f"{100 * r['eps_M']:12.3f} {100 * (r['eps_M'] - r['eps_M_0']):+10.3f}"
    )

print(f"\n{'p_in':>5} {'k_opt const-p':>14} {'k_opt p(X)':>11} {'eta_max':>9}")
for p in MOMENTA:
    k0 = float(optimal_k_constp(p, *Xax)[0])
    kx, ex = el.optimal_k_pofx(*AXIAL, p)
    print(f"{p:5.1f} {k0:14.4f} {kx:11.4f} {ex:9.4f}")

# ------------------------------------------------------------------ 4 deployed
print()
print(RULE)
print("4.  WHAT THE DEPLOYED ESTIMATOR SEES")
print(RULE)
print("Eq. (18) divides by theta_space(p_meas); p_meas is the UPSTREAM tagged")
print("momentum while the numerator is scattering on the degraded profile.")
print("So E[w] = (theta_rms[p(X)] / theta_space(p_in))^2.")
print(
    f"\n{'p_in':>5} {'eps const-p %':>14} {'eps p(X) %':>12} "
    f"{'eps mixed %':>12} {'E[w] mixed':>11} {'E[w] Table I':>13}"
)
for p, r in zip(MOMENTA, rows):
    print(
        f"{p:5.1f} {100 * r['eps_M_0']:14.3f} {100 * r['eps_M']:12.3f} "
        f"{100 * r['eps_mix']:12.3f} {(1 + r['eps_mix']) ** 2:11.4f} "
        f"{(1 + r['eps_M_0']) ** 2:13.4f}"
    )

# ------------------------------------------------------------------ 5 sens.
print()
print(RULE)
print("5.  S(E) SCALE SENSITIVITY (bounds the omitted radiative term)")
print(RULE)
print("Scale S(E) by f and rebuild the range tables.  Radiative loss is not")
print("in S; this converts any omission into an explicit eps_M error.")
print(f"\n{'f':>6} {'p_out(1 GeV)':>13} {'eps p(X) % @1':>15} {'eps p(X) % @6':>15}")
_orig_dedx = el.dedx_of_E
for f in (0.95, 1.00, 1.05, 1.10):
    el.dedx_of_E = (lambda ff: lambda E, m: ff * _orig_dedx(E, m))(f)
    el._RANGE = {m: el._build_range_table(m) for m in ("Al", "Cu", "Pb")}
    a = el.calibrate(*AXIAL, 1.0)
    b = el.calibrate(*AXIAL, 6.0)
    print(
        f"{f:6.2f} {a['p_out']:13.4f} {100 * a['eps_M']:15.4f} {100 * b['eps_M']:15.4f}"
    )
el.dedx_of_E = _orig_dedx
el._RANGE = {m: el._build_range_table(m) for m in ("Al", "Cu", "Pb")}

# ------------------------------------------------------------------ 6 off-Cu
print()
print(RULE)
print("6.  OFF-Cu PATH (25 cm Al)  -- input to Step 3.4")
print(RULE)
print(
    f"x/X0 = {el.x_over_X0(el.ordered_path(*OFFCU)):.4f}, "
    f"mass {sum(_X(OFFCU)):.1f} g/cm^2"
)
print(
    f"{'p_in':>5} {'Dp/p %':>8} {'k const-p':>10} {'k p(X)':>8} "
    f"{'eps const-p %':>14} {'eps p(X) %':>12} {'eps mixed %':>12}"
)
for p in MOMENTA:
    r = el.calibrate(*OFFCU, p)
    print(
        f"{p:5.1f} {100 * r['dp_over_p']:+8.2f} {r['k_0']:10.3f} "
        f"{r['k']:8.3f} {100 * r['eps_M_0']:14.3f} {100 * r['eps_M']:12.3f} "
        f"{100 * r['eps_mix']:12.3f}"
    )

print()
print(RULE)
print("6b. PER-SLICE HIGHLAND QUADRATURE DEFICIT  [justifies 1.5]")
print(RULE)
print(f"{'p':>5} {'theta0 single-log':>19} {'theta0 per-slice':>18} {'ratio':>8}")
for p in MOMENTA:
    path = el.ordered_path(*AXIAL)
    sl, _ = el.slice_path(path, p)
    th_ok, _ = el.highland_pofx(sl, el.x_over_X0(path))
    q = sum(
        float(theta0_highland(s["p"], s["dx"] / MATERIALS[s["mat"]]["X0"])) ** 2
        for s in sl
    )
    print(
        f"{p:5.1f} {th_ok * 1e3:19.4f} {math.sqrt(q) * 1e3:18.4f} "
        f"{math.sqrt(q) / th_ok:8.4f}"
    )

# ------------------------------------------------------------------ 7 beta_eff
print()
print(RULE)
print("7.  beta_eff CONVENTION SENSITIVITY IN THE LOG FACTOR  [1.5]")
print(RULE)
xx0 = el.x_over_X0(el.ordered_path(*AXIAL))
logf = lambda b: 1.0 + 0.038 * math.log(xx0 / b**2)
for p in MOMENTA:
    r = el.calibrate(*AXIAL, p)
    b_in = p / math.hypot(p, el.M_MU)
    b_out = r["p_out"] / math.hypot(r["p_out"], el.M_MU)
    print(
        f"p={p:4.1f}  beta_eff={r['beta_eff']:.6f}  "
        f"log factor entry={logf(b_in):.6f} exit={logf(b_out):.6f} "
        f"used={logf(r['beta_eff']):.6f}  spread on theta_0 = "
        f"{100 * (logf(b_out) / logf(b_in) - 1):+.4f}%"
    )
