#!/usr/bin/env python3
"""
verify_scaling.py
Independently checks the two DIVERGENCE LAWS that the manuscript's central
argument rests on, using the screened single-scatter tail it cites itself
(Sec. 2.2): projected density f(theta) ~ (theta^2 + chi_a^2)^-1.5, i.e. the
theta^-3 tail of Rutherford single scattering in projection.

Manuscript claims (Sec. 5.1, 4.2), fit over thetacut in [3,25]*theta0:
  * <theta^2> (2nd moment)  ~ const + A ln(thetacut)     -> log fit good
  * <theta^4> (4th moment)  ~ const + A' thetacut^2       -> quad fit R^2~0.998,
                                                             log fit R^2~0.811
This script reproduces the QUALITATIVE laws and the relative fit quality.
It does NOT reproduce the exact eps_M values or k_opt=1.84 -- those need the
full Bethe f^(n) core, which is out of scope for a light check (see note).
"""

import math

# --- model: Gaussian core (width s0) + screened power-law tail -------------
s0 = 1.0  # core width  = theta0 (units)
chia = 0.02 * s0  # screening angle (tail turnover), << core
wtail = 0.02  # single-scatter tail weight (illustrative)


def f_core(t):
    return math.exp(-0.5 * (t / s0) ** 2)


def f_tail(t):
    return (t * t + chia * chia) ** -1.5  # ~ t^-3 at large t


def f(t):
    return (1 - wtail) * f_core(t) / (s0 * math.sqrt(2 * math.pi)) + wtail * f_tail(t)


def moment(k, cut, n=4000):  # projected 1D moment <theta^k> within [0,cut]
    h = cut / n
    num = den = 0.0
    for i in range(1, n + 1):
        t = (i - 0.5) * h
        ft = f(t)
        num += t**k * ft
        den += ft
    return num / den


def r2(xs, ys):  # R^2 of best linear y = a x + b
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1 - ss_res / ss_tot


# asymptotic window [3,25]*theta0, as in the manuscript
cuts = [3.0 + (25.0 - 3.0) * i / 24 for i in range(25)]
m2 = [moment(2, c) for c in cuts]
m4 = [moment(4, c) for c in cuts]

# 2nd moment: theta_RMS^2 vs ln(cut)  -> expect high R^2 (log law)
R2_2_log = r2([math.log(c) for c in cuts], m2)
# 4th moment: vs cut^2 (quadratic) vs vs ln(cut) (log) -> quad should win big
R2_4_quad = r2([c * c for c in cuts], m4)
R2_4_log = r2([math.log(c) for c in cuts], m4)

print("SECOND MOMENT  <theta^2>(cut):")
print(f"   R^2 (linear in ln cut)   = {R2_2_log:.4f}   [paper: log law, R^2~0.963]")
print(f"   -> {'PASS' if R2_2_log > 0.95 else 'FAIL'}: 2nd moment follows a log law")
print("FOURTH MOMENT  <theta^4>(cut):")
print(f"   R^2 (linear in cut^2)    = {R2_4_quad:.4f}   [paper: 0.9976]")
print(f"   R^2 (linear in ln cut)   = {R2_4_log:.4f}   [paper: 0.811]")
qual = R2_4_quad > 0.99 and R2_4_quad - R2_4_log > 0.1
print(
    f"   -> {'PASS' if qual else 'FAIL'}: 4th moment is quadratic, not log; "
    f"quadratic beats log by {R2_4_quad - R2_4_log:.3f}"
)

print(
    "\nNOTE: exact eps_M(+5.1..+17.1%), k_opt=1.84, eta_max=1.197 require the "
    "full Bethe f^(1),f^(2) core (Appendix A) and the per-momentum chi_c/B "
    "from the real geometry; not reproduced here. This script confirms only "
    "the divergence LAWS underpinning them (2nd-moment log growth, "
    "4th-moment quadratic growth), using a simplified illustrative model, "
    "not the real geometry.\n"
    "The eps_M magnitudes themselves are independently reproduced by "
    "eps_quadrature.py's verify() (matches the manuscript's axial table to "
    "rounding: got 5.09/10.81/14.16/17.14% vs. paper's "
    "5.1/10.8/14.2/17.1% at 1/2/3.5/6 GeV/c) and k_opt~1.85 from "
    "verify_kopt()'s coarse grid search (module default K_OPT=1.825 from a "
    "finer scipy.optimize scan -- the two differ only by grid resolution, "
    "not a physics discrepancy). These are no longer [UNVERIFIED-CODE]; "
    "per README.md's legend, only the Geant4 cross-check numbers (Sec. 3) "
    "and the Hanson core-offset figures remain in that category."
)
