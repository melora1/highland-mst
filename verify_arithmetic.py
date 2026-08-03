#!/usr/bin/env python3
"""
verify_arithmetic.py
Checks the CLOSED-FORM / arithmetic numbers in "The Highland Constant and the
Divergent Second Moment" against the manuscript's stated values. These are
exact (no Monte Carlo), so each line is PASS/FAIL vs the printed value.
Does NOT verify the Moliere-quadrature outputs (eps_M table, k_opt, eta) -- see
verify_scaling.py for the divergence laws those rest on.
"""
import math

m_mu = 105.6584          # MeV
def approx(a, b, tol=0.02):   # 2% relative tolerance (matches printed rounding)
    return abs(a-b) <= tol*max(abs(b), 1e-12)
def line(name, got, want, unit="", tol=0.02):
    ok = approx(got, want, tol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:38s} got={got:.4g}{unit:>6s}  paper={want:g}{unit}")
    return ok

def beta(p_MeV):             # p in MeV/c
    return p_MeV/math.hypot(p_MeV, m_mu)

def theta0(p_GeV, xX0):      # Highland/Lynch-Dahl, PDG natural-log form (z=1)
    p = p_GeV*1000.0
    b = beta(p)
    return (13.6/(b*p))*math.sqrt(xX0)*(1+0.038*math.log(xX0/b**2))  # rad

ok = True
print("== geometry ==")
xX0 = 10/8.90 + 15/1.44
ok &= line("axial x/X0", xX0, 11.54)

print("== theta_0 (mrad), axial ==")
for p, want in [(1.0,50.8),(2.0,25.3),(3.5,14.4),(6.0,8.42)]:
    ok &= line(f"theta0 @ {p} GeV/c", theta0(p,xX0)*1e3, want, " mrad")

print("== core parameter c = thetacut^2/(2 theta0^2), thetacut=200 mrad ==")
tcut = 0.200
c1 = tcut**2/(2*theta0(1.0,xX0)**2)
ok &= line("c @ 1 GeV/c", c1, 7.75)

print("== Lynch-Dahl core anchor  S2eff = 12.1 + 0.4 c ==")
base = 12.1 - 0.4*math.log(0.02)                       # F=0.98 baseline
ok &= line("baseline S2(F=0.98)", base, 13.665, " MeV")
S2_1 = 12.1 + 0.4*c1
ok &= line("S2eff @ 1 GeV/c", S2_1, 15.20, " MeV")
ok &= line("eps_M core-only @1GeV", (S2_1/base-1)*100, 11.0, " %", tol=0.05)
for p, want in [(2.0,24.6),(3.5,50.5),(6.0,125.0)]:      # c scales ~ p^2
    c = tcut**2/(2*theta0(p,xX0)**2)
    ok &= line(f"S2eff @ {p} GeV/c", 12.1+0.4*c, want, " MeV", tol=0.03)

print("== dipole tagging  delta = 0.3 B L / p  (B=1T, L=0.3m) ==")
for p, want in [(1.0,90),(2.0,45),(3.5,26),(6.0,15)]:
    ok &= line(f"delta @ {p} GeV/c", 0.3*1.0*0.30/p*1e3, want, " mrad", tol=0.03)
ok &= line("small-angle err @90mrad", (0.090**3/24)*1e5, 3.0, " e-5rad")  # ->0.03%

print("== momentum resolution ==")
sig_delta = 2*0.020/30.0*1e3                            # mrad
ok &= line("sigma_delta", sig_delta, 1.33, " mrad")
ok &= line("sigma_p/p @1GeV", sig_delta/90*100, 1.5, " %")
ok &= line("sigma_p/p @6GeV", sig_delta/15*100, 9.0, " %", tol=0.05)

print("== detectability budget (axial) ==")
sig_in  = math.sqrt(2)*0.020/30*1e3
sig_out = math.sqrt(2)*0.020/40*1e3
sig_n   = math.hypot(sig_in, sig_out)                   # mrad per projection
ok &= line("sigma_theta_noise", sig_n, 1.18, " mrad")
budget = {1.0:(0.027,-0.011),2.0:(0.109,-0.044),3.5:(0.333,-0.134),6.0:(0.975,-0.393)}
for p,(en,ep) in budget.items():
    t0 = theta0(p,xX0)*1e3
    eps_noise = (math.sqrt(1+(sig_n/t0)**2)-1)*100
    r = sig_delta/(0.3*1.0*0.30/p*1e3)
    eps_pres  = ((1+r**2)**-0.5-1)*100
    ok &= line(f"eps_noise @ {p}", eps_noise, en, " %", tol=0.06)
    ok &= line(f"eps_pres  @ {p}", eps_pres, ep, " %", tol=0.06)

print("== adaptive cut  thetacut_opt = 1.84 theta0  &  gain = eta_opt/eta_200 ==")
for p, want in [(1.0,93),(2.0,47),(3.5,27),(6.0,15)]:
    ok &= line(f"thetacut_opt @ {p}", 1.84*theta0(p,xX0)*1e3, want, " mrad", tol=0.04)
for p, e200, gain in [(1.0,0.946,27),(2.0,0.693,73),(3.5,0.481,149),(6.0,0.314,281)]:
    ok &= line(f"SNR gain @ {p} (table self-consistency)", (1.197/e200-1)*100, gain, " %", tol=0.03)

print("== fourth-moment divergence is quadratic (analytic) ==")
# <theta^4> ~ int theta^4 * theta^-3 dtheta = theta^2/2  -> exponent 2, not log
# analytic exponent check: d ln(integral)/d ln(cut) -> 2 for large cut
def m4_tail(cut, a=1e-3):     # projected 1D tail f ~ theta^-3 => (theta^2+a^2)^-1.5
    n=20000; h=cut/n; s=0.0
    for i in range(1,n+1):
        t=i*h; s += t**4 * (t**2+a**2)**-1.5   # <theta^4>: integrand theta^4 * f
    return s*h
r = math.log(m4_tail(4.0)/m4_tail(2.0))/math.log(2.0)
ok &= line("d ln<theta^4>/d ln cut (->2)", r, 2.0, "", tol=0.05)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAIL")
