"""Energy-loss-aware p(X) scattering calibration.  [Revision plan Step 1]

Imports the repository's own conventions (config, kinematics, moliere) and
adds only what Sec. II F asks for and does not supply: propagation of the
momentum profile and a varying-p combination rule.

--------------------------------------------------------------------------
1.4  THE VARYING-p COMBINATION USED HERE  (stated, not implied)
--------------------------------------------------------------------------
It is Moliere's own path accumulation, segmented.  It is NOT per-segment
Highland quadrature, and NOT Eq. (A14) evaluated at a single p.

FACT (Moliere 1948; Bethe 1953; Lynch & Dahl 1991, their Eq. 11): chi_c^2 is
the total single-scattering strength accumulated along the path,

    chi_c^2 = sum_j 0.157 Z_j(Z_j+1)(X_j/A_j) / (p_j beta_j)^2 ,

and chi_a^2 combines as a Z(Z+1)X/A-weighted logarithmic average.
Omega_0 = chi_c^2/(1.167 chi_a^2) is the mean number of scatters over the
WHOLE path, so B is solved ONCE, from Eq. (A13), on the accumulated values.

ASSUMPTION (the generalization step): in Lynch & Dahl the sum runs over
materials at a common (p beta); here it runs over slices, each with its local
(p_j beta_j) from the propagated profile.  At constant p the two are
algebraically identical, so this reduces to Eq. (A14) in the weak-loss limit
as manuscript Sec. II F requires (see test_pofx.test_constant_p_limit).
The justification is that the sum is a discretization of the path integral
that defines chi_c^2 in the first place.  The logarithmic enhancement is NOT
summed per slice -- that is the failure mode PDG warns about for Highland --
it enters once through B(Omega_0, total).

--------------------------------------------------------------------------
1.5  THE p(X)-AWARE HIGHLAND REFERENCE (same profile, same weighting)
--------------------------------------------------------------------------
ASSUMPTION (construction choice, stated so a residual is not mistaken for
physics):

    theta_0^2[p(X)] = [1 + 0.038 ln( (x/X_0)_tot / beta_eff^2 )]^2
                      * sum_j (13.6 MeV/(p_j beta_j))^2 (dx_j / X_{0,j})

Local 1/(p beta)^2; the logarithmic enhancement evaluated ONCE on the total
radiation-length path.  Reduces algebraically to Eq. (1) at constant p.
beta_eff carries the same weight as the sum; its effect on theta_0 is at the
1e-4 level (see step1_report.py section 7).

--------------------------------------------------------------------------
GEOMETRY ORDERING
--------------------------------------------------------------------------
geometry.trace_true/trace_ref return unordered (t_Al, t_Cu, t_Pb).  Energy
loss needs traversal ORDER.  ``ordered_path`` imposes the shell ordering
Al/2 -> Cu/2 -> Pb -> Cu/2 -> Al/2, which is exact for an axial ray through
the concentric Al(25)/Cu(15)/Pb geometry of config.py and accurate to the
beam divergence (SIGMA_DIV = 2 mrad) otherwise.  ASSUMPTION, flagged.
"""
import math

import numpy as np

import moliere as ml
from config import MATERIALS, MAT_ORDER, M_MU, THETA_CUT
from kinematics import theta_space_highland
from stopping import dedx_of_E

MEV = 1e3

# ---------------------------------------------------------------- CSDA range
# R_m(E) = int_{E_ref}^{E} dE'/S(E';m)  in g cm^-2.  Propagation through a
# layer of areal density X is then E_out = R_m^{-1}(R_m(E_in) - X), which is
# the exact continuous-slowing-down solution of dE/dX = -S and costs two
# interpolations instead of an RK4 integration per call.

_T_MIN = 1.0e-3          # GeV kinetic energy floor of the tables
_T_MAX = 5.0e1           # GeV
_N_GRID = 6000

def _build_range_table(material):
    T = np.geomspace(_T_MIN, _T_MAX, _N_GRID)
    E = T + M_MU
    inv_S = np.array([1.0 / (dedx_of_E(float(e), material) * 1e-3)
                      for e in E])                          # (g/cm^2) per GeV
    R = np.concatenate([[0.0],
                        np.cumsum(0.5 * (inv_S[1:] + inv_S[:-1]) * np.diff(E))])
    return E, R


_RANGE = {m: _build_range_table(m) for m in MAT_ORDER}


def energy_after(E_in, material, X):
    """Total energy [GeV] after traversing areal density X [g/cm^2]."""
    E_grid, R_grid = _RANGE[material]
    R_in = float(np.interp(E_in, E_grid, R_grid))
    R_out = R_in - float(X)
    if R_out <= R_grid[0]:
        raise RuntimeError(f"muon stopped in {material}: R_in={R_in:.2f} "
                           f"g/cm^2 < X={X:.2f} g/cm^2")
    return float(np.interp(R_out, R_grid, E_grid))


def _p_of_E(E):
    return math.sqrt(max(E * E - M_MU * M_MU, 1e-14))


def _pbeta_of_E(E):
    p = _p_of_E(E)
    return p * p / E


def _p_of_pbeta(q):
    """Invert p beta = p^2/sqrt(p^2+m^2)."""
    u = 0.5 * (q * q + q * math.sqrt(q * q + 4.0 * M_MU * M_MU))
    return math.sqrt(u)


# ---------------------------------------------------------------- path spec
def ordered_path(t_al, t_cu, t_pb):
    """Traversal-ordered [(material, thickness_cm), ...] for the shell."""
    out = []
    if t_al > 0:
        out.append(("Al", 0.5 * t_al))
    if t_cu > 0:
        out.append(("Cu", 0.5 * t_cu))
    if t_pb > 0:
        out.append(("Pb", t_pb))
    if t_cu > 0:
        out.append(("Cu", 0.5 * t_cu))
    if t_al > 0:
        out.append(("Al", 0.5 * t_al))
    return out


def x_over_X0(path):
    return sum(t / MATERIALS[n]["X0"] for n, t in path)


# ---------------------------------------------------------------- 1.1 slicing
def slice_path(path, p_in, tol=0.01, max_slices=2000):
    """Adaptive slicing with |Delta(p beta)/(p beta)| < tol per slice.

    Returns (slices, E_out).  Each slice carries the areal density X_j, the
    thickness dx_j, and the representative (p beta)_j defined so that the
    slice reproduces the exact sub-integral of 1/(p beta)^2 -- the quantity
    both chi_c^2 and the Highland core need -- to Simpson accuracy.
    """
    E = math.hypot(p_in, M_MU)
    slices = []
    for name, t_cm in path:
        if t_cm <= 0.0:
            continue
        X_layer = MATERIALS[name]["rho"] * t_cm
        E_layer_out = energy_after(E, name, X_layer)
        drop = _pbeta_of_E(E) / _pbeta_of_E(E_layer_out) - 1.0
        n = int(min(max(1, math.ceil(drop / tol)), max_slices))
        dX = X_layer / n
        for _ in range(n):
            E0 = E
            Em = energy_after(E0, name, 0.5 * dX)
            E1 = energy_after(E0, name, dX)
            q0, qm, q1 = (_pbeta_of_E(E0), _pbeta_of_E(Em), _pbeta_of_E(E1))
            # Simpson on 1/(p beta)^2 across the slice
            w = (dX / 6.0) * (1.0 / q0**2 + 4.0 / qm**2 + 1.0 / q1**2)
            q_rep = math.sqrt(dX / w)
            p_rep = _p_of_pbeta(q_rep)
            slices.append(dict(mat=name, X=dX,
                               dx=dX / MATERIALS[name]["rho"],
                               pb=q_rep, p=p_rep, beta=q_rep / p_rep))
            E = E1
    return slices, E


# ------------------------------------------------- 1.3 / 1.4  accumulation
def accumulate_moliere(slices):
    """chi_c^2, chi_a^2, B from the segmented varying-p accumulation."""
    chi_c2 = 0.0
    num = den = 0.0
    for s in slices:
        m = MATERIALS[s["mat"]]
        Z, A = m["Z"], m["A"]
        p_mev = s["p"] * MEV
        chi_c2 += ml.chi_c2_single(Z, A, s["X"], p_mev, s["beta"])
        w = Z * (Z + 1.0) * s["X"] / A
        num += w * math.log(ml.chi_a2_single(Z, p_mev, s["beta"]))
        den += w
    if den == 0.0:
        return 0.0, 1.0, 1.0
    chi_a2 = math.exp(num / den)
    return chi_c2, chi_a2, ml.solve_B(chi_c2, chi_a2)


# ---------------------------------------------------- 1.5  Highland with p(X)
def highland_pofx(slices, xx0_total):
    """(theta_0[p(X)] projected [rad], beta_eff)."""
    core2 = 0.0
    wsum = 0.0
    b2 = 0.0
    for s in slices:
        xj = s["dx"] / MATERIALS[s["mat"]]["X0"]
        core2 += (13.6 ** 2) * xj / (s["pb"] * MEV) ** 2
        w = xj / s["pb"] ** 2
        b2 += w * s["beta"] ** 2
        wsum += w
    if wsum == 0.0:
        return 0.0, 1.0
    beta_eff2 = b2 / wsum
    logf = 1.0 + 0.038 * math.log(xx0_total / beta_eff2)
    return math.sqrt(core2) * logf, math.sqrt(beta_eff2)


# ---------------------------------------------------------------- calibration
def calibrate(t_al, t_cu, t_pb, p_in, theta_cut=THETA_CUT, tol=0.01, nmax=2):
    """Full p(X) calibration for one incident momentum and one path.

    Thicknesses in cm.  Returns every quantity Step 1 asks for, including the
    constant-p baseline computed through the repository's own Eq. (A14) path
    so the comparison is implementation-matched.
    """
    path = ordered_path(t_al, t_cu, t_pb)
    xx0 = x_over_X0(path)
    if xx0 <= 0.0:
        raise ValueError("empty path")

    slices, E_out = slice_path(path, p_in, tol=tol)
    p_out = _p_of_E(E_out)

    chi_c2, chi_a2, B = accumulate_moliere(slices)
    Fc, M2, M4 = ml.radial_moments(chi_c2, B, float(theta_cut), nmax=nmax)
    th_rms = math.sqrt(max(M2, 0.0))

    th0_pX, beta_eff = highland_pofx(slices, xx0)
    th_space_pX = math.sqrt(2.0) * th0_pX

    # constant-p baseline through the repo's own combine_path / Eq. (1)
    X_al = MATERIALS["Al"]["rho"] * t_al
    X_cu = MATERIALS["Cu"]["rho"] * t_cu
    X_pb = MATERIALS["Pb"]["rho"] * t_pb
    c2_0, a2_0 = ml.combine_path(X_al, X_cu, X_pb, p_in)
    B0 = ml.solve_B(c2_0, a2_0)
    Fc0, M2_0, M4_0 = ml.radial_moments(c2_0, B0, float(theta_cut), nmax=nmax)
    th_rms_0 = math.sqrt(max(M2_0, 0.0))
    th_space_0 = float(theta_space_highland(p_in, xx0))

    return dict(
        p_in=p_in, p_out=p_out, dp_over_p=p_out / p_in - 1.0,
        dE=math.hypot(p_in, M_MU) - E_out,
        mass=X_al + X_cu + X_pb, xx0=xx0, n_slices=len(slices),
        chi_c2=chi_c2, chi_a2=chi_a2, B=B,
        Omega0=chi_c2 / (1.167 * chi_a2),
        Fc=Fc, M2=M2, M4=M4, th_rms=th_rms,
        th0=th0_pX, th_space=th_space_pX, beta_eff=beta_eff,
        k=float(theta_cut) / th0_pX,
        eps_M=th_rms / th_space_pX - 1.0,
        # deployed estimator: numerator on p(X), denominator at the tagged p_in
        eps_mix=th_rms / th_space_0 - 1.0,
        # constant-p baseline (what Table I currently reports)
        chi_c2_0=c2_0, B_0=B0, Fc_0=Fc0, M4_0=M4_0,
        th_rms_0=th_rms_0, th_space_0=th_space_0,
        k_0=float(theta_cut) / (th_space_0 / math.sqrt(2.0)),
        eps_M_0=th_rms_0 / th_space_0 - 1.0,
        slices=slices)


def efficiency_pofx(t_al, t_cu, t_pb, p_in, k, tol=0.01):
    """Eq. (25) with the cut set by the p(X)-aware theta_0."""
    path = ordered_path(t_al, t_cu, t_pb)
    slices, _ = slice_path(path, p_in, tol=tol)
    th0, _ = highland_pofx(slices, x_over_X0(path))
    c2, a2, B = accumulate_moliere(slices)
    Fc, M2, M4 = ml.radial_moments(c2, B, float(k) * th0, nmax=2)
    var = max(M4 - M2 * M2, 0.0)
    if Fc <= 0.0 or var <= 0.0:
        return 0.0
    return math.sqrt(Fc) * M2 / math.sqrt(var)


def optimal_k_pofx(t_al, t_cu, t_pb, p_in, bounds=(0.5, 8.0)):
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(
        lambda k: -efficiency_pofx(t_al, t_cu, t_pb, p_in, k),
        bounds=bounds, method="bounded", options={"xatol": 2e-4})
    if not r.success:
        raise RuntimeError(r.message)
    k = float(r.x)
    return k, efficiency_pofx(t_al, t_cu, t_pb, p_in, k)
