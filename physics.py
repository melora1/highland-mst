"""Physics kernel for the revised Highland/Moliere analysis.

This module deliberately separates three levels of statement:

1. Constant-momentum radial Moliere theory: this is the analytic model used for
   the exact dimensionless reduction in the paper.
2. A segmented p(X) extension: energy is propagated by continuous slowing down;
   local scattering strengths are accumulated. Two explicit continuations of
   the screening-log average are retained (dchi_c^2 weighting and the common-p
   serial material weight). They coincide at constant momentum; their finite-
   loss spread is reported as model uncertainty rather than hidden. Neither is
   promoted to a new theorem.
3. Detector weights: these may divide the p(X)-generated accepted RMS by an
   upstream momentum-tagged Highland denominator.  That mixed quantity is kept
   distinct from the p(X)-matched epsilon_M.

The radial n<=2 Moliere series is an asymptotic truncation.  Small negative
regions are clipped only for probability sampling/moments and the clipped mass
is reported so this regularisation is never silent.  Above the numerical radial
table, the continuation used here is the point-nucleus Rutherford/Moliere
asymptote.  It deliberately does NOT include a finite nuclear form factor;
absolute large-angle predictions therefore require transport/data validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
import warnings
from math import factorial
from typing import Iterable, Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
from scipy.integrate import IntegrationWarning, quad, cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import j0

from config import (
    ALPHA,
    CUT_CACHE_STEP,
    M_E,
    M_MU,
    MATERIALS,
    MAT_ORDER,
    P_BETA_SLICE_TOL,
    P_CACHE_STEP,
    PDG_MIN_DEDX,
    RADIAL_ETA_MAX,
    SEG_CACHE_STEP,
    THETA_CUT,
)

MEV = 1.0e3
K_BETHE = 0.307075  # MeV mol^-1 cm^2
_LN10 = math.log(10.0)


# ---------------------------------------------------------------------------
# Kinematics / Highland
# ---------------------------------------------------------------------------
def beta_of(p_gev):
    p = np.asarray(p_gev, dtype=float)
    return p / np.sqrt(p * p + M_MU * M_MU)


def theta0_highland(p_gev, x_over_x0):
    """Projected Highland/Lynch-Dahl core width [rad]."""
    p = np.asarray(p_gev, dtype=float)
    x = np.asarray(x_over_x0, dtype=float)
    b = beta_of(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (13.6 / (b * p * MEV)) * np.sqrt(x) * (1.0 + 0.038 * np.log(x / (b * b)))
    return np.where(x > 0.0, out, 0.0)


def theta_space_highland(p_gev, x_over_x0):
    return np.sqrt(2.0) * theta0_highland(p_gev, x_over_x0)


# ---------------------------------------------------------------------------
# Bethe stopping power / CSDA energy propagation
# ---------------------------------------------------------------------------
def _density_effect(bg: float, material: str) -> float:
    m = MATERIALS[material]
    x = math.log10(bg)
    if x >= m.x1:
        return 2.0 * _LN10 * x - m.Cbar
    if x >= m.x0:
        return 2.0 * _LN10 * x - m.Cbar + m.a * (m.x1 - x) ** m.k
    return m.delta0 * 10.0 ** (2.0 * (x - m.x0))


def dedx_mass(p_gev: float, material: str) -> float:
    """Mean collision stopping power, MeV cm^2 g^-1.

    Radiative loss is intentionally not included.  The material constants are
    transcribed from the supplied code and must be checked against the primary
    PDG/LBL table before publication.
    """
    m = MATERIALS[material]
    gamma = math.sqrt(1.0 + (p_gev / M_MU) ** 2)
    beta = p_gev / math.hypot(p_gev, M_MU)
    bg = beta * gamma
    r = M_E / M_MU
    tmax = 2.0 * M_E * bg * bg / (1.0 + 2.0 * gamma * r + r * r)  # GeV
    arg = (2.0 * M_E * 1e9 * bg * bg) * (tmax * 1e9) / (m.I_eV**2)
    return (
        K_BETHE
        * m.Z
        / m.A
        / (beta * beta)
        * (0.5 * math.log(arg) - beta * beta - 0.5 * _density_effect(bg, material))
    )


def dedx_of_E(E_gev: float, material: str) -> float:
    p = math.sqrt(max(E_gev * E_gev - M_MU * M_MU, 1e-15))
    return dedx_mass(p, material)


def validate_stopping_minima():
    out = {}
    for name, want in PDG_MIN_DEDX.items():
        ps = np.geomspace(0.05, 2.0, 12000)
        vals = np.array([dedx_mass(float(p), name) for p in ps])
        j = int(np.argmin(vals))
        out[name] = dict(
            value=float(vals[j]),
            reference=want,
            rel=float(vals[j] / want - 1.0),
            bg=float(ps[j] / M_MU),
        )
    return out


_T_MIN = 1.0e-3
_T_MAX = 50.0
_N_RANGE = 5000


def _build_range_table(material: str):
    T = np.geomspace(_T_MIN, _T_MAX, _N_RANGE)
    E = T + M_MU
    invS = np.array([1.0 / (dedx_of_E(float(e), material) * 1e-3) for e in E])
    R = np.concatenate([[0.0], np.cumsum(0.5 * (invS[1:] + invS[:-1]) * np.diff(E))])
    return E, R


_RANGE = {m: _build_range_table(m) for m in MAT_ORDER}


def energy_after(E_in: float, material: str, X_gcm2: float) -> float:
    E_grid, R_grid = _RANGE[material]
    R_in = float(np.interp(E_in, E_grid, R_grid))
    R_out = R_in - float(X_gcm2)
    if R_out <= 0.0:
        raise RuntimeError(
            f"muon stops in {material}: range={R_in:.3f} < X={X_gcm2:.3f} g/cm^2"
        )
    return float(np.interp(R_out, R_grid, E_grid))


def _p_of_E(E):
    return math.sqrt(max(E * E - M_MU * M_MU, 1e-15))


def _pbeta_of_E(E):
    p = _p_of_E(E)
    return p * p / E


def _p_of_pbeta(q):
    u = 0.5 * (q * q + q * math.sqrt(q * q + 4.0 * M_MU * M_MU))
    return math.sqrt(u)


# ---------------------------------------------------------------------------
# Radial Moliere n<=2 functions
# ---------------------------------------------------------------------------
_RADIAL_TABLE_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 10.0, 1001),
            np.geomspace(10.0, RADIAL_ETA_MAX, 500),
        ]
    )
)
_RADIAL_INT_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 10.0, 5001),
            np.geomspace(10.0, RADIAL_ETA_MAX, 1500),
        ]
    )
)
_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_moliere_radial_cache.npz"
)


def _phi_quad(n: int, eta: float) -> float:
    def integrand(u):
        if u <= 0.0:
            return 0.0
        a = 0.25 * u * u
        return u * j0(eta * u) * math.exp(-a) * (a * math.log(a)) ** n / factorial(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(integrand, 0.0, 30.0, limit=1000, epsabs=1e-12, epsrel=1e-10)
    return float(val)


def _load_radial_tables():
    try:
        z = np.load(_CACHE_FILE)
        if z["eta"].shape == _RADIAL_TABLE_GRID.shape and np.allclose(
            z["eta"], _RADIAL_TABLE_GRID
        ):
            return z["phi1"], z["phi2"]
    except Exception:
        pass
    p1 = np.array([_phi_quad(1, float(e)) for e in _RADIAL_TABLE_GRID])
    p2 = np.array([_phi_quad(2, float(e)) for e in _RADIAL_TABLE_GRID])
    try:
        np.savez_compressed(_CACHE_FILE, eta=_RADIAL_TABLE_GRID, phi1=p1, phi2=p2)
    except OSError:
        pass
    return p1, p2


_PHI1, _PHI2 = _load_radial_tables()
_PHI2_TAIL_C = float(_PHI2[-1] * RADIAL_ETA_MAX**6)


def phi0(eta):
    e = np.asarray(eta, dtype=float)
    return 2.0 * np.exp(-e * e)


def phi1(eta):
    e = np.abs(np.asarray(eta, dtype=float))
    direct = np.interp(e, _RADIAL_TABLE_GRID, _PHI1)
    # Point-nucleus asymptotic continuation.  Do not interpret this as a
    # finite-nuclear-size prediction at arbitrarily large momentum transfer.
    tail = 2.0 / np.maximum(e, 1e-15) ** 4
    return np.where(e > RADIAL_ETA_MAX, tail, direct)


def phi2(eta):
    e = np.abs(np.asarray(eta, dtype=float))
    direct = np.interp(e, _RADIAL_TABLE_GRID, _PHI2)
    tail = _PHI2_TAIL_C / np.maximum(e, 1e-15) ** 6
    return np.where(e > RADIAL_ETA_MAX, tail, direct)


def radial_series_eta(eta, B: float, nmax: int = 2, clip: bool = False):
    g = phi0(eta)
    if nmax >= 1:
        g = g + phi1(eta) / B
    if nmax >= 2:
        g = g + phi2(eta) / (B * B)
    return np.clip(g, 0.0, None) if clip else g


@lru_cache(maxsize=4096)
def _mass_cache(B_key: int, nmax: int):
    B = B_key / 1e6
    eta = _RADIAL_INT_GRID
    raw = radial_series_eta(eta, B, nmax=nmax, clip=False)
    pos = np.clip(raw, 0.0, None)
    neg = np.clip(-raw, 0.0, None)
    raw_mass = float(_trapz(eta * raw, eta))
    pos_mass = float(_trapz(eta * pos, eta))
    neg_mass = float(_trapz(eta * neg, eta))
    tail = 1.0 / (B * RADIAL_ETA_MAX**2) if nmax >= 1 else 0.0
    total = pos_mass + tail
    clipped_fraction = neg_mass / max(raw_mass + tail, 1e-30)
    return total, clipped_fraction


def radial_total_mass(B: float, nmax: int = 2):
    return _mass_cache(round(float(B) * 1e6), int(nmax))


def _dimensionless_moments(eta_cut: float, B: float, nmax: int = 2):
    """Return accepted mass and unnormalised eta^2/eta^4 numerators."""
    if eta_cut <= 0.0:
        return 0.0, 0.0, 0.0
    if eta_cut <= RADIAL_ETA_MAX:
        j = int(np.searchsorted(_RADIAL_INT_GRID, eta_cut, side="right"))
        eg = _RADIAL_INT_GRID[:j]
        if eg.size == 0 or eg[-1] < eta_cut:
            eg = np.append(eg, eta_cut)
        g = radial_series_eta(eg, B, nmax=nmax, clip=True)
        return (
            float(_trapz(eg * g, eg)),
            float(_trapz(eg**3 * g, eg)),
            float(_trapz(eg**5 * g, eg)),
        )
    eg = _RADIAL_INT_GRID
    g = radial_series_eta(eg, B, nmax=nmax, clip=True)
    mass = float(_trapz(eg * g, eg))
    n2 = float(_trapz(eg**3 * g, eg))
    n4 = float(_trapz(eg**5 * g, eg))
    if nmax >= 1:
        e0 = RADIAL_ETA_MAX
        mass += (1.0 / B) * (1.0 / e0**2 - 1.0 / eta_cut**2)
        n2 += (2.0 / B) * math.log(eta_cut / e0)
        n4 += (1.0 / B) * (eta_cut**2 - e0**2)
    return mass, n2, n4


def mu2_eta(eta_cut: float, B: float, nmax: int = 2) -> float:
    mass, n2, _ = _dimensionless_moments(float(eta_cut), float(B), nmax=nmax)
    return n2 / mass if mass > 0.0 else 0.0


def radial_moments(chi_c2: float, B: float, theta_cut: float, nmax: int = 2):
    if chi_c2 <= 0.0 or theta_cut <= 0.0:
        return 0.0, 0.0, 0.0
    s = math.sqrt(chi_c2 * B)
    eta_cut = theta_cut / s
    mass, n2, n4 = _dimensionless_moments(eta_cut, B, nmax=nmax)
    total, clipped = radial_total_mass(B, nmax=nmax)
    if mass <= 0.0:
        return 0.0, 0.0, 0.0
    Fc = min(max(mass / total, 0.0), 1.0)
    return Fc, s * s * n2 / mass, s**4 * n4 / mass


def radial_tail_ratio(theta: float, chi_c2: float, B: float, nmax: int = 2):
    """Internal point-nucleus tail check h*theta^3/(2 chi_c^2) -> 1."""
    if theta <= 0 or chi_c2 <= 0:
        return np.nan
    s = math.sqrt(chi_c2 * B)
    eta = theta / s
    g = float(radial_series_eta(eta, B, nmax=nmax, clip=True))
    total, _ = radial_total_mass(B, nmax=nmax)
    # P=g/(2*pi*s^2)/total; h=theta*g/s^2/total
    h = theta * g / (s * s * total)
    return h * theta**3 / (2.0 * chi_c2)


def radial_cdf_eta(B: float, nmax: int = 2):
    eta = _RADIAL_INT_GRID
    g = radial_series_eta(eta, B, nmax=nmax, clip=True)
    cum = cumulative_trapezoid(eta * g, eta, initial=0.0)
    total, clipped = radial_total_mass(B, nmax=nmax)
    cdf = np.maximum.accumulate(cum / total)
    return eta, cdf, clipped


# ---------------------------------------------------------------------------
# Moliere material/path parameters and exact reduced representation
# ---------------------------------------------------------------------------
def chi_c2_single(Z: float, A: float, X: float, p_mev: float, beta: float) -> float:
    return 0.157 * Z * (Z + 1.0) * X / A / (p_mev * beta) ** 2


def chi_a2_single(Z: float, p_mev: float, beta: float) -> float:
    return (
        2.007e-5 * Z ** (2.0 / 3.0) * (1.0 + 3.34 * (Z * ALPHA / beta) ** 2) / p_mev**2
    )


def solve_B(chi_c2: float, chi_a2: float) -> float:
    rhs = math.log(chi_c2 / (1.167 * chi_a2))
    if rhs < 1.0:
        raise ValueError(f"Moliere Omega0 below validity threshold: lnOmega={rhs:.4g}")
    B = max(rhs + math.log(max(rhs, 1.001)), 1.5)
    for _ in range(80):
        step = (B - math.log(B) - rhs) / (1.0 - 1.0 / B)
        B -= step
        if abs(step) < 1e-12:
            return float(B)
    raise RuntimeError("B solve failed")


def constant_path_parameters(X_by_material: dict[str, float], p_gev: float):
    b = float(beta_of(p_gev))
    p_mev = p_gev * MEV
    c2 = 0.0
    log_num = 0.0
    weight = 0.0
    for name in MAT_ORDER:
        X = float(X_by_material.get(name, 0.0))
        if X <= 0.0:
            continue
        m = MATERIALS[name]
        dc2 = chi_c2_single(m.Z, m.A, X, p_mev, b)
        c2 += dc2
        # At common p, dc2 is proportional to the Lynch-Dahl serial weight.
        log_num += dc2 * math.log(chi_a2_single(m.Z, p_mev, b))
        weight += dc2
    if weight == 0.0:
        return dict(chi_c2=0.0, chi_a2=1.0, B=1.0)
    a2 = math.exp(log_num / weight)
    return dict(chi_c2=c2, chi_a2=a2, B=solve_B(c2, a2))


def x_over_x0_from_X(X_by_material: dict[str, float]) -> float:
    return sum(
        float(X_by_material.get(n, 0.0)) / MATERIALS[n].rho / MATERIALS[n].X0
        for n in MAT_ORDER
    )


def reduced_parameters(X_by_material: dict[str, float], p_gev: float):
    pars = constant_path_parameters(X_by_material, p_gev)
    xx0 = x_over_x0_from_X(X_by_material)
    tspace = float(theta_space_highland(p_gev, xx0))
    R = pars["chi_c2"] / (tspace * tspace)
    return dict(
        **pars,
        x_over_x0=xx0,
        theta_space=tspace,
        R=R,
        sqrt2RB=math.sqrt(2.0 * R * pars["B"]),
    )


def constant_calibration(
    X_by_material: dict[str, float],
    p_gev: float,
    theta_cut: float = THETA_CUT,
    nmax: int = 2,
):
    rp = reduced_parameters(X_by_material, p_gev)
    Fc, M2, M4 = radial_moments(rp["chi_c2"], rp["B"], theta_cut, nmax=nmax)
    trms = math.sqrt(M2)
    t0 = rp["theta_space"] / math.sqrt(2.0)
    k = theta_cut / t0
    eta_cut = theta_cut / math.sqrt(rp["chi_c2"] * rp["B"])
    mu2 = mu2_eta(eta_cut, rp["B"], nmax=nmax)
    # Exact within the radial model:
    exact_ratio2 = rp["R"] * rp["B"] * mu2
    return dict(
        **rp,
        Fc=Fc,
        M2=M2,
        M4=M4,
        theta_rms=trms,
        epsilon=trms / rp["theta_space"] - 1.0,
        k=k,
        eta_cut=eta_cut,
        mu2=mu2,
        exact_ratio2=exact_ratio2,
        clipped_fraction=radial_total_mass(rp["B"], nmax=nmax)[1],
    )


def epsilon_asymptotic(eta_cut: float, R: float, eta1: float = 1.0):
    v = 1.0 + 2.0 * R * math.log(eta_cut / eta1)
    return math.sqrt(v) - 1.0 if v > 0.0 else np.nan


def fit_eta1(eta_values: Sequence[float], eps_values: Sequence[float], R: float):
    """Fixed-slope intercept fit for the asymptotic law.

    This helper deliberately fixes the logarithmic slope at ``2R`` and should
    therefore be treated only as a diagnostic.  For physical interpretation
    use :func:`fit_log_asymptote`, which fits slope and intercept jointly.
    """
    eta = np.asarray(eta_values, dtype=float)
    eps = np.asarray(eps_values, dtype=float)
    y = ((1.0 + eps) ** 2 - 1.0) / (2.0 * R)
    ln_eta1 = float(np.mean(np.log(eta) - y))
    return math.exp(ln_eta1)


def fit_log_asymptote(
    eta_values: Sequence[float], eps_values: Sequence[float], R: float | None = None
):
    """Jointly fit the large-acceptance logarithmic slope and intercept.

    The fitted model is

        y = (1 + eps)^2 - 1 = slope * ln(eta) + intercept
          = slope * ln(eta / eta1),

    so ``eta1 = exp(-intercept/slope)``.  If ``R`` is supplied, the return
    dictionary also reports the expected Rutherford/Moliere slope ``2R`` and
    the fitted-to-expected slope ratio.

    No claim of a universal ``eta1`` is built into this function.  The tail
    coefficient fixes the asymptotic slope, while the finite core contribution
    determines the path-dependent intercept.
    """
    eta = np.asarray(eta_values, dtype=float)
    eps = np.asarray(eps_values, dtype=float)
    if eta.ndim != 1 or eps.ndim != 1 or eta.size != eps.size or eta.size < 2:
        raise ValueError("eta_values and eps_values must be equal-length 1-D arrays")
    if np.any(~np.isfinite(eta)) or np.any(eta <= 0) or np.any(~np.isfinite(eps)):
        raise ValueError("fit inputs must be finite and eta must be positive")
    x = np.log(eta)
    y = (1.0 + eps) ** 2 - 1.0
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(slope)
    intercept = float(intercept)
    eta1 = math.exp(-intercept / slope) if slope != 0.0 else np.nan
    fitted = slope * x + intercept
    resid = y - fitted
    out = dict(
        slope=slope,
        intercept=intercept,
        eta1=eta1,
        rms_residual=float(np.sqrt(np.mean(resid * resid))),
        max_abs_residual=float(np.max(np.abs(resid))),
    )
    if R is not None:
        expected = 2.0 * float(R)
        out.update(
            slope_expected=expected,
            slope_ratio=slope / expected if expected != 0.0 else np.nan,
            slope_rel_error=slope / expected - 1.0 if expected != 0.0 else np.nan,
        )
    return out


def efficiency_constant(
    X_by_material: dict[str, float], p_gev: float, k: float, nmax: int = 2
):
    rp = reduced_parameters(X_by_material, p_gev)
    cut = k * rp["theta_space"] / math.sqrt(2.0)
    Fc, M2, M4 = radial_moments(rp["chi_c2"], rp["B"], cut, nmax=nmax)
    var = M4 - M2 * M2
    return math.sqrt(Fc) * M2 / math.sqrt(var) if Fc > 0 and var > 0 else 0.0


def optimal_k_constant(X_by_material: dict[str, float], p_gev: float, nmax: int = 2):
    r = minimize_scalar(
        lambda k: -efficiency_constant(X_by_material, p_gev, k, nmax=nmax),
        bounds=(0.5, 8.0),
        method="bounded",
        options={"xatol": 2e-4},
    )
    return float(r.x), float(-r.fun)


# ---------------------------------------------------------------------------
# Energy-loss-aware segmented approximation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer:
    material: str
    thickness_cm: float


def path_x_over_x0(path: Sequence[Layer]) -> float:
    return sum(
        l.thickness_cm / MATERIALS[l.material].X0 for l in path if l.thickness_cm > 0.0
    )


def path_mass(path: Sequence[Layer]) -> float:
    return sum(
        l.thickness_cm * MATERIALS[l.material].rho for l in path if l.thickness_cm > 0.0
    )


def slice_path(path: Sequence[Layer], p_in: float, tol: float = P_BETA_SLICE_TOL):
    """Adaptive CSDA slices with <tol fractional p*beta change per slice.

    The representative p*beta reproduces the slice integral of 1/(p beta)^2
    by Simpson quadrature.
    """
    E = math.hypot(p_in, M_MU)
    out = []
    for layer in path:
        if layer.thickness_cm <= 0.0:
            continue
        m = MATERIALS[layer.material]
        X_layer = m.rho * layer.thickness_cm
        E_out = energy_after(E, layer.material, X_layer)
        drop = _pbeta_of_E(E) / _pbeta_of_E(E_out) - 1.0
        n = max(1, int(math.ceil(drop / tol)))
        dX = X_layer / n
        for _ in range(n):
            E0 = E
            Em = energy_after(E0, layer.material, 0.5 * dX)
            E1 = energy_after(E0, layer.material, dX)
            q0, qm, q1 = _pbeta_of_E(E0), _pbeta_of_E(Em), _pbeta_of_E(E1)
            integ = dX / 6.0 * (1.0 / q0**2 + 4.0 / qm**2 + 1.0 / q1**2)
            qrep = math.sqrt(dX / integ)
            prep = _p_of_pbeta(qrep)
            out.append(
                dict(
                    material=layer.material,
                    X=dX,
                    thickness_cm=dX / m.rho,
                    p=prep,
                    beta=qrep / prep,
                    pbeta=qrep,
                )
            )
            E = E1
    return out, E


def accumulate_moliere_pofx(slices, screening_weight: str = "dchi_c2"):
    """Segmented characteristic-strength accumulation for a degrading path.

    ``chi_c^2`` is additive at the single-scatter-strength level.  The effective
    screening logarithm is *not uniquely prescribed* by the manuscript for a
    non-stationary momentum profile, so two explicit continuations of the
    common-p Lynch--Dahl serial rule are supported:

    ``dchi_c2``
        weight each local ``ln chi_a^2`` by the local ``dchi_c^2``.  This gives
        additional weight to downstream, lower-momentum slices.

    ``serial``
        retain the common-p material weight ``Z(Z+1) X/A`` slice by slice, with
        only the local screening scale evaluated at ``p_j,beta_j``.

    The two choices are algebraically identical at constant momentum.  Their
    difference at finite energy loss is reported as a model systematic rather
    than hidden as a convention.  Neither is promoted to a theorem; both need
    independent transport validation for publication-level absolute accuracy.
    """
    if screening_weight not in {"dchi_c2", "serial"}:
        raise ValueError("screening_weight must be 'dchi_c2' or 'serial'")
    c2 = 0.0
    log_num = 0.0
    log_den = 0.0
    for s in slices:
        m = MATERIALS[s["material"]]
        p_mev = s["p"] * MEV
        dc2 = chi_c2_single(m.Z, m.A, s["X"], p_mev, s["beta"])
        a2 = chi_a2_single(m.Z, p_mev, s["beta"])
        serial_w = m.Z * (m.Z + 1.0) * s["X"] / m.A
        w = dc2 if screening_weight == "dchi_c2" else serial_w
        c2 += dc2
        log_num += w * math.log(a2)
        log_den += w
    if c2 <= 0.0 or log_den <= 0.0:
        return 0.0, 1.0, 1.0
    a2 = math.exp(log_num / log_den)
    return c2, a2, solve_B(c2, a2)


def highland_core_pofx_model(slices, x_over_x0_total: float):
    """Energy-loss-aware Highland-core construction used for matched comparison.

    The local Gaussian scattering power is integrated as 1/(p beta)^2, while
    the empirical Highland logarithmic factor is applied once to the total path.
    This reduces exactly to Eq. (1) at constant momentum.  It is a stated model
    construction, not a uniquely defined extension of the Highland fit.
    """
    core2 = 0.0
    wsum = 0.0
    beta2_weighted = 0.0
    for s in slices:
        xj = s["thickness_cm"] / MATERIALS[s["material"]].X0
        pb_mev = s["pbeta"] * MEV
        core2 += 13.6**2 * xj / (pb_mev * pb_mev)
        w = xj / (s["pbeta"] ** 2)
        wsum += w
        beta2_weighted += w * s["beta"] ** 2
    if wsum <= 0.0:
        return 0.0, 1.0
    beta_eff2 = beta2_weighted / wsum
    logf = 1.0 + 0.038 * math.log(x_over_x0_total / beta_eff2)
    return math.sqrt(core2) * logf, math.sqrt(beta_eff2)


def X_by_material_from_path(path: Sequence[Layer]):
    d = {m: 0.0 for m in MAT_ORDER}
    for l in path:
        d[l.material] += l.thickness_cm * MATERIALS[l.material].rho
    return d


def calibrate_pofx(
    path: Sequence[Layer],
    p_in: float,
    theta_cut: float = THETA_CUT,
    tol: float = P_BETA_SLICE_TOL,
    nmax: int = 2,
    screening_weight: str = "dchi_c2",
):
    if path_x_over_x0(path) <= 0:
        raise ValueError("empty path")
    slices, E_out = slice_path(path, p_in, tol=tol)
    c2, a2, B = accumulate_moliere_pofx(slices, screening_weight=screening_weight)
    Fc, M2, M4 = radial_moments(c2, B, theta_cut, nmax=nmax)
    trms = math.sqrt(max(M2, 0.0))
    xx0 = path_x_over_x0(path)
    th0_px, beta_eff = highland_core_pofx_model(slices, xx0)
    tspace_px = math.sqrt(2.0) * th0_px
    tspace_incident = float(theta_space_highland(p_in, xx0))
    p_out = _p_of_E(E_out)
    return dict(
        p_in=p_in,
        p_out=p_out,
        dp_over_p=p_out / p_in - 1.0,
        delta_E=math.hypot(p_in, M_MU) - E_out,
        x_over_x0=xx0,
        mass=path_mass(path),
        n_slices=len(slices),
        chi_c2=c2,
        chi_a2=a2,
        B=B,
        Omega0=c2 / (1.167 * a2),
        Fc=Fc,
        M2=M2,
        M4=M4,
        theta_rms=trms,
        theta0_pofx=th0_px,
        theta_space_pofx=tspace_px,
        beta_eff=beta_eff,
        k_pofx=theta_cut / th0_px,
        eta_cut=theta_cut / math.sqrt(c2 * B),
        epsilon_matched=trms / tspace_px - 1.0,
        epsilon_mixed=trms / tspace_incident - 1.0,
        theta_space_incident=tspace_incident,
        clipped_fraction=radial_total_mass(B, nmax=nmax)[1],
        screening_weight=screening_weight,
        slices=slices,
    )


def efficiency_pofx(
    path: Sequence[Layer],
    p_in: float,
    k: float,
    nmax: int = 2,
    screening_weight: str = "dchi_c2",
):
    slices, _ = slice_path(path, p_in)
    c2, _, B = accumulate_moliere_pofx(slices, screening_weight=screening_weight)
    t0, _ = highland_core_pofx_model(slices, path_x_over_x0(path))
    Fc, M2, M4 = radial_moments(c2, B, k * t0, nmax=nmax)
    var = M4 - M2 * M2
    return math.sqrt(Fc) * M2 / math.sqrt(var) if Fc > 0 and var > 0 else 0.0


def optimal_k_pofx(
    path: Sequence[Layer], p_in: float, nmax: int = 2, screening_weight: str = "dchi_c2"
):
    r = minimize_scalar(
        lambda k: (
            -efficiency_pofx(
                path, p_in, k, nmax=nmax, screening_weight=screening_weight
            )
        ),
        bounds=(0.5, 8.0),
        method="bounded",
        options={"xatol": 2e-4},
    )
    return float(r.x), float(-r.fun)


# ---------------------------------------------------------------------------
# Cached calibration / radial sampler for detector simulation
# ---------------------------------------------------------------------------
def layers_from_segment_thicknesses(seg):
    """Convert [Al_up, Cu_up, Pb, Cu_down, Al_down] cm into ordered Layers."""
    au, cu, pb, cd, ad = map(float, seg)
    vals = [("Al", au), ("Cu", cu), ("Pb", pb), ("Cu", cd), ("Al", ad)]
    return tuple(Layer(m, t) for m, t in vals if t > 0.0)


class PofxCache:
    """Cache p(X) calibration on incident-p and ordered areal-density bins."""

    def __init__(
        self,
        nmax: int = 2,
        tol: float = P_BETA_SLICE_TOL,
        screening_weight: str = "dchi_c2",
    ):
        if screening_weight not in {"dchi_c2", "serial"}:
            raise ValueError("screening_weight must be 'dchi_c2' or 'serial'")
        self.nmax = nmax
        self.tol = tol
        self.screening_weight = screening_weight
        self._cache = {}
        self.max_clipped = 0.0

    @staticmethod
    def _key(p, seg_cm, cut):
        mats = ("Al", "Cu", "Pb", "Cu", "Al")
        X = [float(t) * MATERIALS[m].rho for t, m in zip(seg_cm, mats)]
        return (
            round(float(p) / P_CACHE_STEP),
            *(round(x / SEG_CACHE_STEP) for x in X),
            round(float(cut) / CUT_CACHE_STEP),
        )

    @staticmethod
    def _decode(key):
        p = key[0] * P_CACHE_STEP
        mats = ("Al", "Cu", "Pb", "Cu", "Al")
        X = [k * SEG_CACHE_STEP for k in key[1:6]]
        seg = [x / MATERIALS[m].rho for x, m in zip(X, mats)]
        cut = key[6] * CUT_CACHE_STEP
        return p, seg, cut

    def calibration(self, p, seg_cm, cut=THETA_CUT):
        key = self._key(p, seg_cm, cut)
        if key not in self._cache:
            pp, seg, cc = self._decode(key)
            path = layers_from_segment_thicknesses(seg)
            if not path:
                self._cache[key] = None
            else:
                r = calibrate_pofx(
                    path,
                    pp,
                    theta_cut=cc,
                    tol=self.tol,
                    nmax=self.nmax,
                    screening_weight=self.screening_weight,
                )
                self.max_clipped = max(self.max_clipped, r["clipped_fraction"])
                self._cache[key] = r
        return self._cache[key]

    def arrays(self, p, segments, cut=THETA_CUT):
        p = np.asarray(p, float)
        segments = np.asarray(segments, float)
        if segments.shape != (p.size, 5):
            raise ValueError("segments must have shape (N,5)")
        cuts = (
            np.full(p.size, float(cut)) if np.ndim(cut) == 0 else np.asarray(cut, float)
        )
        trms = np.zeros(p.size)
        eps_match = np.zeros(p.size)
        eps_mix = np.zeros(p.size)
        pout = p.copy()
        groups = {}
        for i in range(p.size):
            groups.setdefault(self._key(p[i], segments[i], cuts[i]), []).append(i)
        for key, idx_list in groups.items():
            idx = np.asarray(idx_list, int)
            # Decode the bucket exactly once; do not let the first raw event
            # subtly choose one side of a cache bin.
            pp, seg, cc = self._decode(key)
            path = layers_from_segment_thicknesses(seg)
            r = (
                None
                if not path
                else calibrate_pofx(
                    path,
                    pp,
                    theta_cut=cc,
                    tol=self.tol,
                    nmax=self.nmax,
                    screening_weight=self.screening_weight,
                )
            )
            self._cache[key] = r
            if r is None:
                continue
            self.max_clipped = max(self.max_clipped, r["clipped_fraction"])
            trms[idx] = r["theta_rms"]
            eps_match[idx] = r["epsilon_matched"]
            eps_mix[idx] = r["epsilon_mixed"]
            pout[idx] = r["p_out"]
        return dict(
            theta_rms=trms, epsilon_matched=eps_match, epsilon_mixed=eps_mix, p_out=pout
        )

    def sample(self, p, segments, rng):
        p = np.asarray(p, float)
        segments = np.asarray(segments, float)
        tx = np.zeros(p.size)
        ty = np.zeros(p.size)
        # Group by cache key to vectorise sampling from each radial CDF.
        keys = [
            self._key(p[i], segments[i], THETA_CUT)[:-1]
            + (round(THETA_CUT / CUT_CACHE_STEP),)
            for i in range(p.size)
        ]
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        for k, idx_list in groups.items():
            idx = np.asarray(idx_list, int)
            r = self.calibration(p[idx[0]], segments[idx[0]], THETA_CUT)
            if r is None:
                continue
            B, c2 = r["B"], r["chi_c2"]
            eta_grid, cdf, clipped = radial_cdf_eta(B, nmax=self.nmax)
            self.max_clipped = max(self.max_clipped, clipped)
            u = rng.random(idx.size)
            eta = np.empty(idx.size)
            end = float(cdf[-1])
            core = u <= end
            eta[core] = np.interp(u[core], cdf, eta_grid)
            if np.any(~core):
                rr = (u[~core] - end) / max(1.0 - end, 1e-30)
                eta[~core] = RADIAL_ETA_MAX / np.sqrt(
                    np.maximum(1.0 - rr, np.finfo(float).tiny)
                )
            theta = math.sqrt(c2 * B) * eta
            az = rng.uniform(0.0, 2.0 * math.pi, idx.size)
            tx[idx] = theta * np.cos(az)
            ty[idx] = theta * np.sin(az)
        return tx, ty
