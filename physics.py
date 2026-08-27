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

The radial n<=2 Moliere series is retained as a point-nucleus reference and as
the current detector sampler.  The finite-size theory path instead inserts its
kernel directly in the single-scatter characteristic exponent and performs an
unexpanded Hankel transform.  The former 100 mrad splice is superseded and the
finite-size detector sampler is deliberately gated until an inverse CDF is
derived from that same transformed density.
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
from scipy.integrate import IntegrationWarning, quad, cumulative_trapezoid, simpson
from scipy.optimize import minimize_scalar
from scipy.special import j0, j1, jv, k1, spherical_jn

from config import (
    ALPHA,
    CUT_CACHE_STEP,
    FINITE_SIZE_PRODUCTION_ENABLED,
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
HBARC_MEV_FM = 197.3
FORM_FACTOR_MODELS = ("none", "gaussian", "uniform_sphere")
FF_MODELS = ("point", "gauss", "sphere")


# Moliere's B equation is written with chi_a'^2 = 1.167 chi_a^2, while the
# screened Rutherford denominator contains the unprimed chi_a^2.  Retaining
# that distinction cancels the Euler-constant term in the small-screening
# expansion and provides a clean closure against the reduced Moliere series.
MOLIERE_SCREENING_FACTOR = 1.167


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


def dimensionless_moments_quad(eta_cut: float, B: float, nmax: int = 2):
    """Adaptive-Gauss--Kronrod evaluation of the accepted reduced moments.

    This is intentionally separate from :func:`_dimensionless_moments`, whose
    production path uses a fixed integration grid.  The radial functions are
    still the independently quadrature-generated Phi tables, but no production
    cumulative/trapezoidal moment table is reused here.
    """
    eta_cut = float(eta_cut)
    B = float(B)
    if eta_cut <= 0.0:
        return 0.0, 0.0, 0.0

    upper = min(eta_cut, RADIAL_ETA_MAX)

    def integrate_power(power):
        def f(e):
            return e**power * float(radial_series_eta(e, B, nmax=nmax, clip=True))

        # Explicit break points prevent interpolation knots at the core/tail
        # transition from degrading QUADPACK's error estimate.
        points = [x for x in (1.0, 2.0, 5.0, 10.0, 20.0) if x < upper]
        bounds = [0.0, *points, upper]
        value = 0.0
        error = 0.0
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)
                part, err = quad(
                    f, lo, hi, limit=200, epsabs=2e-11, epsrel=2e-10
                )
            value += part
            error += err
        return float(value), float(error)

    mass, emass = integrate_power(1)
    n2, en2 = integrate_power(3)
    n4, en4 = integrate_power(5)
    if eta_cut > RADIAL_ETA_MAX and nmax >= 1:
        e0 = RADIAL_ETA_MAX
        mass += (1.0 / B) * (1.0 / e0**2 - 1.0 / eta_cut**2)
        n2 += (2.0 / B) * math.log(eta_cut / e0)
        n4 += (1.0 / B) * (eta_cut**2 - e0**2)
    return mass, n2, n4, emass, en2, en4


def sample_radial_eta(B: float, size: int, rng, nmax: int = 2):
    """Draw reduced radial angles directly from the normalized Moliere PDF."""
    return radial_eta_from_uniform(B, rng.random(int(size)), nmax=nmax)


def radial_eta_from_uniform(B: float, uniform, nmax: int = 2):
    """Inverse-CDF transform for supplied uniform variates.

    Supplying the variates separately lets cache-grid studies use exact common
    random numbers even when the finer grid changes event grouping.
    """
    eta_grid, cdf, _ = radial_cdf_eta(float(B), nmax=nmax)
    u = np.asarray(uniform, float)
    eta = np.empty(u.size)
    end = float(cdf[-1])
    core = u <= end
    eta[core] = np.interp(u[core], cdf, eta_grid)
    if np.any(~core):
        q = (u[~core] - end) / max(1.0 - end, 1e-30)
        eta[~core] = RADIAL_ETA_MAX / np.sqrt(
            np.maximum(1.0 - q, np.finfo(float).tiny)
        )
    return eta


def mu2_eta(eta_cut: float, B: float, nmax: int = 2) -> float:
    mass, n2, _ = _dimensionless_moments(float(eta_cut), float(B), nmax=nmax)
    return n2 / mass if mass > 0.0 else 0.0


def _one_minus_xk1(x):
    """Evaluate ``1 - x K1(x)`` without cancellation at small ``x``."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 1.0e-3
    positive = small & (x > 0.0)
    out[small & ~positive] = 0.0
    if np.any(positive):
        z = x[positive]
        logterm = np.log(2.0 / z) - np.euler_gamma
        # DLMF 10.31.1 through O(x^4).  The retained term is already below
        # double-precision relevance after multiplication by the scattering
        # multiplicity over the range where this branch is used.
        out[positive] = (
            0.5 * z * z * (logterm + 0.5)
            + (z**4 / 16.0) * (logterm + 1.25)
        )
    if np.any(~small):
        z = x[~small]
        out[~small] = 1.0 - z * k1(z)
    return out


def screened_rutherford_characteristic(t, B: float):
    """Point-nucleus characteristic function in Moliere reduced variables.

    ``t = u sqrt(chi_c^2 B)`` and the screened single-scatter kernel is
    proportional to ``(theta^2 + chi_a^2)^-2``.  The B equation uses
    ``chi_a'^2 = 1.167 chi_a^2``.  No expansion in ``1/B`` is made.
    """
    B = float(B)
    t = np.asarray(t, dtype=float)
    multiplicity = MOLIERE_SCREENING_FACTOR * math.exp(B) / B
    x = np.exp(-0.5 * B) * t / math.sqrt(MOLIERE_SCREENING_FACTOR)
    omega = multiplicity * _one_minus_xk1(x)
    return np.exp(-omega)


def transform_moments_g1(
    chi_c2: float,
    B: float,
    theta_cut: float,
    *,
    t_max: float = 24.0,
    points_per_bessel_period: int = 80,
):
    """Accepted radial moments from the unexpanded G=1 Hankel transform.

    This is the Step A.4 closure implementation.  It evaluates the exact
    characteristic exponent of the screened Rutherford kernel and integrates
    analytic finite-cut Bessel moment kernels.  It deliberately does not call
    the existing ``Phi^(n)`` tables.
    """
    if chi_c2 <= 0.0 or theta_cut <= 0.0:
        return 0.0, 0.0, 0.0
    scale = math.sqrt(float(chi_c2) * float(B))
    eta = float(theta_cut) / scale
    # Resolve the fastest Bessel oscillation at the requested cut.  The
    # characteristic function is <1e-25 by t=24 for the B range used here.
    dt = min(2.0e-3, 2.0 * math.pi / (points_per_bessel_period * eta))
    n = max(4001, int(math.ceil(float(t_max) / dt)) + 1)
    t = np.linspace(0.0, float(t_max), n)
    characteristic = screened_rutherford_characteristic(t, B)
    x = eta * t
    J1 = j1(x)
    J2 = jv(2, x)
    J3 = jv(3, x)
    inv_t = np.zeros_like(t)
    inv_t[1:] = 1.0 / t[1:]

    mass_kernel = eta * J1
    n2_kernel = eta**3 * J1 - 2.0 * eta**2 * J2 * inv_t
    n4_kernel = (
        eta**5 * J1
        - 4.0 * eta**4 * J2 * inv_t
        + 8.0 * eta**3 * J3 * inv_t * inv_t
    )
    mass = float(simpson(characteristic * mass_kernel, x=t))
    n2 = float(simpson(characteristic * n2_kernel, x=t))
    n4 = float(simpson(characteristic * n4_kernel, x=t))
    if mass <= 0.0:
        return 0.0, 0.0, 0.0
    return (
        min(max(mass, 0.0), 1.0),
        scale * scale * n2 / mass,
        scale**4 * n4 / mass,
    )


def _normalize_form_factor(model: str | None) -> str:
    model = "none" if model is None else str(model).lower().replace("-", "_")
    aliases = {"sphere": "uniform_sphere", "uniform": "uniform_sphere"}
    model = aliases.get(model, model)
    if model not in FORM_FACTOR_MODELS:
        raise ValueError(f"form_factor must be one of {FORM_FACTOR_MODELS}")
    return model


def _normalize_ff_model(ff_model: str) -> str:
    """Map the validation API names onto the legacy internal model names."""
    aliases = {
        "point": "none",
        "gauss": "gaussian",
        "sphere": "uniform_sphere",
    }
    try:
        return aliases[str(ff_model)]
    except KeyError as exc:
        raise ValueError(f"ff_model must be one of {FF_MODELS}") from exc


def nuclear_radius_fm(A: float) -> float:
    return 1.2 * float(A) ** (1.0 / 3.0)


def nuclear_form_factor_sq(theta, p_gev: float, A: float, model: str):
    """Squared elastic nuclear form factor in the small-angle q=p*theta limit."""
    model = _normalize_form_factor(model)
    theta = np.asarray(theta, float)
    if model == "none":
        return np.ones_like(theta)
    x = p_gev * MEV * theta * nuclear_radius_fm(A) / HBARC_MEV_FM
    if model == "gaussian":
        return np.exp(-(x * x) / 5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(x) > 1e-12, 3.0 * spherical_jn(1, x) / x, 1.0)
    return ratio * ratio


def proton_form_factor_sq(theta, p_gev: float):
    """Dipole proton form factor squared, using q=p*theta consistently."""
    theta = np.asarray(theta, float)
    q2_gev2 = (float(p_gev) * theta) ** 2
    return (1.0 + q2_gev2 / 0.71) ** -4


def finite_size_kernel(theta, components, model: str, include_incoherent=True):
    """Scattering-kernel multiplier used inside the eikonal transform.

    The incoherent approximation interpolates from the exact low-q
    normalisation to an ``A |F_p|^2`` quasi-elastic nucleon floor after loss of
    nuclear coherence:

        G = |F_N|^2 + A/[Z(Z+1)] (1-|F_N|^2) |F_p|^2.

    Thus G(0)=1, the intermediate floor is A/[Z(Z+1)], and the floor terminates
    at the proton scale.  Omitting the second term is retained as a systematic.
    """
    model = _normalize_form_factor(model)
    theta = np.asarray(theta, float)
    if model == "none":
        return np.ones_like(theta)
    rows = _tail_component_tuple(components)
    if not rows:
        raise ValueError("finite-size kernel requires scattering components")
    out = np.zeros_like(theta)
    for frac, Z, A, p in rows:
        F2 = nuclear_form_factor_sq(theta, p, A, model)
        term = F2
        if include_incoherent:
            term = term + A / (Z * (Z + 1.0)) * (1.0 - F2) * proton_form_factor_sq(
                theta, p
            )
        out += frac * term
    return out


@lru_cache(maxsize=256)
def _finite_size_characteristic_table(
    chi_c2: float,
    B: float,
    components,
    model: str,
    include_incoherent: bool,
):
    """Numerically accumulate Omega(t) and return exp(-Omega) on a fixed grid."""
    chi_c2 = float(chi_c2)
    B = float(B)
    model = _normalize_form_factor(model)
    components = _tail_component_tuple(components)
    a2 = chi_c2 * B / (MOLIERE_SCREENING_FACTOR * math.exp(B))
    a = math.sqrt(a2)
    # The q=p*theta convention is the same small-angle convention used by the
    # transport kernel. q=20 GeV/c makes the dipole-squared remainder negligible.
    p_min = min(row[3] for row in components)
    theta_max = max(2.0, 20.0 / p_min)
    theta = np.geomspace(max(a * 1.0e-5, 1.0e-12), theta_max, 7001)
    G = finite_size_kernel(theta, components, model, include_incoherent)
    density = 2.0 * chi_c2 * theta * G / (theta * theta + a2) ** 2
    dtheta = np.diff(theta)
    weights = np.empty_like(theta)
    weights[0] = 0.5 * dtheta[0]
    weights[-1] = 0.5 * dtheta[-1]
    weights[1:-1] = 0.5 * (dtheta[:-1] + dtheta[1:])
    weighted = weights * density

    t = np.linspace(0.0, 24.0, 12001)
    u = t / math.sqrt(chi_c2 * B)
    omega = np.empty_like(t)
    for start in range(0, t.size, 128):
        stop = min(start + 128, t.size)
        omega[start:stop] = (1.0 - j0(u[start:stop, None] * theta[None, :])) @ weighted
    omega[0] = 0.0
    characteristic = np.exp(-np.maximum(omega, 0.0))
    t.setflags(write=False)
    characteristic.setflags(write=False)
    return t, characteristic


def transform_moments_finite_size(
    chi_c2: float,
    B: float,
    theta_cut: float,
    components,
    model: str,
    *,
    include_incoherent: bool = True,
):
    """Accepted moments with the finite-size kernel inside the transform."""
    if chi_c2 <= 0.0 or theta_cut <= 0.0:
        return 0.0, 0.0, 0.0
    t, characteristic = _finite_size_characteristic_table(
        float(chi_c2),
        float(B),
        _tail_component_tuple(components),
        _normalize_form_factor(model),
        bool(include_incoherent),
    )
    scale = math.sqrt(float(chi_c2) * float(B))
    eta = float(theta_cut) / scale
    x = eta * t
    inv_t = np.zeros_like(t)
    inv_t[1:] = 1.0 / t[1:]
    J1 = j1(x)
    J2 = jv(2, x)
    J3 = jv(3, x)
    mass = float(simpson(characteristic * eta * J1, x=t))
    n2 = float(
        simpson(
            characteristic * (eta**3 * J1 - 2.0 * eta**2 * J2 * inv_t), x=t
        )
    )
    n4 = float(
        simpson(
            characteristic
            * (
                eta**5 * J1
                - 4.0 * eta**4 * J2 * inv_t
                + 8.0 * eta**3 * J3 * inv_t * inv_t
            ),
            x=t,
        )
    )
    if mass <= 0.0:
        return 0.0, 0.0, 0.0
    return min(max(mass, 0.0), 1.0), scale**2 * n2 / mass, scale**4 * n4 / mass


def untruncated_finite_size_moments(
    chi_c2: float,
    B: float,
    components,
    model: str,
    *,
    include_incoherent: bool = True,
):
    """Exact full-acceptance M2 and M4 for the compound-scatter kernel.

    For an isotropic compound Poisson law, the total radial second moment is
    the single-jump radial second cumulant.  The radial fourth moment is twice
    its square plus the single-jump fourth cumulant.  Finite nuclear and proton
    form factors make both integrals convergent.
    """
    chi_c2 = float(chi_c2)
    B = float(B)
    model = _normalize_form_factor(model)
    if model == "none":
        raise ValueError("point-nucleus untruncated M2 and M4 diverge")
    rows = _tail_component_tuple(components)
    a2 = chi_c2 * B / (MOLIERE_SCREENING_FACTOR * math.exp(B))
    p_min = min(row[3] for row in rows)
    theta_max = max(2.0, 40.0 / p_min)
    theta = np.geomspace(max(math.sqrt(a2) * 1.0e-6, 1.0e-13), theta_max, 20001)
    G = finite_size_kernel(theta, rows, model, include_incoherent)
    rate = 2.0 * chi_c2 * theta * G / (theta * theta + a2) ** 2
    kappa2 = float(simpson(theta**2 * rate, x=theta))
    kappa4 = float(simpson(theta**4 * rate, x=theta))
    return kappa2, 2.0 * kappa2 * kappa2 + kappa4


def transform_radial_density(
    theta,
    chi_c2: float,
    B: float,
    components,
    model: str,
    *,
    include_incoherent: bool = True,
):
    """Return the radial density h(theta)=2*pi*theta*P(theta)."""
    values = np.atleast_1d(np.asarray(theta, float))
    t, characteristic = _finite_size_characteristic_table(
        float(chi_c2),
        float(B),
        _tail_component_tuple(components),
        _normalize_form_factor(model),
        bool(include_incoherent),
    )
    scale = math.sqrt(float(chi_c2) * float(B))
    # Fixed-grid Simpson weights let the log-grid transforms run as bounded
    # matrix products rather than thousands of Python-level integrations.
    weights = np.ones_like(t)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= (t[1] - t[0]) / 3.0
    weighted = weights * t * characteristic
    out = np.empty_like(values)
    for start in range(0, values.size, 128):
        stop = min(start + 128, values.size)
        angles = values[start:stop]
        eta = angles / scale
        out[start:stop] = angles / scale**2 * (
            j0(eta[:, None] * t[None, :]) @ weighted
        )
    return float(out[0]) if np.ndim(theta) == 0 else out


@lru_cache(maxsize=256)
def finite_size_cdf_eta(
    chi_c2: float,
    B: float,
    components,
    model: str,
    include_incoherent: bool = True,
):
    """Transform-derived radial CDF on a reduced-angle grid."""
    t, characteristic = _finite_size_characteristic_table(
        float(chi_c2), float(B), _tail_component_tuple(components),
        _normalize_form_factor(model), bool(include_incoherent),
    )
    eta = np.unique(np.concatenate([
        np.linspace(0.0, 10.0, 3001),
        np.geomspace(10.0, 80.0, 1001),
    ]))
    # Composite Simpson weights for the fixed, odd-length uniform t grid.
    weights = np.ones_like(t)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= (t[1] - t[0]) / 3.0
    weighted = weights * characteristic
    cdf = np.empty_like(eta)
    for start in range(0, eta.size, 128):
        stop = min(start + 128, eta.size)
        e = eta[start:stop]
        cdf[start:stop] = e * (j1(e[:, None] * t[None, :]) @ weighted)
    cdf[0] = 0.0
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    if cdf[-1] < 1.0 - 2.0e-5:
        raise RuntimeError(f"finite-size CDF grid retains only {cdf[-1]:.8f} probability")
    # The residual is below the numerical closure target and assigning it to
    # the last grid point avoids silently dropping uniform variates near one.
    cdf[-1] = 1.0
    eta.setflags(write=False)
    cdf.setflags(write=False)
    return eta, cdf


def finite_size_eta_from_uniform(
    chi_c2: float,
    B: float,
    uniform,
    components,
    model: str,
    *,
    include_incoherent: bool = True,
):
    eta, cdf = finite_size_cdf_eta(
        float(chi_c2), float(B), _tail_component_tuple(components),
        _normalize_form_factor(model), bool(include_incoherent),
    )
    return np.interp(np.asarray(uniform, float), cdf, eta)


def finite_size_rho(chi_c2: float, B: float, components) -> float:
    """dchi_c2-weighted geometric FF-onset scale in Moliere units."""
    rows = _tail_component_tuple(components)
    if not rows:
        return np.nan
    theta_ff = math.exp(
        sum(
            frac
            * math.log(HBARC_MEV_FM / (p * MEV * nuclear_radius_fm(A)))
            for frac, _, A, p in rows
        )
    )
    return theta_ff / math.sqrt(float(chi_c2) * float(B))


def _tail_component_tuple(components):
    """Canonical (fraction, Z, A, p_GeV) tuples for the tail mixture."""
    if not components:
        return ()
    rows = tuple(tuple(map(float, row)) for row in components)
    total = sum(row[0] for row in rows)
    if total <= 0.0:
        return ()
    return tuple((w / total, z, a, p) for w, z, a, p in rows if w > 0.0)


def radial_moments(
    chi_c2: float,
    B: float,
    theta_cut: float,
    nmax: int = 2,
    *,
    tail_components=(),
    form_factor: str = "none",
):
    if chi_c2 <= 0.0 or theta_cut <= 0.0:
        return 0.0, 0.0, 0.0
    form_factor = _normalize_form_factor(form_factor)
    if form_factor != "none":
        return transform_moments_finite_size(
            chi_c2,
            B,
            theta_cut,
            tail_components,
            form_factor,
        )
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


def constant_tail_components(X_by_material: dict[str, float], p_gev: float):
    """Return dchi_c^2-weighted nuclear-tail components for a fixed-p path."""
    beta = float(beta_of(p_gev))
    p_mev = float(p_gev) * MEV
    rows = []
    for name in MAT_ORDER:
        X = float(X_by_material.get(name, 0.0))
        if X <= 0.0:
            continue
        m = MATERIALS[name]
        dc2 = chi_c2_single(m.Z, m.A, X, p_mev, beta)
        rows.append((dc2, m.Z, m.A, float(p_gev)))
    return _tail_component_tuple(rows)


def sliced_tail_components(slices):
    """Return local-p, dchi_c^2-weighted tail components for a p(X) path."""
    rows = []
    for item in slices:
        m = MATERIALS[item["material"]]
        dc2 = chi_c2_single(
            m.Z, m.A, item["X"], item["p"] * MEV, item["beta"]
        )
        rows.append((dc2, m.Z, m.A, item["p"]))
    return _tail_component_tuple(rows)


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
    form_factor: str = "none",
):
    rp = reduced_parameters(X_by_material, p_gev)
    form_factor = _normalize_form_factor(form_factor)
    components = constant_tail_components(X_by_material, p_gev)
    ff_diag = dict(form_factor=form_factor)
    if form_factor == "none":
        Fc, M2, M4 = radial_moments(
            rp["chi_c2"], rp["B"], theta_cut, nmax=nmax
        )
    else:
        Fc, M2, M4 = transform_moments_finite_size(
            rp["chi_c2"], rp["B"], theta_cut, components, form_factor
        )
    trms = math.sqrt(M2)
    t0 = rp["theta_space"] / math.sqrt(2.0)
    k = theta_cut / t0
    eta_cut = theta_cut / math.sqrt(rp["chi_c2"] * rp["B"])
    mu2 = M2 / (rp["chi_c2"] * rp["B"])
    exact_ratio2 = M2 / (rp["theta_space"] * rp["theta_space"])
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
        tail_components=components,
        **ff_diag,
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
    form_factor: str = "none",
):
    if path_x_over_x0(path) <= 0:
        raise ValueError("empty path")
    slices, E_out = slice_path(path, p_in, tol=tol)
    c2, a2, B = accumulate_moliere_pofx(slices, screening_weight=screening_weight)
    form_factor = _normalize_form_factor(form_factor)
    components = sliced_tail_components(slices)
    ff_diag = dict(form_factor=form_factor)
    if form_factor == "none":
        Fc, M2, M4 = radial_moments(c2, B, theta_cut, nmax=nmax)
    else:
        Fc, M2, M4 = transform_moments_finite_size(
            c2, B, theta_cut, components, form_factor
        )
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
        tail_components=components,
        slices=slices,
        **ff_diag,
    )


def calibrate_pofx_transform(
    path: Sequence[Layer],
    p_in: float,
    theta_cut: float = THETA_CUT,
    *,
    screening_weight: str = "dchi_c2",
    form_factor: str = "none",
    include_incoherent: bool = True,
):
    """Segmented-p calibration using the unspliced characteristic transform.

    This calculation is currently the theory/regeneration path.  Detector
    sampling remains a separate production gate because it requires an
    inverse-CDF table derived from the same transformed density.
    """
    if path_x_over_x0(path) <= 0.0:
        raise ValueError("empty path")
    slices, E_out = slice_path(path, p_in)
    c2, a2, B = accumulate_moliere_pofx(slices, screening_weight=screening_weight)
    components = sliced_tail_components(slices)
    model = _normalize_form_factor(form_factor)
    if model == "none":
        Fc, M2, M4 = transform_moments_g1(c2, B, theta_cut)
    else:
        Fc, M2, M4 = transform_moments_finite_size(
            c2,
            B,
            theta_cut,
            components,
            model,
            include_incoherent=include_incoherent,
        )
    xx0 = path_x_over_x0(path)
    theta0_px, beta_eff = highland_core_pofx_model(slices, xx0)
    theta_space_px = math.sqrt(2.0) * theta0_px
    theta_space_incident = float(theta_space_highland(p_in, xx0))
    theta_rms = math.sqrt(max(M2, 0.0))
    p_out = _p_of_E(E_out)
    return dict(
        p_in=float(p_in),
        p_out=p_out,
        dp_over_p=p_out / p_in - 1.0,
        x_over_x0=xx0,
        chi_c2=c2,
        chi_a2=a2,
        B=B,
        Fc=Fc,
        M2=M2,
        M4=M4,
        theta_rms=theta_rms,
        theta_space_pofx=theta_space_px,
        theta_space_incident=theta_space_incident,
        epsilon_matched=theta_rms / theta_space_px - 1.0,
        epsilon_mixed=theta_rms / theta_space_incident - 1.0,
        ratio2_matched=M2 / theta_space_px**2,
        ratio2_mixed=M2 / theta_space_incident**2,
        R_matched=c2 / theta_space_px**2,
        R_mixed=c2 / theta_space_incident**2,
        mu2=M2 / (c2 * B),
        eta_cut=theta_cut / math.sqrt(c2 * B),
        beta_eff=beta_eff,
        screening_weight=screening_weight,
        form_factor=model,
        include_incoherent=bool(include_incoherent),
        tail_components=components,
        slices=slices,
    )


# ---------------------------------------------------------------------------
# Pre-registered finite-size validation APIs
# ---------------------------------------------------------------------------

_VALIDATION_PATHS = {
    "Al25": (Layer("Al", 25.0),),
    "Cu15": (Layer("Cu", 15.0),),
    "AlCu": (Layer("Al", 5.0), Layer("Cu", 15.0), Layer("Al", 5.0)),
    "Pb15": (Layer("Pb", 15.0),),
}


def _validation_path(path):
    if isinstance(path, str):
        try:
            return _VALIDATION_PATHS[path]
        except KeyError as exc:
            raise ValueError(
                f"path must be one of {tuple(_VALIDATION_PATHS)} or a Layer sequence"
            ) from exc
    value = tuple(path)
    if not value or not all(isinstance(item, Layer) for item in value):
        raise ValueError("path must be a non-empty sequence of Layer objects")
    return value


def _theta_ff_effective(components) -> float:
    """Z(Z+1)X/A (equivalently dchi_c2) weighted FF onset angle."""
    rows = _tail_component_tuple(components)
    if not rows:
        return np.nan
    return math.exp(
        sum(
            frac
            * math.log(HBARC_MEV_FM / (p * MEV * nuclear_radius_fm(A)))
            for frac, _, A, p in rows
        )
    )


def _log_h_grid(calibration, model: str, floor: bool, theta_max_rad: float):
    """Build h(theta) on the Task-1 grid, with an asymptotic tail guard.

    Direct finite-interval Hankel inversion eventually reaches its numerical
    cancellation floor.  Once theta is twelve transform widths into the tail,
    the compound density is single-scatter dominated; there we use the same
    screened rate whose characteristic exponent was transformed.  This avoids
    integrating Hankel roundoff as a fictitious radian-scale floor.
    """
    theta_max_rad = float(theta_max_rad)
    if theta_max_rad <= 1.0e-6:
        raise ValueError("theta_max_rad must exceed 1e-6")
    decades = math.log10(theta_max_rad / 1.0e-6)
    n = max(2, int(math.ceil(400.0 * decades)) + 1)
    theta = np.geomspace(1.0e-6, theta_max_rad, n)
    scale = math.sqrt(calibration["chi_c2"] * calibration["B"])
    switch = min(theta_max_rad, 12.0 * scale)
    core = theta <= switch
    h = np.empty_like(theta)
    h[core] = transform_radial_density(
        theta[core],
        calibration["chi_c2"],
        calibration["B"],
        calibration["tail_components"],
        model,
        include_incoherent=floor,
    )
    if np.any(~core):
        a2 = (
            calibration["chi_c2"]
            * calibration["B"]
            / (MOLIERE_SCREENING_FACTOR * math.exp(calibration["B"]))
        )
        tail_theta = theta[~core]
        G = finite_size_kernel(
            tail_theta,
            calibration["tail_components"],
            model,
            include_incoherent=floor,
        )
        h[~core] = (
            2.0
            * calibration["chi_c2"]
            * tail_theta
            * G
            / (tail_theta * tail_theta + a2) ** 2
        )
    if np.any(~np.isfinite(h)) or np.any(h < 0.0):
        raise RuntimeError("transform radial-density grid is not finite and non-negative")
    return theta, h


def _rms_untruncated_diagnostics(path, p_GeV, ff_model, floor, theta_max_rad):
    model = _normalize_ff_model(ff_model)
    if model == "none":
        raise ValueError("the point-nucleus untruncated second moment diverges")
    ordered = _validation_path(path)
    q = calibrate_pofx_transform(
        ordered,
        float(p_GeV),
        THETA_CUT,
        form_factor=model,
        include_incoherent=bool(floor),
    )
    values = {}
    for upper in (1.0, 3.0, 10.0):
        theta, h = _log_h_grid(q, model, bool(floor), upper)
        mass = float(_trapz(h, theta))
        M2 = float(_trapz(theta**2 * h, theta) / mass)
        values[upper] = (mass, M2)
    relative = abs(values[10.0][1] / values[3.0][1] - 1.0)
    assert relative < 1.0e-3, (
        "Gate A failed: M2 changed by "
        f"{relative:.6g} between 3 and 10 rad for {ff_model}, floor={bool(floor)}"
    )
    requested = float(theta_max_rad)
    if requested in values:
        mass, M2 = values[requested]
    else:
        theta, h = _log_h_grid(q, model, bool(floor), requested)
        mass = float(_trapz(h, theta))
        M2 = float(_trapz(theta**2 * h, theta) / mass)
    return dict(
        ff_model=str(ff_model),
        floor=bool(floor),
        path=path if isinstance(path, str) else "custom",
        p_GeV=float(p_GeV),
        theta_cut_mrad=np.inf,
        theta_max_rad=requested,
        mass=mass,
        M2=M2,
        M2_at_1rad=values[1.0][1],
        M2_at_3rad=values[3.0][1],
        M2_at_10rad=values[10.0][1],
        convergence_3_to_10=relative,
        theta_rms_over_theta_space=math.sqrt(M2) / q["theta_space_pofx"],
    )


def rms_untruncated(path, p_GeV, ff_model, floor, theta_max_rad=3.0):
    """Return theta_rms(theta_cut -> inf) / theta_space.

    The transform and ``q=p*theta`` both use a small-angle convention which is
    invalid near a radian.  The moment is dominated far below that scale; this
    is a convergence diagnostic, not a physical prediction at large angle.
    """
    return _rms_untruncated_diagnostics(
        path, p_GeV, ff_model, floor, theta_max_rad
    )["theta_rms_over_theta_space"]


def efficiency_scan(path, p_GeV, ff_model, floor, cuts_mrad):
    """Return Fc, M2, M4, M4/M2^2, sigma_wQ and E per requested cut."""
    model = _normalize_ff_model(ff_model)
    ordered = _validation_path(path)
    rows = []
    for cut_mrad in cuts_mrad:
        cut_mrad = float(cut_mrad)
        q = calibrate_pofx_transform(
            ordered,
            float(p_GeV),
            cut_mrad * 1.0e-3,
            form_factor=model,
            include_incoherent=bool(floor),
        )
        ratio = q["M4"] / q["M2"] ** 2
        variance = q["M4"] - q["M2"] ** 2
        efficiency = (
            math.sqrt(q["Fc"]) * q["M2"] / math.sqrt(variance)
            if variance > 0.0
            else np.nan
        )
        rows.append(
            dict(
                ff_model=str(ff_model),
                floor=bool(floor),
                path=path if isinstance(path, str) else "custom",
                p_GeV=float(p_GeV),
                theta_cut_mrad=cut_mrad,
                Fc=q["Fc"],
                M2=q["M2"],
                M4=q["M4"],
                M4_over_M2_sq=ratio,
                sigma_wQ=math.sqrt(max(ratio - 1.0, 0.0)),
                efficiency=efficiency,
            )
        )
    return rows


def composition_scan(paths, momenta, cuts_mrad, ff_model, floor):
    """Return the pre-registered composition diagnostics for every scan row."""
    model = _normalize_ff_model(ff_model)
    rows = []
    for path in paths:
        ordered = _validation_path(path)
        for p_GeV in momenta:
            for cut_mrad in cuts_mrad:
                q = calibrate_pofx_transform(
                    ordered,
                    float(p_GeV),
                    float(cut_mrad) * 1.0e-3,
                    form_factor=model,
                    include_incoherent=bool(floor),
                )
                theta_ff = _theta_ff_effective(q["tail_components"])
                rows.append(
                    dict(
                        ff_model=str(ff_model),
                        floor=bool(floor),
                        path=path if isinstance(path, str) else "custom",
                        p_GeV=float(p_GeV),
                        theta_cut_mrad=float(cut_mrad),
                        chi_c=math.sqrt(q["chi_c2"]),
                        chi_a=math.sqrt(q["chi_a2"]),
                        B=q["B"],
                        R=q["R_matched"],
                        s=math.sqrt(q["chi_c2"] * q["B"]),
                        theta_FF=theta_ff,
                        rho=theta_ff / math.sqrt(q["chi_c2"] * q["B"]),
                        eta_cut=q["eta_cut"],
                        mu2=q["mu2"],
                        eps_M=q["epsilon_matched"],
                    )
                )
    return rows


def tail_ratio_scan(path, p_GeV, ff_model, floor, thetas_mrad):
    """Return h(Theta) Theta^3 / (2 chi_c^2) on the requested angle grid."""
    model = _normalize_ff_model(ff_model)
    ordered = _validation_path(path)
    q = calibrate_pofx_transform(
        ordered,
        float(p_GeV),
        THETA_CUT,
        form_factor=model,
        include_incoherent=bool(floor),
    )
    theta = np.asarray(thetas_mrad, float) * 1.0e-3
    if np.any(theta <= 0.0):
        raise ValueError("thetas_mrad must be positive")
    h = transform_radial_density(
        theta,
        q["chi_c2"],
        q["B"],
        q["tail_components"],
        model,
        include_incoherent=bool(floor),
    )
    theta_ff = _theta_ff_effective(q["tail_components"])
    theta_nuc = HBARC_MEV_FM / (float(p_GeV) * MEV * 0.84)
    expected_floor = sum(
        frac * A / (Z * (Z + 1.0))
        for frac, Z, A, _ in _tail_component_tuple(q["tail_components"])
    )
    return [
        dict(
            ff_model=str(ff_model),
            floor=bool(floor),
            path=path if isinstance(path, str) else "custom",
            p_GeV=float(p_GeV),
            theta_cut_mrad=float(angle_mrad),
            theta_mrad=float(angle_mrad),
            h=float(density),
            tail_ratio=float(density * angle**3 / (2.0 * q["chi_c2"])),
            theta_FF_mrad=1000.0 * theta_ff,
            theta_nuc_mrad=1000.0 * theta_nuc,
            expected_floor=expected_floor if floor else 0.0,
        )
        for angle_mrad, angle, density in zip(thetas_mrad, theta, h)
    ]


def efficiency_pofx(
    path: Sequence[Layer],
    p_in: float,
    k: float,
    nmax: int = 2,
    screening_weight: str = "dchi_c2",
    form_factor: str = "none",
):
    slices, _ = slice_path(path, p_in)
    c2, _, B = accumulate_moliere_pofx(slices, screening_weight=screening_weight)
    t0, _ = highland_core_pofx_model(slices, path_x_over_x0(path))
    Fc, M2, M4 = radial_moments(
        c2,
        B,
        k * t0,
        nmax=nmax,
        tail_components=sliced_tail_components(slices),
        form_factor=form_factor,
    )
    var = M4 - M2 * M2
    return math.sqrt(Fc) * M2 / math.sqrt(var) if Fc > 0 and var > 0 else 0.0


def optimal_k_pofx(
    path: Sequence[Layer],
    p_in: float,
    nmax: int = 2,
    screening_weight: str = "dchi_c2",
    form_factor: str = "none",
):
    r = minimize_scalar(
        lambda k: (
            -efficiency_pofx(
                path,
                p_in,
                k,
                nmax=nmax,
                screening_weight=screening_weight,
                form_factor=form_factor,
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


def split_path_equal_length(path: Sequence[Layer], n_parts: int):
    """Split an ordered path into equal geometric-length intervals."""
    n_parts = int(n_parts)
    if n_parts < 1:
        raise ValueError("n_parts must be positive")
    total = sum(float(layer.thickness_cm) for layer in path)
    if total <= 0.0:
        return tuple(() for _ in range(n_parts))
    edges = np.linspace(0.0, total, n_parts + 1)
    starts = []
    x = 0.0
    for layer in path:
        starts.append((x, x + layer.thickness_cm, layer.material))
        x += layer.thickness_cm
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        layers = []
        for a, b, material in starts:
            overlap = max(0.0, min(hi, b) - max(lo, a))
            if overlap > 1e-12:
                layers.append(Layer(material, overlap))
        parts.append(tuple(layers))
    return tuple(parts)


def split_path_equal_dchi_c2(path: Sequence[Layer], p_in: float, n_parts: int):
    """Partition an ordered path into equal accumulated dchi_c^2 intervals.

    Returns ``(parts, centroids)``.  Centroids are global geometric path
    fractions evaluated with dchi_c^2 weighting inside each interval.  The
    adaptive p(X) slices provide the local scattering power; boundaries may
    therefore cut a material slice and are not constrained to equal length.
    """
    n_parts = int(n_parts)
    if n_parts < 1:
        raise ValueError("n_parts must be positive")
    slices, _ = slice_path(path, p_in)
    if not slices:
        return tuple(() for _ in range(n_parts)), np.zeros(n_parts)
    records = []
    z = 0.0
    for item in slices:
        length = float(item["thickness_cm"])
        m = MATERIALS[item["material"]]
        dc2 = chi_c2_single(
            m.Z, m.A, item["X"], item["p"] * MEV, item["beta"]
        )
        records.append((z, z + length, item["material"], dc2))
        z += length
    total_length = z
    total_c2 = sum(row[3] for row in records)
    if total_c2 <= 0.0 or total_length <= 0.0:
        return tuple(() for _ in range(n_parts)), np.zeros(n_parts)

    targets = np.linspace(0.0, total_c2, n_parts + 1)
    parts = []
    centroids = []
    for wlo, whi in zip(targets[:-1], targets[1:]):
        layers = []
        centroid_num = 0.0
        part_weight = 0.0
        accumulated = 0.0
        for zlo, zhi, material, weight in records:
            next_weight = accumulated + weight
            ov_lo = max(wlo, accumulated)
            ov_hi = min(whi, next_weight)
            if ov_hi > ov_lo + 1e-18:
                f0 = (ov_lo - accumulated) / weight
                f1 = (ov_hi - accumulated) / weight
                piece_lo = zlo + f0 * (zhi - zlo)
                piece_hi = zlo + f1 * (zhi - zlo)
                piece_length = piece_hi - piece_lo
                piece_weight = ov_hi - ov_lo
                if layers and layers[-1].material == material:
                    layers[-1] = Layer(material, layers[-1].thickness_cm + piece_length)
                else:
                    layers.append(Layer(material, piece_length))
                centroid_num += piece_weight * 0.5 * (piece_lo + piece_hi)
                part_weight += piece_weight
            accumulated = next_weight
        parts.append(tuple(layers))
        centroids.append(
            centroid_num / part_weight / total_length if part_weight > 0.0 else 0.0
        )
    return tuple(parts), np.asarray(centroids, float)


class PofxCache:
    """Cache p(X) calibration on incident-p and ordered areal-density bins.

    Ray intersections and segment order remain analytic, but the expensive
    scattering calculation uses the discretization declared in ``config.py``:
    incident momentum is rounded to ``P_CACHE_STEP``, each ordered segment's
    areal density to ``SEG_CACHE_STEP``, and the cut to ``CUT_CACHE_STEP``.
    """

    def __init__(
        self,
        nmax: int = 2,
        tol: float = P_BETA_SLICE_TOL,
        screening_weight: str = "dchi_c2",
        form_factor: str = "none",
        p_step: float = P_CACHE_STEP,
        segment_step: float = SEG_CACHE_STEP,
        cut_step: float = CUT_CACHE_STEP,
    ):
        if screening_weight not in {"dchi_c2", "serial"}:
            raise ValueError("screening_weight must be 'dchi_c2' or 'serial'")
        self.nmax = nmax
        self.tol = tol
        self.screening_weight = screening_weight
        self.form_factor = _normalize_form_factor(form_factor)
        if self.form_factor != "none" and not FINITE_SIZE_PRODUCTION_ENABLED:
            raise RuntimeError(
                "finite-size detector production is blocked; "
                "FINITE_SIZE_PRODUCTION_ENABLED is False"
            )
        self.p_step = float(p_step)
        self.segment_step = float(segment_step)
        self.cut_step = float(cut_step)
        if min(self.p_step, self.segment_step, self.cut_step) <= 0.0:
            raise ValueError("cache steps must be positive")
        self._cache = {}
        self._path_cache = {}
        self._kink_cache = {}
        self.max_clipped = 0.0
        self.local_kink_fallbacks = 0

    def _key(self, p, seg_cm, cut):
        mats = ("Al", "Cu", "Pb", "Cu", "Al")
        X = [float(t) * MATERIALS[m].rho for t, m in zip(seg_cm, mats)]
        return (
            round(float(p) / self.p_step),
            *(round(x / self.segment_step) for x in X),
            round(float(cut) / self.cut_step),
        )

    def _decode(self, key):
        p = key[0] * self.p_step
        mats = ("Al", "Cu", "Pb", "Cu", "Al")
        X = [k * self.segment_step for k in key[1:6]]
        seg = [x / MATERIALS[m].rho for x, m in zip(X, mats)]
        cut = key[6] * self.cut_step
        return p, seg, cut

    def calibration(self, p, seg_cm, cut=THETA_CUT):
        key = self._key(p, seg_cm, cut)
        if key not in self._cache:
            pp, seg, cc = self._decode(key)
            path = layers_from_segment_thicknesses(seg)
            if not path:
                self._cache[key] = None
            else:
                path_key = key[:6]
                if path_key not in self._path_cache:
                    self._path_cache[path_key] = calibrate_pofx(
                        path,
                        pp,
                        theta_cut=THETA_CUT,
                        tol=self.tol,
                        nmax=self.nmax,
                        screening_weight=self.screening_weight,
                        form_factor=self.form_factor,
                    )
                base = self._path_cache[path_key]
                Fc, M2, M4 = radial_moments(
                    base["chi_c2"],
                    base["B"],
                    cc,
                    nmax=self.nmax,
                    tail_components=base["tail_components"],
                    form_factor=self.form_factor,
                )
                trms = math.sqrt(max(M2, 0.0))
                r = dict(base)
                r.update(
                    Fc=Fc,
                    M2=M2,
                    M4=M4,
                    theta_rms=trms,
                    k_pofx=cc / base["theta0_pofx"],
                    eta_cut=cc / math.sqrt(base["chi_c2"] * base["B"]),
                    epsilon_matched=trms / base["theta_space_pofx"] - 1.0,
                    epsilon_mixed=trms / base["theta_space_incident"] - 1.0,
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
            # calibration() decodes the bucket rather than allowing the first
            # raw event to choose one side of a bin.
            _, _, cc = self._decode(key)
            r = self.calibration(p[idx[0]], segments[idx[0]], cc)
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

    def sample(self, p, segments, rng, cut=THETA_CUT):
        p = np.asarray(p, float)
        segments = np.asarray(segments, float)
        tx = np.zeros(p.size)
        ty = np.zeros(p.size)
        uniform = rng.random(p.size)
        azimuth = rng.uniform(0.0, 2.0 * math.pi, p.size)
        # Group by cache key to vectorise sampling from each radial CDF.
        keys = [self._key(p[i], segments[i], cut) for i in range(p.size)]
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        for k, idx_list in groups.items():
            idx = np.asarray(idx_list, int)
            r = self.calibration(p[idx[0]], segments[idx[0]], cut)
            if r is None:
                continue
            B, c2 = r["B"], r["chi_c2"]
            _, _, clipped = radial_cdf_eta(B, nmax=self.nmax)
            self.max_clipped = max(self.max_clipped, clipped)
            if self.form_factor == "none":
                eta = radial_eta_from_uniform(B, uniform[idx], nmax=self.nmax)
            else:
                eta = finite_size_eta_from_uniform(
                    c2, B, uniform[idx], r["tail_components"], self.form_factor
                )
            theta = math.sqrt(c2 * B) * eta
            tx[idx] = theta * np.cos(azimuth[idx])
            ty[idx] = theta * np.sin(azimuth[idx])
        return tx, ty

    def _local_kink_calibrations(self, p, seg_cm, n_kinks, cut):
        """Return local calibrations and dchi_c^2-weighted centroids."""
        key = (int(n_kinks),) + self._key(p, seg_cm, cut)
        if key in self._kink_cache:
            return self._kink_cache[key]
        pp, seg, cc = self._decode(key[1:])
        path = layers_from_segment_thicknesses(seg)
        if not path:
            self._kink_cache[key] = None
            return None
        whole = calibrate_pofx(
            path,
            pp,
            theta_cut=cc,
            tol=self.tol,
            nmax=self.nmax,
            screening_weight=self.screening_weight,
            form_factor=self.form_factor,
        )
        parts, centroids = split_path_equal_dchi_c2(path, pp, int(n_kinks))
        local = []
        p_now = pp
        for part in parts:
            try:
                r = calibrate_pofx(
                    part,
                    p_now,
                    theta_cut=cc,
                    tol=self.tol,
                    nmax=self.nmax,
                    screening_weight=self.screening_weight,
                    # Local n<=2 Moliere increments must retain their additive
                    form_factor=self.form_factor,
                )
            except ValueError:
                # Extremely short grazing paths can fall below the formal
                # Moliere-B validity threshold after subdivision.  Preserve
                # the local scattering-strength fraction but use the whole-
                # path B, and expose the count in output metadata.
                part_slices, E_part_out = slice_path(part, p_now, tol=self.tol)
                part_c2 = 0.0
                for item in part_slices:
                    mat = MATERIALS[item["material"]]
                    part_c2 += chi_c2_single(
                        mat.Z,
                        mat.A,
                        item["X"],
                        item["p"] * MEV,
                        item["beta"],
                    )
                r = dict(
                    chi_c2=part_c2,
                    B=whole["B"],
                    p_out=_p_of_E(E_part_out),
                    clipped_fraction=whole["clipped_fraction"],
                    tail_components=sliced_tail_components(part_slices),
                    local_B_fallback=True,
                )
                self.local_kink_fallbacks += 1
            local.append(r)
            p_now = r["p_out"]
        self._kink_cache[key] = (tuple(local), centroids, whole)
        return self._kink_cache[key]

    def sample_kinks(self, p, segments, rng, n_kinks, cut=THETA_CUT):
        """Sample independent local kicks along an ordered degrading path.

        Partitions have equal accumulated dchi_c^2, and each returned geometric
        node is that interval's dchi_c^2-weighted centroid.  The exit momentum
        of one partition feeds the next.
        """
        p = np.asarray(p, float)
        segments = np.asarray(segments, float)
        n_kinks = int(n_kinks)
        if n_kinks < 1:
            raise ValueError("n_kinks must be positive")
        tx = np.zeros((p.size, n_kinks))
        ty = np.zeros((p.size, n_kinks))
        uniform = rng.random((p.size, n_kinks))
        azimuth = rng.uniform(0.0, 2.0 * math.pi, (p.size, n_kinks))
        keys = [self._key(p[i], segments[i], cut) for i in range(p.size)]
        groups = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        fractions = np.zeros((p.size, n_kinks))
        for key, idx_list in groups.items():
            idx = np.asarray(idx_list, int)
            result = self._local_kink_calibrations(
                p[idx[0]], segments[idx[0]], n_kinks, cut
            )
            if result is None:
                continue
            local, centroids, whole = result
            fractions[idx, :] = centroids
            for j, r in enumerate(local):
                B, c2 = r["B"], r["chi_c2"]
                self.max_clipped = max(
                    self.max_clipped, float(r.get("clipped_fraction", 0.0))
                )
                if self.form_factor == "none":
                    eta = radial_eta_from_uniform(B, uniform[idx, j], nmax=self.nmax)
                else:
                    eta = finite_size_eta_from_uniform(
                        c2,
                        B,
                        uniform[idx, j],
                        r["tail_components"],
                        self.form_factor,
                    )
                theta = math.sqrt(c2 * B) * eta
                tx[idx, j] = theta * np.cos(azimuth[idx, j])
                ty[idx, j] = theta * np.sin(azimuth[idx, j])
        return tx, ty, fractions
