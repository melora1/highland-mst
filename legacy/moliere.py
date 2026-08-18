"""Moliere scattering model, truncated at n <= 2.

Primary (manuscript) convention
-------------------------------
The tomography moments and event sampler use the rotationally symmetric
TWO-DIMENSIONAL angular density

    P_M(Theta) = 1/(2*pi*s^2) * sum_{n=0..2} B^{-n} Phi^(n)(eta),
    eta = Theta/s,   s = chi_c*sqrt(B),

with the Hankel-generating functions

    Phi^(n)(eta) = 1/n! * integral_0^inf
        u J0(eta*u) exp(-u^2/4)
        [(u^2/4) ln(u^2/4)]^n du.

The normalized magnitude density is h(Theta)=2*pi*Theta*P_M(Theta).  This is
the probability measure used for acceptance fractions and M_2/M_4.

Projected diagnostic convention
-------------------------------
The signed projected marginal F(theta_plane) is also retained for Appendix-A
cross-checks.  Its correction functions f_p^(n) use the cosine generating
integral.  They MUST NOT be multiplied as F(theta_x)F(theta_y) to construct
the hard-scatter two-dimensional distribution.

Units follow Lynch & Dahl: p is converted to MeV/c inside the material
parameters, X is g cm^-2, and angles are radians.
"""

import os
import warnings
from functools import lru_cache
from math import factorial

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import j0

# numpy>=2.0 renamed trapz -> trapezoid; keep both working
_trapz = getattr(np, "trapezoid", None) or np.trapz
_cumtrap = None
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrap
except ImportError:  # pragma: no cover - old SciPy fallback
    pass

from config import (
    ALPHA,
    MAT_ORDER,
    MATERIALS,
    P_CACHE_STEP,
    THETA_GRID_MAX,
    THETA_GRID_N,
    X_CACHE_STEP,
)
from .kinematics import beta_of


# ============================================================================
# Projected marginal -- diagnostic only
# ============================================================================

_ETA_MAX = 60.0
_F1_ASYMPTOTE_C = 0.5  # f_p^(1) -> 1/(2 |eta|^3)
_PROJ_ETA_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 10.0, 501),
            np.geomspace(10.0, _ETA_MAX, 200),
        ]
    )
)
_PROJ_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_fn_projected.npz"
)


def _fp_quad(n, eta):
    """Projected f_p^(n)(eta), computed from the cosine generating integral."""

    def g(u):
        if u <= 0.0:
            return 0.0
        a = 0.25 * u * u
        return np.exp(-a) * (a * np.log(a)) ** n / factorial(n)

    if eta == 0.0:
        v, _ = quad(g, 0.0, 30.0, limit=800, epsabs=1e-13, epsrel=1e-11)
    else:
        v, _ = quad(
            g, 0.0, 30.0, weight="cos", wvar=eta, limit=800, epsabs=1e-13, epsrel=1e-11
        )
    return v / np.pi


def _build_projected_tables():
    return (
        np.array([_fp_quad(1, e) for e in _PROJ_ETA_GRID]),
        np.array([_fp_quad(2, e) for e in _PROJ_ETA_GRID]),
    )


def _load_projected_tables():
    try:
        if os.path.exists(_PROJ_CACHE_FILE):
            z = np.load(_PROJ_CACHE_FILE)
            if z["eta"].shape == _PROJ_ETA_GRID.shape and np.allclose(
                z["eta"], _PROJ_ETA_GRID
            ):
                return z["f1"], z["f2"]
    except Exception:
        pass
    f1_tab, f2_tab = _build_projected_tables()
    try:
        np.savez_compressed(_PROJ_CACHE_FILE, eta=_PROJ_ETA_GRID, f1=f1_tab, f2=f2_tab)
    except OSError:
        pass
    return f1_tab, f2_tab


_F1_TAB, _F2_TAB = _load_projected_tables()


def f0(eta):
    """Projected Gaussian term f_p^(0)."""
    eta = np.asarray(eta, dtype=float)
    return np.exp(-eta * eta) / np.sqrt(np.pi)


def f1(eta):
    """Projected correction f_p^(1); diagnostic marginal only."""
    e = np.abs(np.asarray(eta, dtype=float))
    out = np.interp(e, _PROJ_ETA_GRID, _F1_TAB)
    return np.where(e > _ETA_MAX, _F1_ASYMPTOTE_C / np.maximum(e, 1e-12) ** 3, out)


def f2(eta):
    """Projected correction f_p^(2); diagnostic marginal only."""
    e = np.abs(np.asarray(eta, dtype=float))
    out = np.interp(e, _PROJ_ETA_GRID, _F2_TAB)
    c2 = _F2_TAB[-1] * _ETA_MAX**4
    return np.where(e > _ETA_MAX, c2 / np.maximum(e, 1e-12) ** 4, out)


# ============================================================================
# Radial two-dimensional Moliere expansion -- primary physics path
# ============================================================================

# Direct Hankel quadrature is stable through eta~30.  Beyond that, the n=1
# Rutherford term is already in its analytic Phi1 -> 2/eta^4 regime and
# dominates the n=2 correction by orders of magnitude.
_RADIAL_ETA_MAX = 30.0
_RADIAL_TABLE_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 10.0, 1001),
            np.geomspace(10.0, _RADIAL_ETA_MAX, 500),
        ]
    )
)
_RADIAL_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_fn_radial.npz"
)

# Dense integration grid used for CDFs and moments.  The direct-Hankel table
# is interpolated onto this grid; no 2-D Cartesian factorization is involved.
_RADIAL_INT_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 10.0, 5001),
            np.geomspace(10.0, _RADIAL_ETA_MAX, 1500),
        ]
    )
)


def _phi_quad(n, eta):
    """Radial Phi^(n)(eta) from the manuscript's Hankel-J0 integral."""

    def integrand(u):
        if u <= 0.0:
            return 0.0
        a = 0.25 * u * u
        return u * j0(eta * u) * np.exp(-a) * (a * np.log(a)) ** n / factorial(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        v, _ = quad(integrand, 0.0, 30.0, limit=1000, epsabs=1e-12, epsrel=1e-10)
    return v


def _build_radial_tables():
    phi1_tab = np.array([_phi_quad(1, e) for e in _RADIAL_TABLE_GRID])
    phi2_tab = np.array([_phi_quad(2, e) for e in _RADIAL_TABLE_GRID])
    return phi1_tab, phi2_tab


def _load_radial_tables():
    try:
        if os.path.exists(_RADIAL_CACHE_FILE):
            z = np.load(_RADIAL_CACHE_FILE)
            if z["eta"].shape == _RADIAL_TABLE_GRID.shape and np.allclose(
                z["eta"], _RADIAL_TABLE_GRID
            ):
                return z["phi1"], z["phi2"]
    except Exception:
        pass
    p1, p2 = _build_radial_tables()
    try:
        np.savez_compressed(
            _RADIAL_CACHE_FILE, eta=_RADIAL_TABLE_GRID, phi1=p1, phi2=p2
        )
    except OSError:
        pass
    return p1, p2


_PHI1_TAB, _PHI2_TAB = _load_radial_tables()
_PHI2_TAIL_C = _PHI2_TAB[-1] * _RADIAL_ETA_MAX**6


def phi0(eta):
    """Radial Gaussian term Phi^(0)=2 exp(-eta^2)."""
    e = np.asarray(eta, dtype=float)
    return 2.0 * np.exp(-e * e)


def phi1(eta):
    """Radial correction Phi^(1), with Rutherford asymptote 2/eta^4."""
    e = np.abs(np.asarray(eta, dtype=float))
    out = np.interp(e, _RADIAL_TABLE_GRID, _PHI1_TAB)
    tail = 2.0 / np.maximum(e, 1e-12) ** 4
    return np.where(e > _RADIAL_ETA_MAX, tail, out)


def phi2(eta):
    """Radial correction Phi^(2).

    The n=2 term is negligible in the far tail compared with Phi1/B.  A
    continuity-matched eta^-6 continuation is used beyond the direct-Hankel
    table; the Rutherford normalization is controlled by Phi1.
    """
    e = np.abs(np.asarray(eta, dtype=float))
    out = np.interp(e, _RADIAL_TABLE_GRID, _PHI2_TAB)
    tail = _PHI2_TAIL_C / np.maximum(e, 1e-12) ** 6
    return np.where(e > _RADIAL_ETA_MAX, tail, out)


def radial_series_eta(eta, B, nmax=2, clip=False):
    """Dimensionless radial series G(eta)=sum B^-n Phi^(n)(eta)."""
    eta = np.asarray(eta, dtype=float)
    G = phi0(eta)
    if nmax >= 1:
        G = G + phi1(eta) / B
    if nmax >= 2:
        G = G + phi2(eta) / (B * B)
    return np.clip(G, 0.0, None) if clip else G


def radial_density(theta, chi_c2, B, nmax=2, clip=True):
    """Two-dimensional density P_M(Theta) per unit angular area."""
    theta = np.asarray(theta, dtype=float)
    if chi_c2 <= 0.0:
        return np.zeros_like(theta)
    s = np.sqrt(chi_c2 * B)
    G = radial_series_eta(theta / s, B, nmax=nmax, clip=clip)
    return G / (2.0 * np.pi * s * s)


def radial_magnitude_density(theta, chi_c2, B, nmax=2, normalize=True):
    """Magnitude density h(Theta)=2*pi*Theta*P_M(Theta)."""
    theta = np.asarray(theta, dtype=float)
    h = 2.0 * np.pi * theta * radial_density(theta, chi_c2, B, nmax=nmax, clip=True)
    if normalize:
        norm, _ = radial_total_mass(B, nmax=nmax)
        h = h / norm
    return h


def _cumulative_trapezoid(y, x):
    if _cumtrap is not None:
        return _cumtrap(y, x, initial=0.0)
    out = np.empty_like(y)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _radial_grid_terms(B, nmax=2):
    eta = _RADIAL_INT_GRID
    raw_G = radial_series_eta(eta, B, nmax=nmax, clip=False)
    pos_G = np.clip(raw_G, 0.0, None)
    q_raw = eta * raw_G
    q_pos = eta * pos_G
    q_neg = eta * np.clip(-raw_G, 0.0, None)
    return eta, q_raw, q_pos, q_neg


@lru_cache(maxsize=4096)
def _radial_mass_cached(B_key, nmax):
    B = B_key / 1e6
    eta, q_raw, q_pos, q_neg = _radial_grid_terms(B, nmax=nmax)
    mass_grid = float(_trapz(q_pos, eta))
    raw_grid = float(_trapz(q_raw, eta))
    neg_grid = float(_trapz(q_neg, eta))
    # For n>=1 the far tail is Rutherford: q(eta) -> 2/(B eta^3),
    # whose integral from eta_max to infinity is 1/(B eta_max^2).
    tail_mass = (1.0 / (B * _RADIAL_ETA_MAX**2)) if nmax >= 1 else 0.0
    total = mass_grid + tail_mass
    raw_total = raw_grid + tail_mass
    clipped_fraction = neg_grid / max(raw_total, 1e-30)
    return total, clipped_fraction


def radial_total_mass(B, nmax=2):
    """Return positive-series normalization and clipped-negative fraction."""
    return _radial_mass_cached(round(float(B) * 1e6), int(nmax))


def radial_moments(chi_c2, B, cut, nmax=2):
    """Acceptance fraction and conditional M2/M4 from the RADIAL distribution.

    Returns (F_c, M2, M4), where M2 and M4 are in rad^2 and rad^4.
    No product F(theta_x)F(theta_y) is used.
    """
    if chi_c2 <= 0.0 or cut <= 0.0:
        return 0.0, 0.0, 0.0
    s = np.sqrt(chi_c2 * B)
    eta_cut = float(cut / s)
    total_mass, _ = radial_total_mass(B, nmax=nmax)

    # Use the common dense grid for eta_cut within the direct-Hankel range.
    if eta_cut <= _RADIAL_ETA_MAX:
        eta = _RADIAL_INT_GRID
        j = int(np.searchsorted(eta, eta_cut, side="right"))
        eg = eta[:j]
        if eg.size == 0 or eg[-1] < eta_cut:
            eg = np.append(eg, eta_cut)
        G = radial_series_eta(eg, B, nmax=nmax, clip=True)
        mass = float(_trapz(eg * G, eg))
        n2 = float(_trapz(eg**3 * G, eg))
        n4 = float(_trapz(eg**5 * G, eg))
    else:
        # Integrate tabulated region, then use the Rutherford Phi1 tail.
        eg = _RADIAL_INT_GRID
        G = radial_series_eta(eg, B, nmax=nmax, clip=True)
        e0 = _RADIAL_ETA_MAX
        mass = float(_trapz(eg * G, eg))
        n2 = float(_trapz(eg**3 * G, eg))
        n4 = float(_trapz(eg**5 * G, eg))
        if nmax >= 1:
            # q=2/(B eta^3)
            mass += (1.0 / B) * (1.0 / e0**2 - 1.0 / eta_cut**2)
            n2 += (2.0 / B) * np.log(eta_cut / e0)
            n4 += (1.0 / B) * (eta_cut**2 - e0**2)

    if mass <= 0.0:
        return 0.0, 0.0, 0.0
    Fc = min(max(mass / total_mass, 0.0), 1.0)
    M2 = s * s * (n2 / mass)
    M4 = s**4 * (n4 / mass)
    return Fc, M2, M4


def radial_cdf_eta(B, nmax=2):
    """CDF of eta=Theta/s on [0, eta_max], retaining analytic tail mass.

    The returned CDF ends below 1 when nmax>=1; the remaining probability is
    sampled analytically from the Rutherford eta^-3 magnitude tail.
    """
    eta, _, q_pos, _ = _radial_grid_terms(B, nmax=nmax)
    cumulative = _cumulative_trapezoid(q_pos, eta)
    total, clipped = radial_total_mass(B, nmax=nmax)
    cdf = cumulative / total
    cdf = np.maximum.accumulate(cdf)
    return eta, cdf, clipped


# ============================================================================
# Material/path parameters
# ============================================================================


def chi_c2_single(Z, A, X, p_mev, beta):
    """Manuscript Eq. (chi_c). X in g cm^-2, p in MeV/c."""
    return 0.157 * Z * (Z + 1.0) * X / A / (p_mev * beta) ** 2


def chi_a2_single(Z, p_mev, beta):
    """Manuscript Eq. (chi_a). p in MeV/c; projectile charge z=1."""
    return (
        2.007e-5 * Z ** (2.0 / 3.0) * (1.0 + 3.34 * (Z * ALPHA / beta) ** 2) / p_mev**2
    )


def combine_path(X_al, X_cu, X_pb, p_gev):
    """Constant-momentum serial multi-material combination.

    This implements the manuscript's common-p approximation.  It must not be
    interpreted as an energy-loss-aware p(X) calculation.
    """
    beta = float(beta_of(p_gev))
    p_mev = p_gev * 1e3

    chi_c2 = 0.0
    num = 0.0
    den = 0.0
    for name, X in zip(MAT_ORDER, (X_al, X_cu, X_pb)):
        if X <= 0.0:
            continue
        Z = MATERIALS[name]["Z"]
        A = MATERIALS[name]["A"]
        chi_c2 += chi_c2_single(Z, A, X, p_mev, beta)
        w = Z * (Z + 1.0) * X / A
        num += w * np.log(chi_a2_single(Z, p_mev, beta))
        den += w
    if den == 0.0:
        return 0.0, 1.0
    return float(chi_c2), float(np.exp(num / den))


def solve_B(chi_c2, chi_a2, tol=1e-10, itmax=60):
    """Solve B-ln(B)=ln[chi_c^2/(1.167 chi_a^2)] for B>1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs = np.log(chi_c2 / (1.167 * chi_a2))
    if not np.isfinite(rhs):
        raise ValueError("non-finite Omega_0 in solve_B")
    if rhs < 1.0:
        raise ValueError(f"Omega_0 too small (rhs={rhs:.3f}); Moliere invalid")
    B = max(rhs + np.log(max(rhs, 1.001)), 1.5)
    for _ in range(itmax):
        g = B - np.log(B) - rhs
        dg = 1.0 - 1.0 / B
        step = g / dg
        B -= step
        if abs(step) < tol:
            break
    else:
        raise RuntimeError("solve_B failed to converge")
    if B <= 1.0:
        raise ValueError(f"solve_B returned B={B} <= 1")
    return float(B)


# ============================================================================
# Projected-marginal grid API retained for diagnostics/backward compatibility
# ============================================================================

_THETA_GRID = np.linspace(-THETA_GRID_MAX, THETA_GRID_MAX, THETA_GRID_N)


def pdf_on_grid(chi_c2, B, nmax=2):
    """Projected marginal F(theta) on _THETA_GRID; DIAGNOSTIC ONLY."""
    if chi_c2 <= 0.0:
        return np.zeros_like(_THETA_GRID), 0.0
    s = np.sqrt(chi_c2 * B)
    eta = _THETA_GRID / s
    F = f0(eta)
    if nmax >= 1:
        F = F + f1(eta) / B
    if nmax >= 2:
        F = F + f2(eta) / B**2
    F = F / s
    raw_norm = _trapz(F, _THETA_GRID)
    neg = np.clip(-F, 0.0, None)
    clipped = _trapz(neg, _THETA_GRID) / max(raw_norm, 1e-30)
    F = np.clip(F, 0.0, None)
    norm = _trapz(F, _THETA_GRID)
    return F / norm, float(clipped)


def cdf_on_grid(chi_c2, B, nmax=2):
    """Projected marginal CDF; retained only for appendix diagnostics."""
    pdf, clipped = pdf_on_grid(chi_c2, B, nmax=nmax)
    cdf = _cumulative_trapezoid(pdf, _THETA_GRID)
    cdf /= cdf[-1]
    cdf = np.maximum.accumulate(cdf + 1e-15 * np.arange(cdf.size))
    cdf /= cdf[-1]
    return cdf, clipped


# ============================================================================
# Radial event sampler
# ============================================================================


class MoliereSampler:
    """Sample the radial 2-D Moliere distribution, then a uniform azimuth.

    This is deliberately NOT two independent draws from the projected
    marginal.  A single hard Rutherford deflection couples theta_x and theta_y;
    radial sampling preserves that correlation.
    """

    def __init__(self, nmax=2):
        self.nmax = int(nmax)
        self._cache = {}
        self.max_clipped = 0.0

    def _key(self, p, X_al, X_cu, X_pb):
        return (
            round(p / P_CACHE_STEP),
            round(X_al / X_CACHE_STEP),
            round(X_cu / X_CACHE_STEP),
            round(X_pb / X_CACHE_STEP),
        )

    def _get(self, key):
        if key in self._cache:
            return self._cache[key]
        p = key[0] * P_CACHE_STEP
        X_al = key[1] * X_CACHE_STEP
        X_cu = key[2] * X_CACHE_STEP
        X_pb = key[3] * X_CACHE_STEP

        # Step 1, Section B: draw from the energy-loss-aware accumulation
        # rather than constant-p combine_path.  chi_c2 accumulates local
        # 0.157*Z(Z+1)*(X_j/A_j)/(p_j*beta_j)^2 over slices of the degraded
        # profile p(X); at constant p this is identical to combine_path (see
        # test_pofx.test_constant_p_limit).  Local import: energy_loss.py
        # imports this module at module scope, so importing it there would
        # be circular.
        from .energy_loss import accumulate_moliere, ordered_path, slice_path

        t_al = X_al / MATERIALS["Al"]["rho"]
        t_cu = X_cu / MATERIALS["Cu"]["rho"]
        t_pb = X_pb / MATERIALS["Pb"]["rho"]
        if t_al + t_cu + t_pb <= 0.0:
            self._cache[key] = None
            return None
        slices, _ = slice_path(ordered_path(t_al, t_cu, t_pb), p)
        chi_c2, chi_a2, B = accumulate_moliere(slices)
        if chi_c2 == 0.0:
            self._cache[key] = None
            return None
        eta_grid, cdf, clipped = radial_cdf_eta(B, nmax=self.nmax)
        self.max_clipped = max(self.max_clipped, clipped)
        item = (np.sqrt(chi_c2 * B), B, eta_grid, cdf)
        self._cache[key] = item
        return item

    def sample(self, p, X_al, X_cu, X_pb, rng):
        """Vectorized arrays in; return correlated (theta_x, theta_y) in rad."""
        p = np.asarray(p, dtype=float)
        X_al = np.asarray(X_al, dtype=float)
        X_cu = np.asarray(X_cu, dtype=float)
        X_pb = np.asarray(X_pb, dtype=float)
        n = p.size
        keys = [self._key(p[i], X_al[i], X_cu[i], X_pb[i]) for i in range(n)]
        order = np.argsort([hash(k) for k in keys])
        tx = np.empty(n)
        ty = np.empty(n)
        i = 0
        ordered_keys = [keys[j] for j in order]

        while i < n:
            j = i + 1
            while j < n and ordered_keys[j] == ordered_keys[i]:
                j += 1
            idx = order[i:j]
            item = self._get(ordered_keys[i])
            if item is None:
                tx[idx] = 0.0
                ty[idx] = 0.0
                i = j
                continue

            scale, B, eta_grid, cdf = item
            u = rng.random(idx.size)
            eta = np.empty(idx.size)
            cdf_end = float(cdf[-1])
            core = u <= cdf_end
            if np.any(core):
                eta[core] = np.interp(u[core], cdf, eta_grid)
            if np.any(~core):
                # Conditional Rutherford tail q(eta) proportional eta^-3
                # for eta >= eta0: CDF = 1-(eta0/eta)^2.
                r = (u[~core] - cdf_end) / max(1.0 - cdf_end, 1e-30)
                eta[~core] = _RADIAL_ETA_MAX / np.sqrt(
                    np.maximum(1.0 - r, np.finfo(float).tiny)
                )

            theta = scale * eta
            az = rng.uniform(0.0, 2.0 * np.pi, idx.size)
            tx[idx] = theta * np.cos(az)
            ty[idx] = theta * np.sin(az)
            i = j

        return tx, ty
