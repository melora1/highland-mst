"""Moliere projected scattering distribution (Appendix A), truncated at n=2.

    F(theta) = 1/(chi_c sqrt(B)) * sum_{n=0..2} B^{-n} f^(n)(eta),
    eta = theta / (chi_c sqrt(B))

Units in this module follow Lynch & Dahl: p in MeV/c, X in g cm^-2,
angles in rad.

f^(1) and f^(2) are computed numerically in the PROJECTED convention from
the generating integral (see the CONVENTION block below), not transcribed
from Bethe's space-angle Table II. Results are cached to _fn_projected.npz.
"""

import os

from math import factorial

import numpy as np
from scipy.integrate import quad

# numpy>=2.0 renamed trapz -> trapezoid; keep both working
_trapz = getattr(np, "trapezoid", None) or np.trapz

from config import (ALPHA, MAT_ORDER, MATERIALS, P_CACHE_STEP, THETA_GRID_MAX,
                    THETA_GRID_N, X_CACHE_STEP)
from kinematics import beta_of

##############################################################################
# CONVENTION -- READ BEFORE TOUCHING f1/f2
#
# Bethe (1953) Table II tabulates f^(n) for the SPACE-angle distribution,
#     P(Theta) Theta dTheta = [f0 + f1/B + f2/B^2] eta d eta,
# normalised as int_0^inf [...] eta d eta = 1, for which f0(eta)=2exp(-eta^2)
# and f1(0)=0.8456, f2(0)=2.4929.
#
# Appendix A of the paper uses the PROJECTED (1-D) normalisation, f0(eta) =
# exp(-eta^2)/sqrt(pi), int_-inf^inf = 1. Splicing Bethe's space-angle table
# into a projected formula mixes conventions; the map between them is an
# integral transform, not a rescaling, so no constant factor repairs it.
#
# Fix: compute the projected f^(n) directly from the generating integral.
# Replacing the Hankel kernel J0 with the Fourier kernel cos projects it:
#
#   f_p^(n)(eta) = (1/pi) Int_0^inf cos(eta u) exp(-u^2/4)
#                          [ (u^2/4) ln(u^2/4) ]^n / n!  du
#
# Self-check at n=0: (1/pi) sqrt(pi) exp(-eta^2) = exp(-eta^2)/sqrt(pi),
# exactly Appendix A. Verified to 9 digits in tests.py.
#
# Asymptote: f1 -> 1/(2 eta^3) (NOT eta^-4 -- that is the space-angle power).
# Substituting back gives F(theta) -> chi_c^2 / (2 theta^3): B cancels and the
# tail is fixed absolutely by chi_c^2, i.e. by the number of scatterers. That
# is the Rutherford single-scattering limit and is the strongest available
# check on this module -- see tests.py::test_single_scatter_limit.
##############################################################################

_ETA_MAX = 60.0                       # beyond this, use the analytic asymptote
_F1_ASYMPTOTE_C = 0.5                 # f1 -> C / eta^3

# nonuniform grid: dense in the core, log-spaced through the tail
_ETA_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 10.0, 501),
    np.geomspace(10.0, _ETA_MAX, 200),
]))

_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_fn_projected.npz")


def _fp_quad(n, eta):
    """Projected f^(n) at a single eta, by oscillatory quadrature.

    epsabs is 1e-13, not smaller: the integral is O(0.5), so demanding 1e-14
    is below what double precision can deliver and QUADPACK correctly reports
    roundoff. At 1e-13 the n=0 case still reproduces exp(-eta^2)/sqrt(pi) to
    2.2e-16 (machine epsilon) across the whole grid, so nothing is lost --
    the tolerance was over-demanding, not the integrator inaccurate.
    """
    def g(u):
        if u <= 0.0:
            return 0.0
        a = 0.25 * u * u
        return np.exp(-a) * (a * np.log(a)) ** n / factorial(n)
    if eta == 0.0:
        v, _ = quad(g, 0.0, 30.0, limit=800, epsabs=1e-13, epsrel=1e-11)
    else:
        v, _ = quad(g, 0.0, 30.0, weight="cos", wvar=eta,
                    limit=800, epsabs=1e-13, epsrel=1e-11)
    return v / np.pi


def _build_tables():
    f1 = np.array([_fp_quad(1, e) for e in _ETA_GRID])
    f2 = np.array([_fp_quad(2, e) for e in _ETA_GRID])
    return f1, f2


if os.path.exists(_CACHE_FILE):
    _z = np.load(_CACHE_FILE)
    if _z["eta"].shape == _ETA_GRID.shape and np.allclose(_z["eta"], _ETA_GRID):
        _F1_TAB, _F2_TAB = _z["f1"], _z["f2"]
    else:
        _F1_TAB, _F2_TAB = _build_tables()
        np.savez_compressed(_CACHE_FILE, eta=_ETA_GRID, f1=_F1_TAB, f2=_F2_TAB)
else:
    _F1_TAB, _F2_TAB = _build_tables()
    np.savez_compressed(_CACHE_FILE, eta=_ETA_GRID, f1=_F1_TAB, f2=_F2_TAB)


def f0(eta):
    return np.exp(-eta * eta) / np.sqrt(np.pi)


def f1(eta):
    e = np.abs(eta)
    out = np.interp(e, _ETA_GRID, _F1_TAB)
    tail = e > _ETA_MAX
    return np.where(tail, _F1_ASYMPTOTE_C / np.maximum(e, 1e-9) ** 3, out)


def f2(eta):
    e = np.abs(eta)
    out = np.interp(e, _ETA_GRID, _F2_TAB)
    tail = e > _ETA_MAX
    # f2 is already ~1e-8 at eta=60 and is suppressed by a further 1/B;
    # continuity-match a 1/eta^4 falloff rather than extend the grid.
    c2 = _F2_TAB[-1] * _ETA_MAX ** 4
    return np.where(tail, c2 / np.maximum(e, 1e-9) ** 4, out)


# ------------------------------------------------------------------ chi_c, chi_a, B

def chi_c2_single(Z, A, X, p_mev, beta):
    """Eq. (A2). X in g cm^-2, p in MeV/c."""
    return 0.157 * Z * (Z + 1.0) * X / A / (p_mev * beta) ** 2


def chi_a2_single(Z, p_mev, beta):
    """Eq. (A3). p in MeV/c. Projectile charge z = 1 (muon)."""
    return (2.007e-5 * Z ** (2.0 / 3.0)
            * (1.0 + 3.34 * (Z * ALPHA / beta) ** 2) / p_mev ** 2)


def combine_path(X_al, X_cu, X_pb, p_gev):
    """Multi-material combination (Appendix A). Scalars in, scalars out.

    Returns (chi_c2, chi_a2). X_i in g cm^-2, p in GeV/c.
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
        w = Z * (Z + 1.0) * X / A                     # Eq. (A6) weight
        num += w * np.log(chi_a2_single(Z, p_mev, beta))
        den += w
    if den == 0.0:
        return 0.0, 1.0
    chi_a2 = float(np.exp(num / den))
    return float(chi_c2), chi_a2


def solve_B(chi_c2, chi_a2, tol=1e-10, itmax=60):
    """B - ln B = ln(chi_c^2 / (1.167 chi_a^2))  [Eq. (A5)], Newton.

    The errstate suppresses the divide-by-zero that fires when chi_c2=0
    (zero areal density after bucketing). Callers that reach here with
    chi_c2=0 are bugs -- the non-finite rhs check immediately below will
    raise a clean ValueError rather than letting a RuntimeWarning propagate.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        rhs = np.log(chi_c2 / (1.167 * chi_a2))
    if not np.isfinite(rhs):
        raise ValueError("non-finite Omega_0 in solve_B")
    if rhs < 1.0:
        # B - ln B >= 1 always; Omega_0 < e means fewer than ~20 scatters.
        # Should not happen for the x/X0 range here -- fail loudly.
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


# ------------------------------------------------------------------ pdf / cdf
_THETA_GRID = np.linspace(-THETA_GRID_MAX, THETA_GRID_MAX, THETA_GRID_N)


def pdf_on_grid(chi_c2, B, nmax=2):
    """Truncated Moliere pdf on _THETA_GRID. Returns (pdf, clipped_fraction)."""
    chi_c = np.sqrt(chi_c2)
    scale = chi_c * np.sqrt(B)
    eta = _THETA_GRID / scale

    F = f0(eta)
    if nmax >= 1:
        F = F + f1(eta) / B
    if nmax >= 2:
        F = F + f2(eta) / B ** 2
    F = F / scale

    raw_norm = _trapz(F, _THETA_GRID)
    neg = np.clip(-F, 0.0, None)
    clipped = _trapz(neg, _THETA_GRID) / max(raw_norm, 1e-30)

    F = np.clip(F, 0.0, None)
    norm = _trapz(F, _THETA_GRID)
    return F / norm, float(clipped)


def cdf_on_grid(chi_c2, B, nmax=2):
    pdf, clipped = pdf_on_grid(chi_c2, B, nmax=nmax)
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (pdf[1:] + pdf[:-1]) * np.diff(_THETA_GRID))])
    cdf /= cdf[-1]
    # enforce strict monotonicity for interp inversion
    cdf = np.maximum.accumulate(cdf + 1e-15 * np.arange(cdf.size))
    cdf /= cdf[-1]
    return cdf, clipped


# ------------------------------------------------------------------ cached sampler
class MoliereSampler:
    """Caches inverse-CDFs bucketed on (p, X_Al, X_Cu, X_Pb).

    Bucket widths are set in config (P_CACHE_STEP, X_CACHE_STEP). Rebuilding
    a CDF per event for 2e6 events is the throughput bottleneck; bucketing
    reduces it to O(10^3-10^4) unique keys.

    Zero-material events (all three areal densities round to bucket 0) are
    cached as None and returned as (0, 0) by sample(). They carry no
    scattering and their weight is zero in all downstream estimators.
    """

    def __init__(self, nmax=2):
        self.nmax = nmax
        self._cache = {}
        self.max_clipped = 0.0

    def _key(self, p, X_al, X_cu, X_pb):
        return (round(p / P_CACHE_STEP),
                round(X_al / X_CACHE_STEP),
                round(X_cu / X_CACHE_STEP),
                round(X_pb / X_CACHE_STEP))

    def _get(self, key):
        if key in self._cache:
            return self._cache[key]
        p = key[0] * P_CACHE_STEP
        X_al = key[1] * X_CACHE_STEP
        X_cu = key[2] * X_CACHE_STEP
        X_pb = key[3] * X_CACHE_STEP
        chi_c2, chi_a2 = combine_path(X_al, X_cu, X_pb, p)
        # Zero-material guard: combine_path returns chi_c2=0 when all
        # areal densities are zero (raster-edge muons that miss the target).
        # Moliere theory is undefined there; store None as sentinel and let
        # sample() return (0, 0) so these events get weight 0 downstream.
        if chi_c2 == 0.0:
            self._cache[key] = None
            return None
        B = solve_B(chi_c2, chi_a2)
        cdf, clipped = cdf_on_grid(chi_c2, B, nmax=self.nmax)
        self.max_clipped = max(self.max_clipped, clipped)
        self._cache[key] = cdf
        return cdf

    def sample(self, p, X_al, X_cu, X_pb, rng):
        """Vectorised: arrays in, (theta_x, theta_y) arrays out [rad]."""
        p = np.asarray(p, float)
        n = p.size
        keys = [self._key(p[i], X_al[i], X_cu[i], X_pb[i]) for i in range(n)]

        # group by key so each CDF is built once and inverted in bulk
        order = np.argsort([hash(k) for k in keys])
        tx = np.empty(n)
        ty = np.empty(n)
        i = 0
        ks = [keys[j] for j in order]
        while i < n:
            j = i
            while j < n and ks[j] == ks[i]:
                j += 1
            idx = order[i:j]
            cdf = self._get(ks[i])
            if cdf is None:
                # zero-material event: no scattering
                tx[idx] = 0.0
                ty[idx] = 0.0
                i = j
                continue
            u1 = rng.random(idx.size)
            u2 = rng.random(idx.size)
            tx[idx] = np.interp(u1, cdf, _THETA_GRID)
            ty[idx] = np.interp(u2, cdf, _THETA_GRID)
            i = j
        return tx, ty