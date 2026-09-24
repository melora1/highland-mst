"""Legacy two-coordinate inverse-CDF cache for finite-size sampling.

The cache coordinates are ``x=1/B`` and ``y=ln(rho)``.  Its stored random
variable is the untruncated reduced radius ``eta=theta/s``.  A physical cut is
applied without another cache dimension by evaluating ``F(eta_cut)`` and then
mapping a uniform variate to ``u*F(eta_cut)`` before inverse-CDF lookup.

The corrected kernel has a separate atomic-electron cutoff, whose reduced
coordinate is ``rho_e=(m_e/m_mu)/s``.  Therefore ``(B,rho)`` is not sufficient;
new production caches must add at least that coordinate (and still validate
material-mixture dependence).  Loading superseded caches and building new 2D
production tables are deliberately blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.special import j1

from physics import (
    FINITE_SIZE_KERNEL_VERSION,
    HBARC_MEV_FM,
    MEV,
    MOLIERE_SCREENING_FACTOR,
    THETA_E_MAX,
    _finite_size_characteristic_table,
    _normalize_form_factor,
    finite_size_kernel,
    nuclear_radius_fm,
    untruncated_finite_size_moments,
)


# By eta=12 the compound distribution is single-scatter dominated.  The tail
# must retain its absolute survival probability; extending the oscillatory
# Hankel CDF farther only makes that small probability round to one.
HANKEL_ETA_JOIN = 12.0
CACHE_SCHEMA_VERSION = "reduced-rhoe-v1"


def default_u_grid():
    """Probability nodes with explicit resolution in the far upper tail."""
    bulk = np.linspace(0.0, 0.995, 1991)
    tail = 1.0 - np.geomspace(5.0e-3, 1.0e-10, 1200)
    return np.unique(np.concatenate((bulk, tail)))


def default_eta_grid(eta_max):
    """Reduced-radius work grid used during precomputation."""
    eta_max = float(eta_max)
    if eta_max <= HANKEL_ETA_JOIN:
        raise ValueError("eta_max must exceed the transform/tail join")
    return np.unique(np.concatenate((
        np.array([0.0]),
        np.geomspace(1.0e-6, HANKEL_ETA_JOIN, 2600),
        np.geomspace(HANKEL_ETA_JOIN, eta_max, 1800),
    )))


def _canonical_components(rho):
    """Pure-Cu fallback used when no empirical node template is supplied."""
    Z, A = 29.0, 63.55
    p_reduced = HBARC_MEV_FM / (MEV * nuclear_radius_fm(A) * float(rho))
    return ((1.0, Z, A, p_reduced),)


def _node_cdf(B, rho, eta_grid, model, floor, components, rho_e=None):
    """Build one untruncated reduced CDF.

    The Hankel transform is used through ``HANKEL_ETA_JOIN``. Beyond that point the
    compound distribution is single-scatter dominated; its survival function
    is continued with the same screened finite-size kernel and normalized to
    the transform at the join.  This avoids undersampling highly oscillatory
    Bessel functions for grazing-path eta values of O(10^3).
    """
    B = float(B)
    rho = float(rho)
    eta_grid = np.asarray(eta_grid, float)
    if eta_grid.ndim != 1 or eta_grid[0] != 0.0 or np.any(np.diff(eta_grid) <= 0.0):
        raise ValueError("eta_grid must be strictly increasing and start at zero")
    c2 = 1.0 / B  # s=sqrt(c2*B)=1 in reduced coordinates
    components = tuple(tuple(map(float, row)) for row in components)
    transform_args = () if rho_e is None else ("step", float(rho_e))
    t, characteristic = _finite_size_characteristic_table(
        c2, B, components, model, bool(floor), *transform_args
    )
    join = HANKEL_ETA_JOIN
    core_mask = eta_grid <= join
    core_eta = eta_grid[core_mask]
    cdf = np.empty_like(eta_grid)
    cdf[0] = 0.0
    for start in range(1, core_eta.size, 128):
        stop = min(start + 128, core_eta.size)
        e = core_eta[start:stop]
        cdf[start:stop] = e * simpson(
            characteristic[None, :] * j1(e[:, None] * t[None, :]), x=t, axis=1
        )
    cdf[:core_eta.size] = np.maximum.accumulate(
        np.clip(cdf[:core_eta.size], 0.0, 1.0)
    )
    if core_eta[-1] < join:
        raise ValueError("eta_grid must contain the transform/tail join")

    tail_eta = eta_grid[core_eta.size - 1:]
    a2 = 1.0 / (MOLIERE_SCREENING_FACTOR * math.exp(B))
    kernel_args = () if rho_e is None else ("step", float(rho_e))
    G = finite_size_kernel(
        tail_eta, components, model, bool(floor), *kernel_args
    )
    rate = (2.0 / B) * tail_eta * G / (tail_eta * tail_eta + a2) ** 2
    reverse = cumulative_trapezoid(rate[::-1], tail_eta[::-1], initial=0.0)
    survival_shape = -reverse[::-1]
    tail_n2 = float(np.trapezoid(tail_eta**2 * rate, tail_eta))
    core_cdf = cdf[:core_eta.size].copy()
    core_mass = float(core_cdf[-1])
    core_n2 = float(
        join**2 * core_mass
        - np.trapezoid(2.0 * core_eta * core_cdf, core_eta)
    )
    full_n2, _ = untruncated_finite_size_moments(
        c2, B, components, model, include_incoherent=bool(floor),
        electron_theta_max=(THETA_E_MAX if rho_e is None else float(rho_e)),
    )
    tail_mass = max(float(survival_shape[0]), 0.0)
    denominator = tail_n2 - tail_mass * core_n2 / core_mass
    tail_factor = (
        (full_n2 - core_n2 / core_mass) / denominator
        if core_mass > 0.0 and denominator > 0.0
        else 1.0
    )
    # Compound tails approach the single-scatter shape but can still carry a
    # few-percent convolution correction at the join.  Fix its normalization
    # from the exact untruncated second cumulant, which is available without a
    # Hankel inversion.
    tail_factor = max(float(tail_factor), 0.0)
    survival_shape *= tail_factor
    join_survival = max(float(survival_shape[0]), 0.0)
    if join_survival <= np.finfo(float).tiny:
        cdf[core_eta.size - 1:] = 1.0
    else:
        # At large eta the probability can round out of the oscillatory
        # Hankel CDF even though its eta^2-weighted contribution remains
        # material.  The screened single-scatter rate is absolute (not just a
        # tail shape), so anchor the core to its survival at the join rather
        # than multiplying it by the numerically rounded ``1-CDF(join)``.
        target_join_cdf = max(1.0 - join_survival, 0.0)
        core_join_cdf = cdf[core_eta.size - 1]
        if core_join_cdf > 0.0:
            cdf[:core_eta.size] *= target_join_cdf / core_join_cdf
        cdf[core_eta.size - 1:] = 1.0 - survival_shape
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    cdf[-1] = 1.0
    return cdf


def _node_inverse(args):
    B, rho, eta_grid, model, floor, components, u_grid = args
    cdf = _node_cdf(B, rho, eta_grid, model, floor, components)
    keep = np.concatenate(([True], np.diff(cdf) > 0.0))
    return np.interp(u_grid, cdf[keep], eta_grid[keep])


def _node_inverse_3d(args):
    B, rho, rho_e, eta_grid, model, floor, components, u_grid = args
    cdf = _node_cdf(
        B, rho, eta_grid, model, floor, components, rho_e=float(rho_e)
    )
    keep = np.concatenate(([True], np.diff(cdf) > 0.0))
    return np.interp(u_grid, cdf[keep], eta_grid[keep])


def build_reduced_cache(
    B_grid,
    rho_grid,
    eta_grid,
    ff_model,
    floor,
    *,
    templates=None,
    u_grid=None,
    workers=1,
    allow_invalid_2d=False,
):
    """Return inverse-CDF nodes ``CDF^-1(u; B, rho)`` on a parameter grid.

    ``templates[i,j]`` may provide the empirical material-component tuple for
    a node.  The component momenta must already be in reduced units (physical
    momentum multiplied by the template path's ``s``) and scaled so the node's
    effective onset is ``rho_grid[j]``.  Without templates a pure-Cu kernel is
    used; production validation supplies empirical templates.
    """
    if not allow_invalid_2d:
        raise RuntimeError(
            "the corrected finite-size kernel is not a two-parameter (B, rho) "
            "family: the electron cutoff adds rho_e=(m_e/m_mu)/s. Extend the "
            "cache axes or use a validated direct-sampler fallback."
        )
    model = _normalize_form_factor(ff_model)
    if model == "none":
        raise ValueError("the reduced finite-size cache requires a form factor")
    B_grid = np.asarray(B_grid, float)
    rho_grid = np.asarray(rho_grid, float)
    eta_grid = np.asarray(eta_grid, float)
    u_grid = default_u_grid() if u_grid is None else np.asarray(u_grid, float)
    if np.any(B_grid <= 0.0) or np.any(rho_grid <= 0.0):
        raise ValueError("B and rho grids must be positive")
    if np.any(np.diff(B_grid) <= 0.0) or np.any(np.diff(rho_grid) <= 0.0):
        raise ValueError("B and rho grids must be strictly increasing")
    if u_grid[0] != 0.0 or np.any(np.diff(u_grid) <= 0.0) or u_grid[-1] >= 1.0:
        raise ValueError("u_grid must increase from zero to a value below one")
    if templates is not None and np.shape(templates) != (len(B_grid), len(rho_grid)):
        raise ValueError("templates must have shape (len(B_grid), len(rho_grid))")

    # Float32 quantisation is O(1e-7) relative on this domain, comfortably
    # below the 3.5e-4 physics gate, and halves the resident production table.
    inverse = np.empty((len(B_grid), len(rho_grid), len(u_grid)), np.float32)
    jobs = []
    for i, B in enumerate(B_grid):
        for j, rho in enumerate(rho_grid):
            components = (
                _canonical_components(rho) if templates is None else templates[i, j]
            )
            jobs.append((
                float(B), float(rho), eta_grid, model, bool(floor),
                components, u_grid,
            ))
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            values = list(pool.map(_node_inverse, jobs, chunksize=1))
    else:
        values = list(map(_node_inverse, jobs))
    inverse[:] = np.asarray(values, dtype=np.float32).reshape(inverse.shape)
    return ReducedSamplerCache(
        B_grid=B_grid,
        rho_grid=rho_grid,
        u_grid=u_grid,
        inverse=inverse,
        ff_model=model,
        floor=bool(floor),
    )


def build_reduced_cache_3d(
    B_grid,
    rho_grid,
    rho_e_grid,
    eta_grid,
    ff_model,
    floor,
    *,
    u_grid=None,
    workers=1,
):
    """Build the universal ``(B, rho, rho_e)`` candidate cache.

    This constructor is intentionally separate from the blocked legacy 2D
    builder.  A production caller must still pass the real-path heterogeneity
    gate; the three reduced coordinates do not encode elemental composition.
    """
    model = _normalize_form_factor(ff_model)
    if model == "none":
        raise ValueError("the reduced finite-size cache requires a form factor")
    B_grid = np.asarray(B_grid, float)
    rho_grid = np.asarray(rho_grid, float)
    rho_e_grid = np.asarray(rho_e_grid, float)
    eta_grid = np.asarray(eta_grid, float)
    u_grid = default_u_grid() if u_grid is None else np.asarray(u_grid, float)
    for name, grid in (("B", B_grid), ("rho", rho_grid), ("rho_e", rho_e_grid)):
        if len(grid) < 2 or np.any(grid <= 0.0) or np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"{name}_grid must contain at least two increasing positives")
    inverse = np.empty(
        (len(B_grid), len(rho_grid), len(rho_e_grid), len(u_grid)), np.float32
    )
    jobs = [
        (float(B), float(rho), float(rho_e), eta_grid, model, bool(floor),
         _canonical_components(rho), u_grid)
        for B in B_grid for rho in rho_grid for rho_e in rho_e_grid
    ]
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            values = list(pool.map(_node_inverse_3d, jobs, chunksize=1))
    else:
        values = list(map(_node_inverse_3d, jobs))
    inverse[:] = np.asarray(values, np.float32).reshape(inverse.shape)
    return ReducedSamplerCache3D(
        B_grid, rho_grid, rho_e_grid, u_grid, inverse, model, bool(floor)
    )


@dataclass
class ReducedSamplerCache:
    B_grid: np.ndarray
    rho_grid: np.ndarray
    u_grid: np.ndarray
    inverse: np.ndarray
    ff_model: str
    floor: bool
    kernel_version: str = FINITE_SIZE_KERNEL_VERSION

    def __post_init__(self):
        if str(self.kernel_version) != FINITE_SIZE_KERNEL_VERSION:
            raise RuntimeError(
                "reduced cache uses finite-size kernel "
                f"{self.kernel_version!r}; expected {FINITE_SIZE_KERNEL_VERSION!r}. "
                "Rebuild the cache before finite-size production."
            )
        self.B_grid = np.asarray(self.B_grid, float)
        self.rho_grid = np.asarray(self.rho_grid, float)
        self.u_grid = np.asarray(self.u_grid, float)
        self.inverse = np.asarray(self.inverse, dtype=np.float32)
        expected = (len(self.B_grid), len(self.rho_grid), len(self.u_grid))
        if self.inverse.shape != expected:
            raise ValueError(f"inverse table has shape {self.inverse.shape}, expected {expected}")
        self._x_grid = 1.0 / self.B_grid[::-1]
        self._y_grid = np.log(self.rho_grid)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            B_grid=self.B_grid,
            rho_grid=self.rho_grid,
            u_grid=self.u_grid,
            inverse=self.inverse,
            ff_model=np.asarray(self.ff_model),
            floor=np.asarray(self.floor),
            kernel_version=np.asarray(self.kernel_version),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if "kernel_version" not in data.files:
                raise RuntimeError(
                    "reduced cache predates the corrected proton/electron "
                    "finite-size kernel; rebuild it before use"
                )
            return cls(
                B_grid=data["B_grid"], rho_grid=data["rho_grid"],
                u_grid=data["u_grid"], inverse=data["inverse"],
                ff_model=str(data["ff_model"]), floor=bool(data["floor"]),
                kernel_version=str(data["kernel_version"]),
            )

    def _cell(self, B, rho):
        B = float(B)
        rho = float(rho)
        if not (self.B_grid[0] <= B <= self.B_grid[-1]):
            raise RuntimeError(
                f"reduced-cache B={B:.9g} outside "
                f"[{self.B_grid[0]:.9g}, {self.B_grid[-1]:.9g}]"
            )
        if not (self.rho_grid[0] <= rho <= self.rho_grid[-1]):
            raise RuntimeError(
                f"reduced-cache rho={rho:.9g} outside "
                f"[{self.rho_grid[0]:.9g}, {self.rho_grid[-1]:.9g}]"
            )
        x = 1.0 / B
        y = math.log(rho)
        ix = min(max(np.searchsorted(self._x_grid, x) - 1, 0), len(self._x_grid) - 2)
        iy = min(max(np.searchsorted(self._y_grid, y) - 1, 0), len(self._y_grid) - 2)
        tx = (x - self._x_grid[ix]) / (self._x_grid[ix + 1] - self._x_grid[ix])
        ty = (y - self._y_grid[iy]) / (self._y_grid[iy + 1] - self._y_grid[iy])
        # x uses reversed B ordering.
        ib0 = len(self.B_grid) - 1 - ix
        ib1 = ib0 - 1
        return ib0, ib1, iy, float(tx), float(ty)

    def _corners(self, B, rho):
        ib0, ib1, ir, tx, ty = self._cell(B, rho)
        tables = (
            self.inverse[ib0, ir], self.inverse[ib1, ir],
            self.inverse[ib0, ir + 1], self.inverse[ib1, ir + 1],
        )
        weights = (
            (1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
            (1.0 - tx) * ty, tx * ty,
        )
        return tables, weights

    def cdf_at(self, B, rho, eta):
        tables, weights = self._corners(B, rho)
        # Interpolation is defined on inverse-CDF nodes.  Invert that same
        # bilinearly interpolated curve here; averaging four independently
        # inverted corner CDFs is a different distribution and can let the
        # subsequent truncated draw exceed eta_cut.
        interpolated_inverse = sum(w * q for q, w in zip(tables, weights))
        return float(np.interp(
            float(eta), interpolated_inverse, self.u_grid, left=0.0, right=1.0
        ))

    def inverse_at(self, B, rho, probability):
        p = np.asarray(probability, float)
        if np.any((p < 0.0) | (p >= 1.0)):
            raise ValueError("probabilities must lie in [0,1)")
        tables, weights = self._corners(B, rho)
        return sum(
            w * np.interp(p, self.u_grid, q)
            for q, w in zip(tables, weights)
        )

    def sample_eta(self, B, rho, eta_cut, uniform):
        """Sample the reduced radius conditional on ``eta <= eta_cut``."""
        Fc = self.cdf_at(B, rho, eta_cut)
        if not (0.0 < Fc <= 1.0):
            raise RuntimeError(f"invalid reduced-cache acceptance CDF {Fc}")
        u = np.asarray(uniform, float)
        return self.inverse_at(B, rho, np.minimum(u * Fc, self.u_grid[-1]))

    def truncated_moment(self, B, rho, eta_cut, power):
        """Reduced accepted radial moment for validation/calibration.

        Integrating on the cache's tail-resolved probability grid is both more
        accurate and faster than fixed-order Gauss--Legendre quadrature when
        the accepted CDF is within O(1e-8) of unity.
        """
        Fc = self.cdf_at(B, rho, eta_cut)
        probabilities = np.unique(np.concatenate((
            self.u_grid[self.u_grid < Fc], np.asarray([Fc])
        )))
        eta = self.inverse_at(
            B, rho, np.minimum(probabilities, self.u_grid[-1])
        )
        return float(np.trapezoid(eta ** int(power), probabilities) / Fc)

    def truncated_m2(self, B, rho, eta_cut, order=None):
        return self.truncated_moment(B, rho, eta_cut, 2)


@dataclass
class ReducedSamplerCache3D:
    """Trilinear inverse-CDF table in ``(1/B, ln rho, ln rho_e)``."""

    B_grid: np.ndarray
    rho_grid: np.ndarray
    rho_e_grid: np.ndarray
    u_grid: np.ndarray
    inverse: np.ndarray
    ff_model: str
    floor: bool
    kernel_version: str = FINITE_SIZE_KERNEL_VERSION
    cache_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self):
        if str(self.kernel_version) != FINITE_SIZE_KERNEL_VERSION:
            raise RuntimeError(
                f"3D cache kernel {self.kernel_version!r} does not match "
                f"{FINITE_SIZE_KERNEL_VERSION!r}"
            )
        if str(self.cache_version) != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"3D cache schema {self.cache_version!r} does not match "
                f"{CACHE_SCHEMA_VERSION!r}"
            )
        self.B_grid = np.asarray(self.B_grid, float)
        self.rho_grid = np.asarray(self.rho_grid, float)
        self.rho_e_grid = np.asarray(self.rho_e_grid, float)
        self.u_grid = np.asarray(self.u_grid, float)
        self.inverse = np.asarray(self.inverse, np.float32)
        expected = (
            len(self.B_grid), len(self.rho_grid), len(self.rho_e_grid),
            len(self.u_grid),
        )
        if self.inverse.shape != expected:
            raise ValueError(
                f"inverse table has shape {self.inverse.shape}, expected {expected}"
            )
        for name, grid in (("B", self.B_grid), ("rho", self.rho_grid),
                           ("rho_e", self.rho_e_grid), ("u", self.u_grid)):
            if np.any(np.diff(grid) <= 0.0):
                raise ValueError(f"{name} grid must be strictly increasing")
        self._x_grid = 1.0 / self.B_grid[::-1]
        self._y_grid = np.log(self.rho_grid)
        self._z_grid = np.log(self.rho_e_grid)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, B_grid=self.B_grid, rho_grid=self.rho_grid,
            rho_e_grid=self.rho_e_grid, u_grid=self.u_grid,
            inverse=self.inverse, ff_model=np.asarray(self.ff_model),
            floor=np.asarray(self.floor), kernel_version=np.asarray(self.kernel_version),
            cache_version=np.asarray(self.cache_version),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            required = {"rho_e_grid", "kernel_version", "cache_version"}
            missing = required.difference(data.files)
            if missing:
                raise RuntimeError(
                    "cache is not a current 3D rho_e table; missing "
                    + ", ".join(sorted(missing))
                )
            return cls(
                data["B_grid"], data["rho_grid"], data["rho_e_grid"],
                data["u_grid"], data["inverse"], str(data["ff_model"]),
                bool(data["floor"]), str(data["kernel_version"]),
                str(data["cache_version"]),
            )

    @staticmethod
    def _bracket(grid, value, label):
        value = float(value)
        if not (grid[0] <= value <= grid[-1]):
            raise RuntimeError(
                f"3D reduced-cache {label}={value:.9g} outside "
                f"[{grid[0]:.9g}, {grid[-1]:.9g}]"
            )
        i = min(max(np.searchsorted(grid, value)-1, 0), len(grid)-2)
        t = (value-grid[i])/(grid[i+1]-grid[i])
        return int(i), float(t)

    def _corners(self, B, rho, rho_e):
        ix, tx = self._bracket(self._x_grid, 1.0/float(B), "1/B")
        iy, ty = self._bracket(self._y_grid, math.log(float(rho)), "ln(rho)")
        iz, tz = self._bracket(self._z_grid, math.log(float(rho_e)), "ln(rho_e)")
        ib = (len(self.B_grid)-1-ix, len(self.B_grid)-2-ix)
        tables, weights = [], []
        for xb, wx in ((0, 1-tx), (1, tx)):
            for yb, wy in ((0, 1-ty), (1, ty)):
                for zb, wz in ((0, 1-tz), (1, tz)):
                    tables.append(self.inverse[ib[xb], iy+yb, iz+zb])
                    weights.append(wx*wy*wz)
        return tables, weights

    def _inverse_curve(self, B, rho, rho_e):
        tables, weights = self._corners(B, rho, rho_e)
        return sum(w*q for q, w in zip(tables, weights))

    def cdf_at(self, B, rho, rho_e, eta):
        curve = self._inverse_curve(B, rho, rho_e)
        return float(np.interp(float(eta), curve, self.u_grid, left=0.0, right=1.0))

    def inverse_at(self, B, rho, rho_e, probability):
        p = np.asarray(probability, float)
        if np.any((p < 0.0) | (p >= 1.0)):
            raise ValueError("probabilities must lie in [0,1)")
        return np.interp(p, self.u_grid, self._inverse_curve(B, rho, rho_e))

    def sample_eta(self, B, rho, rho_e, eta_cut, uniform):
        Fc = self.cdf_at(B, rho, rho_e, eta_cut)
        if not (0.0 < Fc <= 1.0):
            raise RuntimeError(f"invalid 3D reduced-cache acceptance CDF {Fc}")
        u = np.asarray(uniform, float)
        return self.inverse_at(
            B, rho, rho_e, np.minimum(u*Fc, self.u_grid[-1])
        )

    def truncated_moment(self, B, rho, rho_e, eta_cut, power):
        Fc = self.cdf_at(B, rho, rho_e, eta_cut)
        probabilities = np.unique(np.concatenate((
            self.u_grid[self.u_grid < Fc], np.asarray([Fc])
        )))
        eta = self.inverse_at(
            B, rho, rho_e, np.minimum(probabilities, self.u_grid[-1])
        )
        return float(np.trapezoid(eta**int(power), probabilities)/Fc)
