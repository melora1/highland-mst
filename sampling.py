"""Accepted-angle inverse-CDF sampler for the finite-size transform."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from config import THETA_CUT
from physics import (
    _normalize_ff_model,
    _validation_path,
    calibrate_pofx_transform,
    transform_radial_density,
)


class TransformSampler:
    """Sample angles conditional on ``theta <= theta_cut``.

    ``p_profile`` may be a scalar incident momentum or a one-dimensional array;
    array entries define an equally weighted momentum mixture.  The CDF is
    normalized at the cut, so every returned angle is accepted by construction.
    """

    def __init__(self, path, p_profile, ff_model, floor, theta_cut):
        self.path = path
        self.ff_model = str(ff_model)
        self.floor = bool(floor)
        self.theta_cut = float(theta_cut)
        if not (0.0 < self.theta_cut <= 10.0):
            raise ValueError("theta_cut must lie in (0, 10] rad")
        self._model = _normalize_ff_model(ff_model)
        self._path = _validation_path(path)
        momenta = np.atleast_1d(np.asarray(p_profile, float))
        if momenta.ndim != 1 or momenta.size == 0 or np.any(momenta <= 0.0):
            raise ValueError("p_profile must contain positive incident momenta")
        self.p_profile = momenta.copy()

        decades = math.log10(self.theta_cut / 1.0e-6)
        n = max(1201, int(math.ceil(600.0 * max(decades, 0.0))) + 1)
        positive = np.geomspace(1.0e-6, self.theta_cut, n)
        theta = np.concatenate(([0.0], positive))
        h = np.zeros_like(theta)
        references = []
        for p_GeV in momenta:
            q = calibrate_pofx_transform(
                self._path,
                float(p_GeV),
                self.theta_cut,
                form_factor=self._model,
                include_incoherent=self.floor,
            )
            references.append(q)
            h[1:] += transform_radial_density(
                positive,
                q["chi_c2"],
                q["B"],
                q["tail_components"],
                self._model,
                include_incoherent=self.floor,
            ) / momenta.size
        if np.any(~np.isfinite(h)) or np.any(h < 0.0):
            raise RuntimeError("sampler density is not finite and non-negative")

        raw_cdf = cumulative_trapezoid(h, theta, initial=0.0)
        if raw_cdf[-1] <= 0.0:
            raise RuntimeError("sampler CDF has zero mass at theta_cut")
        cdf = raw_cdf / raw_cdf[-1]
        # Log-grid trapezoids close to zero can underflow to equal CDF values.
        # Drop those nodes before PCHIP and assert strict monotonicity explicitly.
        keep = np.concatenate(([True], np.diff(cdf) > 0.0))
        theta_nodes = theta[keep]
        cdf_nodes = cdf[keep]
        if cdf_nodes[0] != 0.0:
            theta_nodes = np.concatenate(([0.0], theta_nodes))
            cdf_nodes = np.concatenate(([0.0], cdf_nodes))
        cdf_nodes[-1] = 1.0
        assert np.all(np.diff(cdf_nodes) > 0.0), "inverse-CDF nodes are not strict"
        self.theta_nodes = theta_nodes
        self.cdf_nodes = cdf_nodes
        self._inverse = PchipInterpolator(cdf_nodes, theta_nodes, extrapolate=False)
        self._cdf = PchipInterpolator(theta_nodes, cdf_nodes, extrapolate=False)
        self.Fc = float(np.mean([q["Fc"] for q in references]))
        accepted_weights = np.asarray([q["Fc"] for q in references], float)
        accepted_weights /= accepted_weights.sum()
        self.M2 = float(
            sum(w * q["M2"] for w, q in zip(accepted_weights, references))
        )
        self.M4 = float(
            sum(w * q["M4"] for w, q in zip(accepted_weights, references))
        )

    def sample(self, n, rng):
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        values = np.asarray(self._inverse(rng.random(n)), float)
        assert np.all(values <= self.theta_cut), "accepted sampler exceeded theta_cut"
        return values

    def cdf(self, theta):
        """Analytic interpolation of the normalized accepted-angle CDF."""
        values = np.asarray(theta, float)
        return np.where(
            values <= 0.0,
            0.0,
            np.where(values >= self.theta_cut, 1.0, self._cdf(values)),
        )
