"""Highland / Lynch-Dahl projected scattering RMS (Eq. 1).

The SAME beta convention (full kinematic beta = p/sqrt(p^2+m^2)) is used
here and in moliere.py -- see Sec. 2.1. Do not change one without the other.
"""

import numpy as np

from config import M_MU


def beta_of(p):
    """p in GeV/c -> beta."""
    p = np.asarray(p, dtype=float)
    return p / np.sqrt(p * p + M_MU * M_MU)


def theta0_highland(p, x_over_X0):
    """Projected-plane Highland RMS [rad].

    p          : GeV/c
    x_over_X0  : dimensionless path length in radiation lengths

    theta0 = 13.6 MeV /(beta c p) * sqrt(x/X0) * [1 + 0.038 ln(x/(X0 beta^2))]
    """
    p = np.asarray(p, dtype=float)
    x = np.asarray(x_over_X0, dtype=float)
    b = beta_of(p)
    p_mev = p * 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        core = (13.6 / (b * p_mev)) * np.sqrt(x)
        log = 1.0 + 0.038 * np.log(x / (b * b))
        out = core * log
    return np.where(x > 0.0, out, 0.0)


def theta_space_highland(p, x_over_X0):
    """Space-angle Highland RMS = sqrt(2) * theta0  (Sec. 2.1)."""
    return np.sqrt(2.0) * theta0_highland(p, x_over_X0)
