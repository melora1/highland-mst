"""Configuration for the revised Highland/Moliere tomography study.

Units are cm, GeV/c, rad unless stated otherwise.  The geometry/material
numbers match the final revision plan rather than the rounded legacy values.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

M_MU = 0.10566
M_E = 0.000510998950
ALPHA = 1.0 / 137.036


@dataclass(frozen=True)
class Material:
    Z: float
    A: float
    rho: float  # g cm^-3
    X0: float  # cm
    I_eV: float
    Cbar: float
    x0: float
    x1: float
    a: float
    k: float
    delta0: float


# Radiation lengths/densities follow the reference quantities adopted in the
# revision plan. Sternheimer constants are the same transcribed values used in
# the supplied code and must be checked against the primary PDG/LBL table before
# publication; tests only provide an indirect minimum-ionisation closure check.
MATERIALS = {
    "Al": Material(
        13.0, 26.98, 2.699, 8.896, 166.0, 4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12
    ),
    "Cu": Material(
        29.0, 63.55, 8.96, 1.436, 322.0, 4.4190, -0.0254, 3.2792, 0.14339, 2.9044, 0.08
    ),
    "Pb": Material(
        82.0, 207.2, 11.35, 0.5612, 823.0, 6.2018, 0.3776, 3.8073, 0.09359, 3.1608, 0.14
    ),
}
MAT_ORDER = ("Al", "Cu", "Pb")
PDG_MIN_DEDX = {"Al": 1.615, "Cu": 1.403, "Pb": 1.122}  # MeV cm^2 g^-1, validation only

# Target
AL_HALF = 12.5
CU_HALF = 7.5
PB_R = 2.0
PB_HALF_Z = 7.5
PB_CX = 3.0
PB_CY = 2.0

# Tracking/momentum tagger
SIGMA_HIT = 0.020
STATION_Z = np.array([-120.0, -90.0, -45.0, -15.0, 25.0, 65.0])
B_FIELD = 1.0
L_EFF = 0.30
BL = B_FIELD * L_EFF
Z_MAGNET_CM = -65.0

# Beam/exposure
MOMENTA = (1.0, 2.0, 3.5, 6.0)
MOM_BITE = 0.01
SIGMA_DIV = 2.0e-3
SIGMA_XY = 1.0
# Revised raster reaches the Al-only region while staying 1.5 cm from shell edge.
RASTER_HALF = 11.0
RASTER_NX = 9
RASTER_NY = 9
# Used by simulation.py. "per_setting" is production; "none" is retained as
# a diagnostic showing the several-cm momentum-position correlation an
# uncompensated dipole would create.
STEER_COMPENSATION = "per_setting"

# Acceptance/reconstruction
THETA_CUT = 0.200
VOX_HALF = 15.0
N_VOX = 50
VOX_SIZE = 2 * VOX_HALF / N_VOX
MIN_VOX_COUNT = 20
PB_ROI_R = 2.0
PB_ROI_ZHALF = 7.5
CU_ROI_R = 3.75
CU_ROI_ZHALF = 7.5
ROI_GUARD_GAPS_CM = (0.0, 0.6, 1.2)

# Numerical controls
P_CACHE_STEP = 0.010  # GeV/c
X_CACHE_STEP = 0.25  # g cm^-2
SEG_CACHE_STEP = 0.25  # g cm^-2, ordered-segment cache
CUT_CACHE_STEP = 0.002  # rad
P_BETA_SLICE_TOL = 0.01
RADIAL_ETA_MAX = 30.0
RADIAL_TABLE_ETA_MAX = 30.0

SEED_BASE = 20260713
SPLIT_SEED = 20260819
OUT_DIR = "out"
