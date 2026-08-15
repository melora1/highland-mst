"""Single source of truth for all constants. Units: cm, GeV/c unless noted.

Note on units: Highland (Eq. 1) and the Moliere appendix expressions use
p in MeV/c and areal density X in g cm^-2. Conversions are done explicitly
at the call sites in kinematics.py / moliere.py; everything stored in the
event table is in GeV/c, cm, rad.
"""

import numpy as np

# ---------------------------------------------------------------- physics
M_MU = 0.10566        # GeV/c^2
ALPHA = 1.0 / 137.036

# ---------------------------------------------------------------- materials
# name -> (Z, A, rho [g/cm^3], X0 [cm])   PDG / LBL values per Sec. 3.1
MATERIALS = {
    "Al": dict(Z=13.0, A=26.98, rho=2.70, X0=8.90),
    "Cu": dict(Z=29.0, A=63.55, rho=8.96, X0=1.44),
    "Pb": dict(Z=82.0, A=207.2, rho=11.35, X0=0.56),
}
MAT_ORDER = ("Al", "Cu", "Pb")

# ---------------------------------------------------------------- geometry
AL_HALF = 12.5        # cm, outer aluminium shell 25^3
CU_HALF = 7.5         # cm, copper block 15^3
PB_R = 2.0            # cm
PB_HALF_Z = 7.5       # cm  (full height 15 cm, co-extensive with Cu in z)
PB_CX = 3.0           # cm  transverse offset
PB_CY = 2.0           # cm

# ---------------------------------------------------------------- detector
SIGMA_HIT = 0.020     # cm  (200 um)
STATION_Z = np.array([-120.0, -90.0, -45.0, -15.0, +25.0, +65.0])  # cm
Z_PRE = (STATION_Z[0], STATION_Z[1])    # lever arm 30 cm
Z_POST = (STATION_Z[2], STATION_Z[3])   # lever arm 30 cm
Z_DOWN = (STATION_Z[4], STATION_Z[5])   # lever arm 40 cm

# dipole
B_FIELD = 1.0         # T
L_EFF = 0.30          # m
Z_MAGNET = -0.65      # m -> cm below; kick applied at magnet centre
Z_MAGNET_CM = -65.0   # cm  (magnet spans -80 .. -50 cm)
BL = B_FIELD * L_EFF  # T.m

# ---------------------------------------------------------------- beam
MOMENTA = (1.0, 2.0, 3.5, 6.0)   # GeV/c
N_PER_SETTING = 500_000
SIGMA_DIV = 2.0e-3    # rad angular divergence
MOM_BITE = 0.01       # fractional, Gaussian

# BEAM_MODE -- this is a SPEC DECISION, not a tuning knob. See README.
#
#   'pencil'  sigma_xy = SIGMA_XY Gaussian spot, single fixed position. This
#             is what an earlier draft of Sec. 4.1 specified. Combined with
#             the dipole, which steers the beam by 0.3*B*L/p * |Z_MAGNET_CM|
#             = 5.85/p cm at the target (5.84 cm at 1 GeV/c down to 0.97 cm
#             at 6) when STEER_COMPENSATION='none', each momentum lights a
#             ~2 cm strip at a different x with y pinned near 0. Even with
#             STEER_COMPENSATION='per_setting' (on axis), a single sigma_xy=1
#             cm spot only covers a 7.7 cm (95%) span against the 15 cm Cu
#             block -- test_beam_covers_target_face fails under this mode by
#             design; there is no tomogram to reconstruct.
#
#   'uniform' Flat illumination over UNIFORM_HALF in x and y, covering the
#             full Cu face. Solves coverage but is not how a tagged momentum
#             beam is actually operated (real facilities do not flood a
#             target uniformly with a characterized beam).
#
#   'raster'  DEFAULT. A sigma_xy=SIGMA_XY pencil beam, as in 'pencil', swept
#             over an RASTER_NX x RASTER_NY grid of center positions spanning
#             the Cu face (RASTER_HALF in x and y), with equal exposure per
#             node. This is how tagged-beam tomography facilities actually
#             cover an extended target: narrow, well-characterized spot,
#             scanned. Retains the momentum-dependent dipole steering
#             (STEER_COMPENSATION still applies on top of the raster grid),
#             so the momentum-position correlation Sec. 2.3's artifact
#             mechanism needs is whatever STEER_COMPENSATION leaves behind --
#             the raster grid itself carries no p-dependence.
#
BEAM_MODE = "raster"
SIGMA_XY = 1.0        # cm, used when BEAM_MODE in ('pencil', 'raster')
UNIFORM_HALF = 11.0   # cm, used when BEAM_MODE == 'uniform'

# Raster grid: node spacing chosen so 3*SIGMA_XY overlap between adjacent
# nodes gives near-uniform effective coverage, out to +/- RASTER_HALF, which
# comfortably spans the Cu block (CU_HALF = 7.5 cm) including corners once
# spot width is added.
RASTER_NX = 7
RASTER_NY = 7
RASTER_HALF = 7.5     # cm; node centers span [-7.5, +7.5] in x and y

# STEER_COMPENSATION -- the decisive knob for Sec. 2.3, more so than BEAM_MODE.
#
#   'none'         The dipole kick is applied and never corrected, so setting p
#                  lands 5.85/p cm off axis. This is what the code did
#                  implicitly. It manufactures a large momentum-position
#                  correlation -- which is exactly the correlation Sec. 2.3's
#                  spatially-structured artifact requires.
#
#   'per_setting'  The beamline is retuned for each momentum setting so every
#                  setting lands on the target axis, as a real tagged beamline
#                  would be operated. The residual momentum-position
#                  correlation is then only what the 1% momentum bite produces
#                  WITHIN a setting: ~0.06 cm at 1 GeV/c, not 5.85 cm.
#
# This is not a tuning choice, it is a claim about how the experiment is run,
# and Sec. 2.3's central mechanism stands or falls on it. Quantified in the
# README.
#
# NOTE (do not "fix" this default without reading tests.py first): an
# earlier patch attempt flipped this default to "per_setting" on the theory
# that "none" is unsafe as a silent default. That patch was WRONG and has
# been reverted. tests.py::test_momentum_position_correlation_exists
# explicitly validates THIS default and its own docstring says failing means
# "the configuration cannot support the paper's central claim" -- i.e. under
# 'per_setting', the repo's own pre-flight gate fails, by design, because the
# spatially-structured-artifact mechanism Sec. 2.3/4.4 claims has no basis
# once the beamline is honestly re-steered. See README.md's new section
# "IMPORTANT: tests.py validates the wrong configuration for
# results_pipeline.py" for why this default and results_pipeline.py's forced
# 'per_setting' are now DELIBERATELY inconsistent, and why that inconsistency
# itself is the actual bug that needs a human decision, not a code patch.
STEER_COMPENSATION = "none"

# ---------------------------------------------------------------- selection
THETA_CUT = 200e-3    # rad, space-angle acceptance cut

# ---------------------------------------------------------------- imaging
N_VOX = 50
VOX_HALF = 15.0       # cm, grid spans +-15 cm
VOX_SIZE = 2 * VOX_HALF / N_VOX   # 0.6 cm
PB_ROI_R = 2.0
PB_ROI_ZHALF = 7.5
CU_ROI_R = 3.75
CU_ROI_ZHALF = 7.5
MIN_VOX_COUNT = 20    # mask for artifact maps

# ---------------------------------------------------------------- moliere sampler
THETA_GRID_MAX = 400e-3   # rad; CDF support, well beyond THETA_CUT
THETA_GRID_N = 4001
# cache bucketing (coarse enough to reuse CDFs, fine enough not to bias)
P_CACHE_STEP = 0.010      # GeV/c
X_CACHE_STEP = 0.25       # g/cm^2

# ---------------------------------------------------------------- run control
SEED_BASE = 20260713
OUT_DIR = "out"