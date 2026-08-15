"""Pre-flight validation for the p(X) energy-loss treatment (Step 1).

    python3 test_pofx.py

Same runner convention as tests.py.  Replaces the first version, which had
two tests that compared an exact calculation against a bucketed one and
failed for that reason rather than for a physics reason.

NOTE ON BUCKETING.  eps_quadrature.eps_M / theta_RMS and the vectorised
eps_quadrature_pofx entry points quantize the areal density to
config.X_CACHE_STEP (0.25 g/cm^2).  The axial reference path has
X_Cu = 8.96*15 = 134.4 g/cm^2, which rounds UP to 134.5, so the bucketed
numerator sees 0.28 mm of extra copper against an unrounded denominator.
Comparisons between exact and bucketed calculations must either snap the
inputs onto the grid or account for that offset.  Two tests below do the
former; two more pin the offset so it cannot drift unnoticed.
"""
import math

import numpy as np

import energy_loss as el
import moliere as ml
import stopping
from config import MATERIALS, MOMENTA, THETA_CUT, X_CACHE_STEP
from eps_quadrature import eps_M as eps_M_constp, optimal_k
from eps_quadrature_pofx import eps_M_mixed, eps_M_pofx, theta_RMS_pofx
from kinematics import theta0_highland, theta_space_highland

AXIAL = (10.0, 15.0, 0.0)     # t_Al, t_Cu, t_Pb  in cm


def _areal(t):
    return (MATERIALS["Al"]["rho"] * t[0],
            MATERIALS["Cu"]["rho"] * t[1],
            MATERIALS["Pb"]["rho"] * t[2])


def _snap(X):
    """Round an areal density onto the cache grid, as the bucketed API does."""
    return float(np.rint(X / X_CACHE_STEP) * X_CACHE_STEP)


def _snapped(t):
    """(thicknesses, areal densities) landing exactly on the bucket grid."""
    X = tuple(_snap(x) for x in _areal(t))
    return (X[0] / MATERIALS["Al"]["rho"],
            X[1] / MATERIALS["Cu"]["rho"],
            X[2] / MATERIALS["Pb"]["rho"]), X


def _eps_M_exact(p, X_al, X_cu, X_pb, cut=THETA_CUT):
    """Unbucketed eps_M from the repo's own primitives."""
    c2, a2 = ml.combine_path(float(X_al), float(X_cu), float(X_pb), float(p))
    B = ml.solve_B(c2, a2)
    _, M2, _ = ml.radial_moments(c2, B, float(cut), nmax=2)
    xx0 = (X_al / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
           + X_cu / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"]
           + X_pb / MATERIALS["Pb"]["rho"] / MATERIALS["Pb"]["X0"])
    return math.sqrt(M2) / float(theta_space_highland(p, xx0)) - 1.0


# --------------------------------------------------------- stopping power
def test_stopping_power_matches_pdg_minima():
    """Indirect validation of the Sternheimer constants: the computed minimum
    mass stopping power must reproduce the published minima to 1%."""
    for name, (got, want, rel, bg) in stopping.validate().items():
        assert abs(rel) < 0.01, (name, got, want, rel)
        assert 2.5 < bg < 4.0, (name, bg)


def test_range_table_matches_rk4():
    """The CSDA range-table propagator must agree with direct RK4 integration
    of dE/dX = -S(E) to better than 1e-4 relative in p_out."""
    for material, X in (("Al", 27.0), ("Cu", 134.4), ("Pb", 100.0)):
        for p_in in (1.0, 6.0):
            E = math.hypot(p_in, el.M_MU)
            n = 20000
            h = X / n
            f = lambda Ei: -stopping.dedx_of_E(Ei, material) * 1e-3
            for _ in range(n):
                k1 = f(E); k2 = f(E + 0.5 * h * k1)
                k3 = f(E + 0.5 * h * k2); k4 = f(E + h * k3)
                E += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            p_rk4 = math.sqrt(E * E - el.M_MU ** 2)
            p_tab = math.sqrt(el.energy_after(math.hypot(p_in, el.M_MU),
                                              material, X) ** 2 - el.M_MU ** 2)
            assert abs(p_tab / p_rk4 - 1.0) < 1e-4, (material, p_in,
                                                     p_tab, p_rk4)


# --------------------------------------------------------- reduction
def test_constant_p_limit():
    """With the stopping power switched off, the segmented varying-p
    accumulation must reproduce Eq. (A14) EXACTLY.  This is the Sec. II F
    reduction requirement and is what makes the construction legitimate
    rather than an ad hoc scattering-power model."""
    real = el.energy_after
    try:
        el.energy_after = lambda E, m, X: E
        for p in (1.0, 6.0):
            r = el.calibrate(*AXIAL, p)
            assert abs(r["eps_M"] - r["eps_M_0"]) < 1e-9, (
                p, r["eps_M"], r["eps_M_0"])
            assert abs(r["th_space"] / r["th_space_0"] - 1.0) < 1e-12
            assert abs(r["chi_c2"] / r["chi_c2_0"] - 1.0) < 1e-12
    finally:
        el.energy_after = real


def test_constp_baseline_matches_unbucketed_repo_primitives():
    """calibrate()'s constant-p baseline must equal the repo's own
    combine_path + radial_moments + Eq. (1) evaluated WITHOUT the areal-
    density cache.  This implementation match is what lets the reported p(X)
    shift be a physics difference rather than a code difference."""
    X = _areal(AXIAL)
    for p in MOMENTA:
        mine = el.calibrate(*AXIAL, p)["eps_M_0"]
        repo = _eps_M_exact(p, *X)
        assert abs(mine - repo) < 1e-12, (p, mine, repo)


def test_constp_baseline_matches_bucketed_api_on_grid():
    """On inputs lying exactly on the X_CACHE_STEP grid, the bucketed
    eps_quadrature.eps_M must agree with the exact calculation to machine
    precision.  Failure here would mean the cache does something beyond
    rounding."""
    t_snap, X_snap = _snapped(AXIAL)
    for p in MOMENTA:
        mine = el.calibrate(*t_snap, p)["eps_M_0"]
        repo = float(eps_M_constp(p, *X_snap)[0])
        assert abs(mine - repo) < 1e-12, (p, mine, repo)


def test_areal_density_bucketing_bias_is_bounded_and_positive():
    """Pin the Table I quantization offset.

    Published Table I is high by roughly +0.03 to +0.04 pp purely because
    X_Cu = 134.4 rounds to 134.5 on the cache grid.  Small on its own, but
    about 27% of the 6 GeV/c p(X) shift, so it must be removed before the
    two are compared.  This test does not assert the bias is acceptable; it
    asserts we know its size, so it cannot change silently.
    """
    X = _areal(AXIAL)
    assert abs(_snap(X[1]) - X[1]) > 1e-9, (
        "X_Cu now lands on the cache grid; this test's premise is gone")
    for p in MOMENTA:
        bias = 100.0 * (float(eps_M_constp(p, *X)[0]) - _eps_M_exact(p, *X))
        assert 0.02 < bias < 0.06, (p, bias)


def test_optimal_k_insensitive_to_bucketing():
    """k_opt must not inherit the areal-density offset: efficiency() calls
    combine_path on the raw X, and bucketing shifts the cut and the width
    together.  Verify rather than assume, since Table I quotes k_opt."""
    X = _areal(AXIAL)
    _, X_snap = _snapped(AXIAL)
    for p in MOMENTA:
        k_raw = float(optimal_k(p, *X)[0])
        k_snap = float(optimal_k(p, *X_snap)[0])
        assert abs(k_raw - k_snap) < 5e-3, (p, k_raw, k_snap)


# --------------------------------------------------------- slicing
def test_slicing_converged_at_one_percent():
    """Tightening the (p beta) slicing criterion 10x must not move eps_M by
    more than 0.001 percentage points."""
    base = el.calibrate(*AXIAL, 1.0, tol=0.01)["eps_M"]
    fine = el.calibrate(*AXIAL, 1.0, tol=0.001)["eps_M"]
    assert abs(fine - base) < 1e-5, (base, fine)


def test_slice_count_scales_with_loss():
    """Low momentum must need more slices than high; ~30 at 1 GeV/c on the
    axial path is the revision plan's own estimate."""
    n1 = el.calibrate(*AXIAL, 1.0)["n_slices"]
    n6 = el.calibrate(*AXIAL, 6.0)["n_slices"]
    assert n1 > n6, (n1, n6)
    assert 15 <= n1 <= 60, n1


def test_moliere_still_valid_on_degraded_profile():
    """Omega_0 >= 20 is Moliere's own validity condition (App. A)."""
    for p in MOMENTA:
        assert el.calibrate(*AXIAL, p)["Omega0"] > 20.0


def test_ordered_path_preserves_totals():
    """Imposing traversal order must not change the material budget."""
    path = el.ordered_path(10.0, 15.0, 3.0)
    tot = {}
    for n, t in path:
        tot[n] = tot.get(n, 0.0) + t
    for name, want in (("Al", 10.0), ("Cu", 15.0), ("Pb", 3.0)):
        assert abs(tot[name] - want) < 1e-12, (name, tot[name])
    xx0 = (10.0 / MATERIALS["Al"]["X0"] + 15.0 / MATERIALS["Cu"]["X0"]
           + 3.0 / MATERIALS["Pb"]["X0"])
    assert abs(el.x_over_X0(path) - xx0) < 1e-12


# --------------------------------------------------------- physics
def test_energy_loss_lowers_eps_M_and_most_at_low_p():
    """Mechanism: energy loss inflates chi_c, shrinking the reduced acceptance
    at a FIXED 200 mrad cut, so the self-consistent eps_M must fall, and fall
    hardest where Delta p / p is largest."""
    shifts = []
    for p in MOMENTA:
        r = el.calibrate(*AXIAL, p)
        shifts.append(r["eps_M"] - r["eps_M_0"])
        assert r["eps_M"] < r["eps_M_0"], (p, r["eps_M"], r["eps_M_0"])
        assert r["k"] < r["k_0"], (p, r["k"], r["k_0"])
    assert abs(shifts[0]) > abs(shifts[-1]), shifts


def test_mixed_exceeds_selfconsistent():
    """The deployed estimator divides by theta_space at the TAGGED momentum,
    so its mismatch must exceed the self-consistent one wherever there is
    loss.  If these are ever equal, the wrong quantity is in use."""
    for p in MOMENTA:
        r = el.calibrate(*AXIAL, p)
        assert r["eps_mix"] > r["eps_M"], (p, r["eps_mix"], r["eps_M"])


def test_per_slice_highland_quadrature_is_too_small():
    """PDG states per-segment Highland quadrature underestimates the width.
    Quantify it, so highland_pofx's single-global-log construction is
    justified numerically and not only by citation."""
    for p in MOMENTA:
        path = el.ordered_path(*AXIAL)
        slices, _ = el.slice_path(path, p)
        th_ok, _ = el.highland_pofx(slices, el.x_over_X0(path))
        q = sum(float(theta0_highland(
            s["p"], s["dx"] / MATERIALS[s["mat"]]["X0"])) ** 2 for s in slices)
        ratio = math.sqrt(q) / th_ok
        assert 0.80 < ratio < 0.98, (p, ratio)


# --------------------------------------------------------- vectorised API
def test_vectorised_api_matches_scalar_core_on_grid():
    """The bucketed p(X) API must reproduce energy_loss.calibrate exactly on
    inputs lying on the cache grid."""
    t_snap, X_snap = _snapped(AXIAL)
    for p in (1.0, 3.5):
        direct = el.calibrate(*t_snap, p)
        assert abs(float(eps_M_pofx(p, *X_snap)[0])
                   - direct["eps_M"]) < 1e-9, p
        assert abs(float(eps_M_mixed(p, *X_snap)[0])
                   - direct["eps_mix"]) < 1e-9, p
        assert abs(float(theta_RMS_pofx(p, *X_snap)[0])
                   - direct["th_rms"]) < 1e-12, p

def test_vectorised_api_shape_contract():
    """Must match eps_quadrature's historical scalar -> length-1-array API."""
    X = _areal(AXIAL)
    assert eps_M_pofx(2.0, *X).shape == (1,)
    assert eps_M_pofx(np.array([1.0, 2.0, 3.5, 6.0]), *X).shape == (4,)


    """Corrections to test_pofx.py after the exact-calibration change.

1. DELETE `test_vectorised_api_bucketing_bias_tracks_constant_p_case`.
   Its premise was wrong.  eps_M_pofx takes numerator AND denominator from
   the same calibrate() call at the same bucketed X, so the sqrt(X) scaling
   cancels and only the truncation residual survives (-0.0086 pp measured).
   eps_M_constp rounds the numerator only (+0.0327 pp).  They cannot track
   each other, and the well-behaved one was being penalised for it.

2. ADD everything below.  The first two replace the deleted test with the
   comparison that IS structurally required; the rest cover the exact API
   and the two momentum interpolants.
"""

# --------------------------------------------------------------------------
# Replaces the deleted test
# --------------------------------------------------------------------------

def test_pofx_selfconsistent_bias_is_small_and_negative():
    """eps_M_pofx buckets BOTH sides of the ratio, so the sqrt(X) scaling
    cancels and only the truncation residual survives.  It must be an order
    of magnitude smaller than the one-sided offset, and negative."""
    X = _areal(AXIAL)
    for p in MOMENTA:
        bias = 100.0 * (float(eps_M_pofx(p, *X)[0])
                        - el.calibrate(*AXIAL, p)["eps_M"])
        assert -0.03 < bias < 0.0, (p, bias)


def test_pofx_mixed_bias_tracks_constant_p_case():
    """eps_M_mixed and eps_quadrature.eps_M both round the numerator only, so
    both carry the one-sided offset on the axial path and both are positive.

    They do NOT match exactly, and should not.  In the p(X) numerator the
    extra 0.1 g/cm^2 of copper from the rounding also costs extra energy,
    lowering p through the remainder of the path and raising theta_rms by
    more than the sqrt(X) scaling alone.  The p(X) offset is therefore the
    larger of the two, and the excess must shrink as momentum rises and the
    fractional energy loss falls.  Measured at 1 GeV/c: 4.4e-4 vs 3.3e-4.
    """
    X = _areal(AXIAL)
    ratios = []
    for p in MOMENTA:
        r = el.calibrate(*AXIAL, p)
        bias_mixed = float(eps_M_mixed(p, *X)[0]) - r["eps_mix"]
        bias_constp = float(eps_M_constp(p, *X)[0]) - r["eps_M_0"]
        assert bias_mixed > 0.0 and bias_constp > 0.0, (p, bias_mixed,
                                                        bias_constp)
        assert bias_mixed >= bias_constp, (p, bias_mixed, bias_constp)
        assert bias_mixed < 1e-3, (p, bias_mixed)
        ratios.append(bias_mixed / bias_constp)
    assert all(1.0 <= q < 2.0 for q in ratios), ratios
    assert ratios[0] > ratios[-1], ratios

# --------------------------------------------------------------------------
# Exact constant-p API (eps_quadrature)
# --------------------------------------------------------------------------

def test_exact_api_is_unbucketed():
    """eps_M_exact must vary smoothly with X, not in 0.25 g/cm^2 steps, and
    must equal the repo primitives computed directly."""
    from eps_quadrature import eps_M_exact
    X_al, X_cu, X_pb = _areal(AXIAL)
    for p in MOMENTA:
        a = eps_M_exact(p, X_al, X_cu, X_pb)
        b = eps_M_exact(p, X_al, X_cu + 0.01, X_pb)    # 11 microns of Cu
        assert abs(a - b) < 2e-5, (p, a, b)
        assert abs(a - _eps_M_exact(p, X_al, X_cu, X_pb)) < 1e-14, p


def test_exact_matches_bucketed_on_grid():
    """On grid-aligned inputs the two APIs must agree to machine precision."""
    from eps_quadrature import eps_M_exact
    _, X_snap = _snapped(AXIAL)
    for p in MOMENTA:
        assert abs(eps_M_exact(p, *X_snap)
                   - float(eps_M_constp(p, *X_snap)[0])) < 1e-12, p


def test_marginal_interpolant_accurate():
    """eps_M_marginal's ln-p tabulation must reproduce eps_M_exact well below
    the offset it was introduced to remove (+0.033 pp)."""
    from eps_quadrature import _marginal_interp_error
    err = _marginal_interp_error()
    assert err < 3e-3, err


def test_marginal_is_exact_at_reference_momenta():
    """The four manuscript momenta must come back as the exact values."""
    from eps_quadrature import eps_M_exact, eps_M_marginal
    X_al, X_cu = _areal(AXIAL)[0], _areal(AXIAL)[1]
    for p in MOMENTA:
        got = float(eps_M_marginal(p)[0])
        want = eps_M_exact(p, X_al, X_cu, 0.0)
        assert abs(got - want) < 3e-5, (p, got, want)


def test_marginal_handles_pathological_momenta():
    """Reconstructed p_meas can land far outside the tabulated range when
    delta_meas is small, or be non-finite.  Neither may extrapolate or crash."""
    from eps_quadrature import eps_M_exact, eps_M_marginal
    X_al, X_cu = _areal(AXIAL)[0], _areal(AXIAL)[1]
    for p in (0.05, 500.0):
        got = float(eps_M_marginal(p)[0])
        assert np.isfinite(got), p
        assert abs(got - eps_M_exact(p, X_al, X_cu, 0.0)) < 1e-12, p
    bad = eps_M_marginal(np.array([-1.0, 0.0, np.nan]))
    assert np.all(bad == 0.0), bad


def test_marginal_shape_contract_unchanged():
    """results_pipeline calls this with a 1-D array of p_meas."""
    from eps_quadrature import eps_M_marginal
    assert eps_M_marginal(2.0).shape == (1,)
    assert eps_M_marginal(np.array([1.0, 2.0, 3.5, 6.0])).shape == (4,)


# --------------------------------------------------------------------------
# Exact p(X) API (eps_quadrature_pofx)
# --------------------------------------------------------------------------

def test_pofx_exact_api_matches_calibrate():
    """The exact p(X) wrappers must be transparent over energy_loss."""
    from eps_quadrature_pofx import (calibrate_exact, eps_M_mixed_exact,
                                     eps_M_pofx_exact, theta_RMS_pofx_exact)
    X = _areal(AXIAL)
    for p in (1.0, 6.0):
        direct = el.calibrate(*AXIAL, p)
        assert abs(eps_M_pofx_exact(p, *X) - direct["eps_M"]) < 1e-14, p
        assert abs(eps_M_mixed_exact(p, *X) - direct["eps_mix"]) < 1e-14, p
        assert abs(theta_RMS_pofx_exact(p, *X) - direct["th_rms"]) < 1e-14, p
        assert calibrate_exact(p, *X)["n_slices"] == direct["n_slices"], p


def test_pofx_marginal_is_exact_and_mixed_by_default():
    """eps_M_marginal_pofx feeds I_p, whose denominator sits at the tagged
    momentum.  Its default must therefore be the MIXED quantity, and it must
    reproduce calibrate_exact rather than the bucketed eps_M_mixed."""
    from eps_quadrature_pofx import eps_M_marginal_pofx
    for p in MOMENTA:
        direct = el.calibrate(*AXIAL, p)
        got = float(eps_M_marginal_pofx(p)[0])
        assert abs(got - direct["eps_mix"]) < 3e-5, (p, got, direct["eps_mix"])
        alt = float(eps_M_marginal_pofx(p, mixed=False)[0])
        assert abs(alt - direct["eps_M"]) < 3e-5, (p, alt, direct["eps_M"])
        assert got > alt, (p, got, alt)


def test_pofx_marginal_interpolant_accurate():
    from eps_quadrature_pofx import _marginal_interp_error
    err = _marginal_interp_error()
    assert err < 3e-3, err


def test_pofx_marginal_shape_and_pathological_momenta():
    from eps_quadrature_pofx import eps_M_marginal_pofx
    assert eps_M_marginal_pofx(2.0).shape == (1,)
    assert eps_M_marginal_pofx(np.array([1.0, 2.0, 3.5, 6.0])).shape == (4,)
    bad = eps_M_marginal_pofx(np.array([-1.0, 0.0, np.nan]))
    assert np.all(bad == 0.0), bad
    # a 0.1 GeV/c muon cannot cross 161 g/cm^2; must return 0, not raise
    assert float(eps_M_marginal_pofx(0.1)[0]) == 0.0

def _self_check_registry():
    import ast as _ast
    src = _ast.parse(open(__file__).read())
    declared = {n.name for n in src.body
                if isinstance(n, _ast.FunctionDef) and n.name.startswith("test_")}
    collected = {k for k in globals() if k.startswith("test_")}
    missing = declared - collected
    assert not missing, f"tests declared but not collected: {sorted(missing)}"
    return len(declared)


if __name__ == "__main__":
    import sys
    n_declared = _self_check_registry()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    assert len(fns) == n_declared, "test registry mismatch"
    fails = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except Exception as e:
            fails += 1
            print(f"FAIL {f.__name__}: {e}")
    sys.exit(1 if fails else 0)
