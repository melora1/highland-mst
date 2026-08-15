"""Pre-flight validation (plan Sec. 7 step 1-3, Sec. 8 pitfalls).

Run these BEFORE the 2e6-event production run. Every failure here is a bug
that would otherwise silently corrupt eps_M.
"""

import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz

import moliere as ml
from config import (AL_HALF, CU_HALF, MATERIALS, MOMENTA, PB_CX, PB_CY,
                    PB_R, SIGMA_HIT, THETA_CUT, VOX_SIZE)
from geometry import trace_ref, trace_true, x_over_X0
from kinematics import theta0_highland


def test_geometry_axial():
    """z-directed ray through the Pb axis: t_Pb=15, t_Cu=0, t_Al=10, sum=25."""
    o = np.array([[PB_CX, PB_CY, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    tAl, tCu, tPb = trace_true(o, u)
    assert np.isclose(tPb[0], 15.0), tPb
    assert np.isclose(tCu[0], 0.0), tCu
    assert np.isclose(tAl[0], 10.0), tAl
    assert np.isclose(tAl[0] + tCu[0] + tPb[0], 2 * AL_HALF)


def test_geometry_offaxis():
    """z-directed ray missing the Pb: t_Cu=15, t_Al=10."""
    o = np.array([[-6.0, -6.0, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    tAl, tCu, tPb = trace_true(o, u)
    assert np.isclose(tPb[0], 0.0)
    assert np.isclose(tCu[0], 2 * CU_HALF)
    assert np.isclose(tAl[0], 2 * (AL_HALF - CU_HALF))


def test_geometry_total_invariant():
    """Every z-directed ray inside the shell traverses exactly 25 cm."""
    rng = np.random.default_rng(0)
    n = 2000
    o = np.stack([rng.uniform(-11, 11, n), rng.uniform(-11, 11, n),
                  np.full(n, -100.0)], 1)
    u = np.tile([0.0, 0.0, 1.0], (n, 1))
    tAl, tCu, tPb = trace_true(o, u)
    assert np.allclose(tAl + tCu + tPb, 2 * AL_HALF, atol=1e-9)


def test_ref_has_no_pb():
    o = np.array([[PB_CX, PB_CY, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    _, tCu, tPb = trace_ref(o, u)
    assert tPb[0] == 0.0
    assert np.isclose(tCu[0], 2 * CU_HALF)


def test_ref_xx0_less_than_true():
    """The reference geometry must under-predict scattering where Pb is
    present -- that is the imaging signal (Sec. 2.3)."""
    o = np.array([[PB_CX, PB_CY, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    xt = x_over_X0(*trace_true(o, u))
    xr = x_over_X0(*trace_ref(o, u))
    assert xt[0] > xr[0]


def test_solve_B():
    """B - ln B = Omega_0 must be satisfied to machine precision."""
    for om in [5.0, 8.0, 12.0, 20.0]:
        chi_c2 = 1.0
        chi_a2 = chi_c2 / (1.167 * np.exp(om))
        B = ml.solve_B(chi_c2, chi_a2)
        assert abs(B - np.log(B) - om) < 1e-9, (om, B)


def test_pdf_normalisation():
    """Primary radial n<=2 series must remain normalized and non-negative
    clipping must be negligible over representative paths."""
    for p in (1.0, 2.0, 3.5, 6.0):
        for X in (27.0, 100.0, 170.0):
            chi_c2, chi_a2 = ml.combine_path(0.3 * X, 0.7 * X, 0.0, p)
            B = ml.solve_B(chi_c2, chi_a2)
            norm, clipped = ml.radial_total_mass(B, nmax=2)
            assert abs(norm - 1.0) < 2e-4, (p, X, B, norm)
            assert clipped < 1e-2, (p, X, B, clipped)

def test_gaussian_limit_matches_highland():
    """The radial n=0 core has magnitude RMS s and projected RMS s/sqrt(2),
    which must lie inside the Lynch-Dahl/Highland core-width band."""
    p = 2.0
    t_al, t_cu = 10.0, 15.0
    X_al = MATERIALS["Al"]["rho"] * t_al
    X_cu = MATERIALS["Cu"]["rho"] * t_cu
    xx0 = t_al / MATERIALS["Al"]["X0"] + t_cu / MATERIALS["Cu"]["X0"]

    chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, 0.0, p)
    B = ml.solve_B(chi_c2, chi_a2)
    scale = np.sqrt(chi_c2 * B)
    _, M2, _ = ml.radial_moments(chi_c2, B, 10.0 * scale, nmax=0)
    plane_rms = np.sqrt(M2 / 2.0)

    t0 = float(theta0_highland(p, xx0))
    assert abs(plane_rms / t0 - 1.0) < 0.11, (plane_rms, t0, plane_rms / t0)

def test_projected_kernel_n0():
    """The generating integral must reproduce Appendix A's f^(0) exactly.
    This validates the cos-kernel projection that defines f1 and f2."""
    for e in (0.0, 1.0, 2.0, 3.0, 4.0):
        got = ml._fp_quad(0, e)
        want = np.exp(-e * e) / np.sqrt(np.pi)
        assert abs(got - want) < 1e-10, (e, got, want)


def test_correction_terms_integrate_to_zero():
    """f1 and f2 are corrections: they must integrate to zero over the full
    projected range, so the n<=2 series stays normalised (App. A)."""
    e = np.linspace(-ml._ETA_MAX, ml._ETA_MAX, 40001)
    for f in (ml.f1, ml.f2):
        I = _trapz(f(e), e)
        assert abs(I) < 2e-3, (f.__name__, I)


def test_tail_asymptote():
    """PRIMARY radial areal density must approach the Rutherford Theta^-4 tail."""
    p = 3.5
    chi_c2, chi_a2 = ml.combine_path(27.0, 134.0, 0.0, p)
    B = ml.solve_B(chi_c2, chi_a2)
    scale = np.sqrt(chi_c2 * B)
    eta = np.geomspace(10.0, 25.0, 150)
    th = eta * scale
    P = ml.radial_density(th, chi_c2, B, nmax=2)
    slope = np.polyfit(np.log(th), np.log(P), 1)[0]
    assert -4.25 < slope < -3.75, slope

def test_single_scatter_limit():
    """Absolute radial Rutherford normalization:

        P_M(Theta) -> chi_c^2 / (pi Theta^4).

    This is the manuscript's defining two-dimensional tail.  It checks the
    Hankel normalization, chi_c, and cancellation of B in the n=1 tail.
    """
    cases = [(2.0, 27.0, 134.0), (3.5, 27.0, 134.0),
             (6.0, 27.0, 134.0), (3.5, 27.0, 60.0)]
    for p, X_al, X_cu in cases:
        chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, 0.0, p)
        B = ml.solve_B(chi_c2, chi_a2)
        scale = np.sqrt(chi_c2 * B)
        eta = np.geomspace(12.0, 25.0, 100)
        th = eta * scale
        P = ml.radial_density(th, chi_c2, B, nmax=2)
        ratio = P / (chi_c2 / (np.pi * th ** 4))
        assert abs(ratio.mean() - 1.0) < 0.05, (p, X_cu, B, ratio.mean())

def test_radial_kernel_n0():
    """The Hankel generating integral must give Phi0=2 exp(-eta^2)."""
    for e in (0.0, 1.0, 2.0, 3.0, 4.0):
        got = ml._phi_quad(0, e)
        want = 2.0 * np.exp(-e * e)
        assert abs(got - want) < 1e-10, (e, got, want)


def test_radial_correction_terms_preserve_normalisation():
    """Radial correction terms integrate to zero with measure eta d eta."""
    e = np.linspace(0.0, ml._RADIAL_ETA_MAX, 50001)
    I1 = _trapz(e * ml.phi1(e), e) + 1.0 / ml._RADIAL_ETA_MAX ** 2
    I2 = _trapz(e * ml.phi2(e), e)
    assert abs(I1) < 2e-4, I1
    assert abs(I2) < 2e-4, I2


def test_radial_sampler_matches_quadrature():
    """Accepted-space-angle RMS from sampling must close on radial quadrature."""
    p = 2.0
    X_al = MATERIALS["Al"]["rho"] * 10.0
    X_cu = MATERIALS["Cu"]["rho"] * 15.0
    chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, 0.0, p)
    B = ml.solve_B(chi_c2, chi_a2)
    Fc, M2, _ = ml.radial_moments(chi_c2, B, THETA_CUT)

    n = 150000
    rng = np.random.default_rng(123)
    sampler = ml.MoliereSampler(nmax=2)
    tx, ty = sampler.sample(np.full(n, p), np.full(n, X_al),
                            np.full(n, X_cu), np.zeros(n), rng)
    r = np.hypot(tx, ty)
    keep = r < THETA_CUT
    sample_fc = keep.mean()
    sample_rms = np.sqrt(np.mean(r[keep] ** 2))
    assert abs(sample_fc - Fc) < 0.003, (sample_fc, Fc)
    assert abs(sample_rms / np.sqrt(M2) - 1.0) < 0.01, (sample_rms, np.sqrt(M2))


def test_radial_sampler_is_azimuthally_isotropic():
    """Hard scatters must not be concentrated along Cartesian axes."""
    p = 3.5
    X_al = MATERIALS["Al"]["rho"] * 10.0
    X_cu = MATERIALS["Cu"]["rho"] * 15.0
    chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, 0.0, p)
    B = ml.solve_B(chi_c2, chi_a2)
    scale = np.sqrt(chi_c2 * B)

    n = 250000
    rng = np.random.default_rng(321)
    sampler = ml.MoliereSampler(nmax=2)
    tx, ty = sampler.sample(np.full(n, p), np.full(n, X_al),
                            np.full(n, X_cu), np.zeros(n), rng)
    r = np.hypot(tx, ty)
    tail = r > 3.0 * scale
    phi = np.arctan2(ty[tail], tx[tail])
    assert tail.sum() > 1000
    # cos(4 phi) is the leading four-fold anisotropy of an x/y-factorized tail.
    assert abs(np.mean(np.cos(4.0 * phi))) < 0.04


def test_sigma_delta_emergent():
    """sigma_delta must EMERGE from hit smearing as 2 sigma_hit / dz."""
    from simulate import simulate_setting
    df = simulate_setting(6.0, n=20000, mode="gauss")
    # at 6 GeV/c, delta = 15 mrad; spread of delta_meas about it
    sd = np.std(df.delta_meas.values)
    expect = 2 * SIGMA_HIT / 30.0
    assert abs(sd / expect - 1.0) < 0.10, (sd, expect)


def test_rois_are_illuminated():
    """The Pb inclusion's transverse footprint must actually receive flux.

    The dipole steers the beam by 0.3*B*L/p * |z_magnet| = 5.85/p cm at the
    target, so a sigma_xy = 1 cm pencil beam lights a ~2 cm strip at a
    DIFFERENT x for every momentum (5.84 cm at 1 GeV/c, 0.97 cm at 6), with y
    pinned near 0. The Pb inclusion at (3,2) with r=2 spans y in [0,4], so
    only its bottom edge is ever lit and both ROIs end up mostly empty.

    That is what makes edge_response' erf fit diverge, but the fit is only the
    symptom: with unlit ROIs, SNR/CNR/PSF are all meaningless. This test fails
    BEFORE the production run rather than after it.

    Coverage is checked in the TRANSVERSE PLANE, not on the 3-D voxel grid:
    3-D occupancy is statistics-limited (468 Pb voxels), so at test statistics
    it cannot distinguish 'beam misses the target' from 'not enough events'.
    The 2-D footprint is ~35 bins and is populated whenever the beam covers it.
    """
    import branch_b as bb
    from simulate import simulate_setting

    df = pd.concat([simulate_setting(p, n=20000, mode="gauss")
                    for p in MOMENTA], ignore_index=True)
    df = df[df.pass_reco]

    C = bb.CENTERS
    ix = np.clip(np.digitize(df.poca_x.values, bb.EDGES) - 1, 0, bb.N_VOX - 1)
    iy = np.clip(np.digitize(df.poca_y.values, bb.EDGES) - 1, 0, bb.N_VOX - 1)
    hit2d = np.zeros((bb.N_VOX, bb.N_VOX), dtype=bool)
    hit2d[ix, iy] = True

    GX, GY = np.meshgrid(C, C, indexing="ij")
    foot = np.hypot(GX - PB_CX, GY - PB_CY) <= PB_R
    empty = (~hit2d[foot]).mean()
    assert empty < 0.20, (
        f"{empty:.0%} of the Pb transverse footprint receives no flux. "
        "The beam does not illuminate the inclusion -- see BEAM_MODE "
        "in config.py.")


def test_beam_covers_target_face():
    """Transverse beam extent must cover the Cu block, else there is no
    tomogram to reconstruct."""
    from simulate import simulate_setting
    df = pd.concat([simulate_setting(p, n=20000, mode="gauss")
                    for p in MOMENTA], ignore_index=True)
    for ax in ("poca_x", "poca_y"):
        lo, hi = np.percentile(df[ax], [2.5, 97.5])
        assert hi - lo > 2 * CU_HALF * 0.8, (
            f"{ax} 95% span is {hi - lo:.1f} cm; Cu block is "
            f"{2 * CU_HALF} cm across. Beam does not cover the target.")


def test_uncompensated_configuration_has_momentum_position_correlation():
    """Diagnostic/legacy configuration check.

    config.py deliberately defaults to STEER_COMPENSATION='none'.  Under that
    uncompensated configuration, nominal momentum settings land at measurably
    different transverse positions.  This test validates that diagnostic
    mechanism only; it does NOT validate the production results pipeline,
    which explicitly uses STEER_COMPENSATION='per_setting'.
    """
    from config import STEER_COMPENSATION
    from simulate import simulate_setting

    assert STEER_COMPENSATION == "none", (
        "This diagnostic test assumes config.py's deliberate 'none' default; "
        f"got {STEER_COMPENSATION!r}."
    )
    med = [np.median(simulate_setting(p, n=15000, mode="gauss").poca_x)
           for p in MOMENTA]
    spread = max(med) - min(med)
    assert spread > VOX_SIZE, (
        f"uncompensated median PoCA-x spread is only {spread:.3f} cm, "
        f"below one {VOX_SIZE:.3f} cm voxel; the diagnostic steering "
        "correlation is unexpectedly absent."
    )


def test_production_configuration_has_no_gross_setting_steering():
    """Production steering must not manufacture a multi-voxel correlation.

    results_pipeline.py sets config.STEER_COMPENSATION='per_setting' before
    importing simulate.py.  Reproduce that import-time configuration here,
    then restore the module state so the rest of the test suite continues to
    exercise config.py's deliberate default.
    """
    import importlib
    import config
    import simulate

    original = config.STEER_COMPENSATION
    try:
        config.STEER_COMPENSATION = "per_setting"
        simulate = importlib.reload(simulate)
        med = [np.median(simulate.simulate_setting(p, n=15000, mode="gauss").poca_x)
               for p in MOMENTA]
        spread = max(med) - min(med)
        assert spread < VOX_SIZE, (
            f"production median PoCA-x spread is {spread:.3f} cm, exceeding "
            f"one {VOX_SIZE:.3f} cm voxel; per-setting steering is "
            "introducing a gross setting-dependent spatial correlation."
        )
    finally:
        config.STEER_COMPENSATION = original
        importlib.reload(simulate)


def test_no_jensen_shortcut():
    """theta_pred at <p> must be SMALLER than the per-event value (Jensen).
    This is the bug this test exists to prevent."""
    from branch_a import theta_pred
    rng = np.random.default_rng(1)
    p = 2.0 * (1 + 0.1 * rng.normal(size=100000))
    xx0 = np.full(p.size, 12.0)
    per_event = theta_pred(p, xx0)
    at_mean = theta_pred(np.full(p.size, p.mean()), xx0)
    assert per_event > at_mean


def _self_check_registry():
    """Guard against a test being silently orphaned by an edit that replaces a
    `def` line. Two tests were lost this way during development, and the suite
    still reported all-green. Compares the functions the runner collects
    against the module's AST.
    """
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