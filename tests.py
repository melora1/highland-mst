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
    """Before clipping, the truncated series must integrate to ~1, and the
    clipped negative fraction must be << 1%. If not, n<=2 is failing."""
    for p in (1.0, 2.0, 3.5, 6.0):
        for X in (27.0, 100.0, 170.0):     # g/cm^2, representative paths
            chi_c2, chi_a2 = ml.combine_path(0.3 * X, 0.7 * X, 0.0, p)
            B = ml.solve_B(chi_c2, chi_a2)
            _, clipped = ml.pdf_on_grid(chi_c2, B, nmax=2)
            assert clipped < 1e-2, (p, X, B, clipped)


def test_gaussian_limit_matches_highland():
    """n=0 term alone must reproduce a Gaussian whose RMS is within the
    Lynch-Dahl 11% band of the Highland theta0."""
    p = 2.0
    t_al, t_cu = 10.0, 15.0
    X_al = MATERIALS["Al"]["rho"] * t_al
    X_cu = MATERIALS["Cu"]["rho"] * t_cu
    xx0 = t_al / MATERIALS["Al"]["X0"] + t_cu / MATERIALS["Cu"]["X0"]

    chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, 0.0, p)
    B = ml.solve_B(chi_c2, chi_a2)
    pdf, _ = ml.pdf_on_grid(chi_c2, B, nmax=0)
    th = ml._THETA_GRID
    rms = np.sqrt(_trapz(pdf * th ** 2, th))

    t0 = float(theta0_highland(p, xx0))
    assert abs(rms / t0 - 1.0) < 0.11, (rms, t0, rms / t0)


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
    """PROJECTED tail power is theta^-3, not theta^-4.

    theta^-4 is the SPACE-angle (Rutherford) power; projecting a 2-D
    theta^-4 tail onto one axis gives theta^-3. Asserting -4 here was the
    original bug.
    """
    p = 2.0
    chi_c2, chi_a2 = ml.combine_path(27.0, 134.0, 0.0, p)
    B = ml.solve_B(chi_c2, chi_a2)
    scale = np.sqrt(chi_c2 * B)
    pdf, _ = ml.pdf_on_grid(chi_c2, B, nmax=2)
    th = ml._THETA_GRID
    eta = th / scale
    m = (eta > 6.0) & (eta < ml._ETA_MAX) & (pdf > 0)
    slope = np.polyfit(np.log(th[m]), np.log(pdf[m]), 1)[0]
    assert -3.3 < slope < -2.7, slope


def test_single_scatter_limit():
    """ABSOLUTE tail normalisation, the strongest check in this module.

        f1 -> 1/(2 eta^3)  =>  F(theta) -> chi_c^2 / (2 theta^3)

    B cancels: the tail is fixed by chi_c^2 alone, i.e. by the number of
    scatterers. No free parameter. This catches a wrong f1 normalisation,
    a wrong chi_c, and a wrong prefactor, none of which the slope test sees.

    Checked at several (p, path) points, since B varies between them and a
    surviving B dependence would show up as a spread.
    """
    cases = [(2.0, 27.0, 134.0), (1.0, 27.0, 134.0),
             (6.0, 27.0, 134.0), (2.0, 27.0, 0.0), (3.5, 27.0, 60.0)]
    tested = 0
    for p, X_al, X_cu in cases:
        chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, 0.0, p)
        B = ml.solve_B(chi_c2, chi_a2)
        scale = np.sqrt(chi_c2 * B)
        eta_reach = ml.THETA_GRID_MAX / scale
        if eta_reach < 12.0:
            # theta grid does not extend far enough into the tail at this
            # (p, path) to test the asymptote. Expected at low p, where the
            # angular scale is large. Not a failure -- just not testable here.
            continue
        pdf, _ = ml.pdf_on_grid(chi_c2, B, nmax=2)
        th = ml._THETA_GRID
        eta = th / scale
        m = (eta > 10.0) & (eta < min(ml._ETA_MAX, eta_reach)) & (pdf > 0)
        assert m.sum() > 20, (p, X_cu, "window too narrow")
        ratio = pdf[m] / (chi_c2 / (2.0 * th[m] ** 3))
        assert abs(ratio.mean() - 1.0) < 0.05, (p, X_cu, B, ratio.mean())
        tested += 1
    assert tested >= 3, f"only {tested} cases reached the asymptotic tail"


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


def test_momentum_position_correlation_exists():
    """Branch B's entire premise requires that muons of different momenta
    sample different regions of the target (Sec. 2.3). This test asserts that
    the CHOSEN configuration actually produces that correlation.

    Measured spread of median PoCA x across the four settings:

        pencil  / steer=none         4.90 cm
        pencil  / steer=per_setting  0.04 cm
        uniform / steer=none         4.68 cm
        uniform / steer=per_setting  0.13 cm

    The correlation comes entirely from the uncorrected dipole kick, which
    displaces setting p by 5.85/p cm. Retune the beamline per setting -- as a
    real tagged beamline is operated -- and the spread falls below one voxel
    (0.6 cm), so the artifact is a near-uniform rescaling and Sec. 2.3's
    spatially-structured claim has no mechanism left.

    Failing here does not mean the code is wrong. It means the configuration
    cannot support the paper's central claim, and one of the two has to change.
    """
    from simulate import simulate_setting
    med = [np.median(simulate_setting(p, n=15000, mode="gauss").poca_x)
           for p in MOMENTA]
    spread = max(med) - min(med)
    assert spread > VOX_SIZE, (
        f"median PoCA x spans only {spread:.2f} cm across settings, under one "
        f"{VOX_SIZE:.1f} cm voxel. No momentum-position correlation, so the "
        "Sec. 2.3 artifact cannot be spatially structured. See "
        "STEER_COMPENSATION in config.py.")


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