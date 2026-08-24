#!/usr/bin/env python3
"""Physics/geometry pre-flight tests for the condensed revision codebase.

These tests are deliberately biased toward closures that can catch a wrong
physics implementation, not merely API/syntax regressions.  They remain fast
enough to run before every production job.
"""

from __future__ import annotations

import math
import sys
import numpy as np
import pandas as pd

from analysis import (
    CENTERS,
    PATHS,
    build_weights,
    fiducial_voxel_mask,
    highland_path_derivative,
    optimal_global_scale,
    roi_masks,
)
from config import AL_HALF, MATERIALS, MOMENTA, RADIAL_ETA_MAX, THETA_CUT, VOX_SIZE
from geometry import trace_paths
from physics import (
    Layer,
    PofxCache,
    accumulate_moliere_pofx,
    beta_of,
    calibrate_pofx,
    constant_calibration,
    constant_path_parameters,
    dedx_of_E,
    dimensionless_moments_quad,
    energy_after,
    fit_log_asymptote,
    layers_from_segment_thicknesses,
    phi0,
    phi1,
    radial_moments,
    radial_tail_ratio,
    radial_total_mass,
    reduced_parameters,
    theta0_highland,
    validate_stopping_minima,
)
from simulation import (
    momentum_fractions,
    nominal_target_offset_cm,
    raster_nodes,
    seed_entropy,
    simulate_fixed_node,
)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def geometry_reference_central():
    o = np.array([[3.0, 2.0, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    r = trace_paths(o, u, reference=True)
    assert np.allclose(r["segments"][0], [5, 15, 0, 0, 5], atol=1e-10), r["segments"]


@test
def geometry_true_pb_axis():
    o = np.array([[3.0, 2.0, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    r = trace_paths(o, u, reference=False)
    assert np.allclose(r["segments"][0], [5, 0, 15, 0, 5], atol=1e-10), r["segments"]


@test
def geometry_offcu_is_25cm_al():
    o = np.array([[10.0, 0.0, -100.0]])
    u = np.array([[0.0, 0.0, 1.0]])
    r = trace_paths(o, u, reference=True)
    assert np.allclose(r["segments"][0], [25, 0, 0, 0, 0], atol=1e-10), r["segments"]


@test
def reference_numbers_match_plan():
    xx = 10 / MATERIALS["Al"].X0 + 15 / MATERIALS["Cu"].X0
    assert abs(xx - 11.570) < 0.001
    vals = [50.88, 25.32, 14.46, 8.43]
    for p, want in zip(MOMENTA, vals):
        got = float(theta0_highland(p, xx)) * 1000
        assert abs(got - want) < 0.03, (p, got, want)


@test
def reduced_identity_exact():
    for p in MOMENTA:
        r = constant_calibration(PATHS["AlCu"], p)
        assert abs((1 + r["epsilon"]) ** 2 - r["exact_ratio2"]) < 2e-12
        assert abs(r["eta_cut"] - r["k"] / math.sqrt(2 * r["R"] * r["B"])) < 1e-12


@test
def R_B_nearly_momentum_invariant():
    rs = [reduced_parameters(PATHS["AlCu"], p) for p in MOMENTA]
    B = np.array([r["B"] for r in rs])
    R = np.array([r["R"] for r in rs])
    assert np.ptp(B) / B.mean() < 0.002, B
    assert np.ptp(R) / R.mean() < 0.002, R
    RB = R * B
    assert np.ptp(RB) / RB.mean() < 3e-4, RB


@test
def joint_log_asymptote_fit_recovers_slope_and_intercept():
    # Exact synthetic closure for y=(1+eps)^2-1 = m ln(eta/eta1).
    R = 0.062
    m = 2.0 * R
    eta1 = 1.37
    eta = np.geomspace(10.0, 200.0, 31)
    y = m * np.log(eta / eta1)
    eps = np.sqrt(1.0 + y) - 1.0
    f = fit_log_asymptote(eta, eps, R=R)
    assert abs(f["slope"] - m) < 2e-14
    assert abs(f["eta1"] - eta1) < 2e-13
    assert abs(f["slope_ratio"] - 1.0) < 2e-13


@test
def radial_subtable_log_slope_is_consistent_with_rutherford():
    # Independent finite-eta diagnostic: every fit point lies in the numerical
    # Phi^(1)/Phi^(2) table.  Unlike a 50--500 fit, this cannot inherit 2R from
    # the analytic Rutherford continuation above RADIAL_ETA_MAX.
    X = PATHS["AlCu"]
    rp = reduced_parameters(X, 1.0)
    eta = np.geomspace(10.0, RADIAL_ETA_MAX, 25)
    assert float(np.max(eta)) <= RADIAL_ETA_MAX
    eps = []
    for e in eta:
        cut = e * math.sqrt(rp["chi_c2"] * rp["B"])
        eps.append(constant_calibration(X, 1.0, cut, nmax=2)["epsilon"])
    f = fit_log_asymptote(eta, eps, R=rp["R"])
    # Finite-eta corrections are expected; this is a consistency test, not an
    # assertion that the asymptotic coefficient has already been reached.
    assert abs(f["slope_ratio"] - 1.0) < 0.03, f


@test
def phi0_gaussian_kernel_and_n0_moments():
    eta = np.array([0.0, 0.2, 1.0, 2.5, 5.0])
    assert np.allclose(phi0(eta), 2.0 * np.exp(-eta * eta), rtol=0, atol=2e-15)
    r = constant_calibration(PATHS["AlCu"], 2.0)
    s = math.sqrt(r["chi_c2"] * r["B"])
    Fc, M2, M4 = radial_moments(r["chi_c2"], r["B"], 10 * s, nmax=0)
    assert abs(Fc - 1.0) < 1e-10
    assert abs(M2 / s**2 - 1.0) < 2e-6
    assert abs(M4 / (2 * s**4) - 1.0) < 5e-6
    # Highland is a fitted core width rather than an identity with s, but the
    # two core scales should agree at the few-percent level for this reference.
    assert abs(s / r["theta_space"] - 1.0) < 0.05


@test
def radial_normalization_clipping_and_absolute_tail():
    r = constant_calibration(PATHS["AlCu"], 6.0)
    total, clipped = radial_total_mass(r["B"], nmax=2)
    assert abs(total - 1.0) < 5e-4, total
    assert clipped < 1e-3, clipped
    s = math.sqrt(r["chi_c2"] * r["B"])
    # Inside the numerical table, Phi^(1) must already approach its Rutherford
    # coefficient 2/eta^4.  This checks the table itself rather than the
    # hard-coded analytic continuation.
    for eta in (20.0, 25.0):
        assert eta < RADIAL_ETA_MAX
        coeff = float(phi1(eta)) * eta**4 / 2.0
        assert abs(coeff - 1.0) < 0.05, (eta, coeff)
    # Above the numerical table, separately check the absolute normalized tail.
    for eta in (50.0, 100.0):
        assert eta > RADIAL_ETA_MAX
        ratio = radial_tail_ratio(eta * s, r["chi_c2"], r["B"], nmax=2)
        assert abs(ratio - 1.0) < 5e-3, (eta, ratio)


@test
def segmented_screening_both_rules_reduce_to_serial_at_constant_p():
    p = 2.0
    b = float(beta_of(p))
    slices = []
    for name, t in (("Al", 5.0), ("Cu", 15.0), ("Al", 5.0)):
        m = MATERIALS[name]
        X = t * m.rho
        slices.append(
            dict(material=name, X=X, thickness_cm=t, p=p, beta=b, pbeta=p * b)
        )
    X = {"Al": 10 * MATERIALS["Al"].rho, "Cu": 15 * MATERIALS["Cu"].rho, "Pb": 0.0}
    q = constant_path_parameters(X, p)
    for mode in ("dchi_c2", "serial"):
        c2, a2, B = accumulate_moliere_pofx(slices, screening_weight=mode)
        assert abs(c2 / q["chi_c2"] - 1) < 1e-13
        assert abs(a2 / q["chi_a2"] - 1) < 1e-13
        assert abs(B / q["B"] - 1) < 1e-13


@test
def pofx_screening_convention_spread_is_exposed():
    path = (Layer("Al", 5.0), Layer("Cu", 15.0), Layer("Al", 5.0))
    d = calibrate_pofx(path, 1.0, screening_weight="dchi_c2")
    s = calibrate_pofx(path, 1.0, screening_weight="serial")
    assert d["screening_weight"] == "dchi_c2" and s["screening_weight"] == "serial"
    # They must differ under appreciable loss, but remain a small model spread.
    spread_pp = 100 * abs(d["epsilon_matched"] - s["epsilon_matched"])
    assert 1e-3 < spread_pp < 0.5, spread_pp


@test
def range_table_matches_independent_RK4():
    def rk4(E0, material, X, n=12000):
        h = X / n
        E = float(E0)
        f = lambda e: -1e-3 * dedx_of_E(e, material)  # GeV per g/cm^2
        for _ in range(n):
            k1 = f(E)
            k2 = f(E + 0.5 * h * k1)
            k3 = f(E + 0.5 * h * k2)
            k4 = f(E + h * k3)
            E += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return E

    E0 = math.hypot(1.0, 0.10566)
    for mat, X in (("Al", 27.0), ("Cu", 100.0), ("Pb", 50.0)):
        tab = energy_after(E0, mat, X)
        ref = rk4(E0, mat, X)
        assert abs(tab - ref) < 2e-4, (mat, tab, ref, (tab - ref) * 1000)


@test
def energy_loss_monotonic_and_nontrivial():
    path = (Layer("Al", 5.0), Layer("Cu", 15.0), Layer("Al", 5.0))
    a = calibrate_pofx(path, 1.0)
    b = calibrate_pofx(path, 6.0)
    assert a["p_out"] < 1.0 and b["p_out"] < 6.0
    assert a["dp_over_p"] < b["dp_over_p"] < 0.0
    assert a["n_slices"] > b["n_slices"]


@test
def sampler_matches_quadrature_and_is_isotropic():
    n = 60_000
    rng = np.random.default_rng(20260816)
    cache = PofxCache(nmax=2)
    p = np.full(n, 2.0)
    seg0 = np.array([5.0, 15.0, 0.0, 0.0, 5.0])
    seg = np.tile(seg0, (n, 1))
    tx, ty = cache.sample(p, seg, rng)
    th = np.hypot(tx, ty)
    keep = th <= THETA_CUT
    r = cache.calibration(2.0, seg0, THETA_CUT)

    # Acceptance closure against binomial counting uncertainty.
    fc_obs = float(keep.mean())
    sig_fc = math.sqrt(r["Fc"] * (1 - r["Fc"]) / n)
    assert abs(fc_obs - r["Fc"]) < 6 * sig_fc + 5e-4, (fc_obs, r["Fc"], sig_fc)

    # Accepted M2 closure using the model's own fourth moment as the sampling
    # variance of theta^2; this has substantially more teeth than a fixed % cut.
    q = th[keep] ** 2
    m2_obs = float(q.mean())
    nacc = int(keep.sum())
    se_m2 = math.sqrt(max(r["M4"] - r["M2"] ** 2, 0.0) / nacc)
    assert abs(m2_obs - r["M2"]) < 6 * se_m2 + 0.003 * r["M2"], (m2_obs, r["M2"], se_m2)

    # Azimuthal isotropy on the accepted sample.  Do not use the untruncated
    # formal Rutherford tail here: its heavy tail makes finite-sample component
    # variances unnecessarily unstable even when azimuth is exactly uniform.
    xa, ya = tx[keep], ty[keep]
    for v in (xa, ya):
        assert abs(v.mean()) < 6 * v.std(ddof=1) / math.sqrt(v.size)
    vx, vy = np.var(xa, ddof=1), np.var(ya, ddof=1)
    assert abs(vx / vy - 1.0) < 0.04, (vx, vy)
    assert abs(np.mean(xa * ya)) < 0.03 * math.sqrt(vx * vy)


@test
def adaptive_quadrature_matches_production_grid():
    for p in MOMENTA:
        r = constant_calibration(PATHS["AlCu"], p)
        for eta_cut in (10.0, r["eta_cut"]):
            mass, n2, _, *_ = dimensionless_moments_quad(eta_cut, r["B"])
            mu_quad = n2 / mass
            from physics import mu2_eta

            assert abs(mu2_eta(eta_cut, r["B"]) / mu_quad - 1.0) < 1e-5


@test
def cut_independent_path_cache_reproduces_direct_calibration():
    seg = np.array([5.0, 15.0, 0.0, 0.0, 5.0])
    cache = PofxCache()
    for cut in (0.1, 0.2, 0.3):
        cached = cache.calibration(2.0, seg, cut)
        pp, decoded_seg, cc = cache._decode(cache._key(2.0, seg, cut))
        direct = calibrate_pofx(layers_from_segment_thicknesses(decoded_seg), pp, cc)
        for key in ("M2", "M4", "Fc", "epsilon_matched", "epsilon_mixed"):
            assert abs(cached[key] / direct[key] - 1.0) < 2e-13, (cut, key)
    assert len(cache._path_cache) == 1


@test
def highland_path_derivative_matches_centered_difference():
    p = np.array([1.0, 2.0, 6.0])
    x = np.array([2.8, 7.0, 11.57])
    h = 1e-5
    numeric = (theta0_highland(p, x + h) - theta0_highland(p, x - h)) / (2 * h)
    analytic = highland_path_derivative(p, x)
    assert np.allclose(analytic, numeric, rtol=2e-9, atol=1e-12)


@test
def multi_kink_transport_is_finite_and_changes_poca_geometry():
    one = simulate_fixed_node(2.0, 4000, (0.0, 0.0), seed=81, n_kinks=1)
    three = simulate_fixed_node(2.0, 4000, (0.0, 0.0), seed=81, n_kinks=3)
    for d in (one, three):
        assert np.all(np.isfinite(d[["dth_true", "dth_reco", "poca_z"]]))
    assert np.std(three.poca_z) > np.std(one.poca_z)


@test
def steering_modes_are_not_dead_config():
    assert nominal_target_offset_cm(1.0, "per_setting") == 0.0
    assert nominal_target_offset_cm(6.0, "per_setting") == 0.0
    assert abs(nominal_target_offset_cm(1.0, "none") - 5.85) < 1e-12
    assert abs(nominal_target_offset_cm(6.0, "none") - 0.975) < 1e-12


@test
def seed_entropy_unique_over_production_grid():
    vals = [
        seed_entropy(p, tuple(xy), seed=0) for p in MOMENTA for xy in raster_nodes()
    ]
    assert len(vals) == len(set(vals))


@test
def gradient_fractions_normalized():
    x = np.linspace(-11, 11, 101)
    f = momentum_fractions(x)
    assert np.allclose(f.sum(axis=1), 1.0)
    assert np.all(f >= 0)
    assert f[0, 0] > f[0, -1] and f[-1, -1] > f[-1, 0]


@test
def stopping_minimum_indirect_closure():
    d = validate_stopping_minima()
    for m, r in d.items():
        assert abs(r["rel"]) < 0.01, (m, r)


@test
def fiducial_mask_keeps_only_fully_contained_voxels():
    m = fiducial_voxel_mask()
    assert m.shape == (len(CENTERS), len(CENTERS), len(CENTERS))
    half = 0.5 * VOX_SIZE
    keep = np.where(np.abs(CENTERS) + half <= AL_HALF + 1e-12)[0]
    drop = np.where(np.abs(CENTERS) + half > AL_HALF + 1e-12)[0]
    assert keep.size > 0 and drop.size > 0
    assert np.all(m[np.ix_(keep, keep, keep)])
    assert not np.any(m[drop, :, :])
    assert not np.any(m[:, drop, :])
    assert not np.any(m[:, :, drop])


@test
def global_scale_optimizer_matches_closed_form():
    x = np.array([1.0, 2.0, 4.0, 8.0]).reshape(2, 2)
    y = np.array([0.8, 1.7, 3.9, 7.5]).reshape(2, 2)
    valid = np.ones_like(x, dtype=bool)
    c = optimal_global_scale(x, y, valid)
    expected = float(np.sum(x * x) / np.sum(x * y))
    assert abs(c - expected) < 1e-14
    a = 1.0 / c
    # The derivative of ||a*x-y||^2 vanishes at the optimum.
    assert abs(float(np.sum(x * (a * x - y)))) < 1e-13


@test
def event_mean_scalar_excludes_empty_reference_paths():
    data = {
        "p_meas": [1.0, 1.0],
        "p_true": [1.0, 1.0],
        "dth_reco": [0.01, 0.01],
        "dth_true": [0.01, 0.01],
        "xx0_ref_reco": [1.0, 0.0],
    }
    names = ("al_up", "cu_up", "pb", "cu_down", "al_down")
    for prefix in ("ref_reco", "ref_true"):
        for name in names:
            data[f"{prefix}_{name}"] = [1.0 if name == "al_up" else 0.0, 0.0]
    df = pd.DataFrame(data)

    class FakeCache:
        def arrays(self, p, segments, cut):
            p = np.asarray(p, float)
            hit = np.sum(np.asarray(segments, float), axis=1) > 0
            return {
                "theta_rms": np.where(hit, 0.02, 0.0),
                "epsilon_matched": np.where(hit, 0.1, 0.0),
                "epsilon_mixed": np.where(hit, 0.2, 0.0),
                "p_out": p,
            }

    weights, _ = build_weights(df, cache=FakeCache())
    assert np.array_equal(weights["valid_reference"], [True, False])
    assert np.isfinite(weights["eps_event"][0])
    assert np.isnan(weights["eps_event"][1])
    assert weights["eps_bar_event_mean"] == weights["eps_event"][0]



@test
def roi_guard_gap_is_disjoint_and_shrinks_cu_region():
    pb0, cu0 = roi_masks(0.0)
    pb1, cu1 = roi_masks(0.6)
    pb2, cu2 = roi_masks(1.2)
    assert np.array_equal(pb0, pb1) and np.array_equal(pb1, pb2)
    assert not np.any(pb0 & cu0)
    assert not np.any(pb1 & cu1)
    assert not np.any(pb2 & cu2)
    assert cu0.sum() > cu1.sum() > cu2.sum() > 0

def main():
    bad = 0
    for fn in TESTS:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            bad += 1
            print("FAIL", fn.__name__, repr(e))
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
