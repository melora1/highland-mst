"""Branch A: Highland validation and residual decomposition (Sec. 4.2).

Four residuals per momentum setting, all using Eq. (3):

    eps = (theta_RMS - theta_pred) / theta_pred

    eps_full     : reco angles, p_meas   (Moliere run)
    eps_true_ang : true angles, p_meas   (Moliere run)
    eps_true_p   : true angles, p_true   (Moliere run)
    eps_gauss    : true angles, p_true   (Gaussian n=0 control run)

Decomposition:
    noise      = eps_full     - eps_true_ang
    p_res      = eps_true_ang - eps_true_p
    truncation = eps_gauss
    eps_M      = eps_true_p   - eps_gauss

theta_pred is ALWAYS per-event (Eq. 14), through the TRUE geometry
(xx0_true). Never evaluate at <p> -- Jensen (Sec. 4.2).
"""

import os

import numpy as np
import pandas as pd

from config import MOMENTA, OUT_DIR
from kinematics import theta0_highland

N_BOOT = 500
RNG = np.random.default_rng(12345)


def theta_pred(p, xx0):
    """Eq. (14): sqrt( mean( 2 * theta0^2(p_i, x/X0_i) ) )."""
    t0 = theta0_highland(p, xx0)
    return np.sqrt(np.mean(2.0 * t0 ** 2))


def theta_rms(dth):
    return np.sqrt(np.mean(dth ** 2))


def residual(dth, p, xx0):
    return (theta_rms(dth) - theta_pred(p, xx0)) / theta_pred(p, xx0)


def residual_boot(dth, p, xx0, n_boot=N_BOOT):
    """Bootstrap the residual (resample events, recompute both terms)."""
    n = dth.size
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        out[b] = residual(dth[idx], p[idx], xx0[idx])
    return float(np.std(out, ddof=1))


def analytic_floor(n):
    """Gaussian second-moment expectation: sigma(theta_RMS)/theta_RMS = 1/(2 sqrt N)."""
    return 1.0 / (2.0 * np.sqrt(n))


def _load(tag, p):
    return pd.read_parquet(os.path.join(OUT_DIR, f"events_{tag}_p{p:.1f}.parquet"))


def branch_a(momenta=MOMENTA):
    rows = []
    for p in momenta:
        mol = _load("moliere", p)
        gau = _load("gauss", p)

        # cut applied on the SAME observable used for theta_RMS
        m_reco = mol[mol.pass_reco]
        m_true = mol[mol.pass_true]
        g_true = gau[gau.pass_true]

        eps_full = residual(m_reco.dth_reco.values,
                            m_reco.p_meas.values, m_reco.xx0_true.values)
        eps_true_ang = residual(m_true.dth_true.values,
                                m_true.p_meas.values, m_true.xx0_true.values)
        eps_true_p = residual(m_true.dth_true.values,
                              m_true.p_true.values, m_true.xx0_true.values)
        eps_gauss = residual(g_true.dth_true.values,
                             g_true.p_true.values, g_true.xx0_true.values)

        e_full = residual_boot(m_reco.dth_reco.values,
                               m_reco.p_meas.values, m_reco.xx0_true.values)
        e_tp = residual_boot(m_true.dth_true.values,
                             m_true.p_true.values, m_true.xx0_true.values)
        e_g = residual_boot(g_true.dth_true.values,
                            g_true.p_true.values, g_true.xx0_true.values)

        eps_M = eps_true_p - eps_gauss
        err_M = np.hypot(e_tp, e_g)   # bootstrap samples are independent runs

        rows.append(dict(
            p=p,
            n_pass=len(m_reco),
            eps_full=eps_full, eps_full_err=e_full,
            eps_true_ang=eps_true_ang,
            eps_true_p=eps_true_p, eps_true_p_err=e_tp,
            eps_gauss=eps_gauss, eps_gauss_err=e_g,
            noise=eps_full - eps_true_ang,
            p_res=eps_true_ang - eps_true_p,
            truncation=eps_gauss,
            eps_M=eps_M, eps_M_err=err_M,
            analytic_floor=analytic_floor(len(m_reco)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "branch_a.csv"), index=False)
    return df


def fit_eps_M(df):
    """Log-linear interpolation model: eps_M(p) = a + b ln p."""
    x = np.log(df.p.values)
    y = df.eps_M.values
    w = 1.0 / np.maximum(df.eps_M_err.values, 1e-9) ** 2
    b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
    np.save(os.path.join(OUT_DIR, "eps_M_fit.npy"), np.array([a, b]))
    return float(a), float(b)


def eps_M_of(p, a, b):
    return a + b * np.log(np.asarray(p, float))


def sanity(df):
    """Sign/magnitude checks from the plan (Sec. 4)."""
    msgs = []
    if not (df.noise.iloc[-1] > df.noise.iloc[0]):
        msgs.append("FAIL: noise term should GROW with momentum")
    if not (abs(df.truncation.iloc[0]) > abs(df.truncation.iloc[-1])):
        msgs.append("FAIL: truncation bias should be largest at LOW p")
    if not (df.truncation < 0).all():
        msgs.append("FAIL: truncation bias should be negative")
    if not (df.p_res <= 0).all():
        msgs.append("FAIL: momentum-resolution term should be negative (Jensen)")
    return msgs or ["all sanity checks passed"]


if __name__ == "__main__":
    d = branch_a()
    pd.set_option("display.width", 200)
    print(d.to_string(index=False))
    print()
    print("eps_M(p) = a + b ln p ->", fit_eps_M(d))
    for m in sanity(d):
        print(m)
