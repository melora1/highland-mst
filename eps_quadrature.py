#!/usr/bin/env python3
"""
eps_quadrature.py  --  deterministic Highland mismatch eps_M(p, path) by 2D
acceptance quadrature, built on this repo's own moliere.py (chi_c2/chi_a2/B,
projected f0+f1/B+f2/B^2). Manuscript Sec. 2.3: eps_M is a DETERMINISTIC
functional of the Moliere distribution and the acceptance, evaluated by
quadrature -- not fitted from a sampled Monte Carlo run (that is what
branch_a.py's eps_M_fit does, and it answers a different question: it
decomposes RECONSTRUCTION noise/resolution/truncation, none of which belong
in the imaging weight of Eq. (2)/(7)).

This module supplies the single number the weight actually needs:

    eps_M(p, path) = theta_RMS(cut; p, path) / theta_space_highland(p, xX0) - 1

with theta_RMS computed by 2D disc quadrature over the SAME acceptance the
weight applies (theta_cut = THETA_CUT), using the path's own chi_c2, chi_a2, B
-- i.e. per-event material composition, not a momentum-only marginal.

Reproduces the manuscript's axial eps_M table to rounding (see verify() at
the bottom): +5.1/+10.8/+14.2/+17.1 % at 1.0/2.0/3.5/6.0 GeV/c.

Cached per (path signature, momentum bucket) exactly like moliere.py's CDF
cache, since only a handful of distinct (chi_c2, chi_a2) pairs occur in the
simulated geometry.
"""
import math
from functools import lru_cache

import numpy as np

import moliere as ml
from config import MATERIALS, MAT_ORDER, P_CACHE_STEP, THETA_CUT, X_CACHE_STEP
from kinematics import theta_space_highland

_DISC_N = 900          # grid points per axis for the 2D disc quadrature


def _theta_rms_disc(chi_c2, chi_a2, B, cut=THETA_CUT, n=_DISC_N):
    """theta_RMS within the disc theta_x^2+theta_y^2<cut^2, using the SAME
    projected density F(theta) = (1/(chi_c sqrt B)) sum f^(n)/B^n that
    moliere.py's sampler CDF is built from (n<=2)."""
    scale = math.sqrt(chi_c2 * B)
    th = np.linspace(-cut, cut, n)
    eta = th / scale
    F = ml.f0(eta) + ml.f1(eta) / B + ml.f2(eta) / B ** 2
    F = np.clip(F, 0.0, None)
    TX, TY = np.meshgrid(th, th)
    inside = (TX ** 2 + TY ** 2) < cut ** 2
    W = F[:, None] * F[None, :]
    num = np.sum(((TX ** 2 + TY ** 2) * W)[inside])
    den = np.sum(W[inside])
    return math.sqrt(num / den)


@lru_cache(maxsize=8192)
def _eps_M_bucketed(p_key, X_al_key, X_cu_key, X_pb_key):
    p = p_key * P_CACHE_STEP
    X_al = X_al_key * X_CACHE_STEP
    X_cu = X_cu_key * X_CACHE_STEP
    X_pb = X_pb_key * X_CACHE_STEP
    chi_c2, chi_a2 = ml.combine_path(X_al, X_cu, X_pb, p)
    B = ml.solve_B(chi_c2, chi_a2)
    trms = _theta_rms_disc(chi_c2, chi_a2, B)
    xX0 = (X_al / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
           + X_cu / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"]
           + X_pb / MATERIALS["Pb"]["rho"] / MATERIALS["Pb"]["X0"])
    tspace = float(theta_space_highland(p, xX0))
    return trms / tspace - 1.0


def eps_M(p, X_al, X_cu, X_pb):
    """Vectorised: arrays in, eps_M array out. Buckets to the same
    P_CACHE_STEP/X_CACHE_STEP grid moliere.py's sampler uses, so the weight's
    eps_M and the sampler's distribution are evaluated on matching supports."""
    p = np.atleast_1d(np.asarray(p, float))
    X_al = np.atleast_1d(np.asarray(X_al, float))
    X_cu = np.atleast_1d(np.asarray(X_cu, float))
    X_pb = np.atleast_1d(np.asarray(X_pb, float))
    out = np.empty(p.size)
    for i in range(p.size):
        out[i] = _eps_M_bucketed(
            round(p[i] / P_CACHE_STEP),
            round(X_al[i] / X_CACHE_STEP),
            round(X_cu[i] / X_CACHE_STEP),
            round(X_pb[i] / X_CACHE_STEP),
        )
    return out


def theta_space_corrected(p, X_al, X_cu, X_pb):
    """(1+eps_M) * theta_space_highland -- Eq. (7)/(13), evaluated per event
    at the SUPPLIED path (caller passes the reference path for I_p/I_nom, the
    true path for I_ideal -- see results_pipeline.py)."""
    xX0 = (X_al / MATERIALS["Al"]["rho"] / MATERIALS["Al"]["X0"]
           + X_cu / MATERIALS["Cu"]["rho"] / MATERIALS["Cu"]["X0"]
           + X_pb / MATERIALS["Pb"]["rho"] / MATERIALS["Pb"]["X0"])
    ts = theta_space_highland(p, xX0)
    return ts * (1.0 + eps_M(p, X_al, X_cu, X_pb))


def eps_M_marginal(p, xX0_axial=10 / 8.90 + 15 / 1.44):
    """Momentum-only marginal at the axial reference path (Al 10cm + Cu 15cm,
    no Pb) -- this is theta_space's calibration for I_p (Eq. 7): per-momentum
    only, evaluated at the REFERENCE geometry, not the true per-event path."""
    X_al = MATERIALS["Al"]["rho"] * 10.0
    X_cu = MATERIALS["Cu"]["rho"] * 15.0
    return eps_M(np.atleast_1d(p), np.full_like(np.atleast_1d(p), X_al, float),
                np.full_like(np.atleast_1d(p), X_cu, float),
                np.zeros_like(np.atleast_1d(p)))


def verify():
    """Reproduce the manuscript's axial eps_M table (Table, Sec. 2.3)."""
    paper = {1.0: 5.1, 2.0: 10.8, 3.5: 14.2, 6.0: 17.1}
    print(f"{'p':>5} {'eps_M/%':>9} {'paper/%':>9}")
    for p in (1.0, 2.0, 3.5, 6.0):
        e = eps_M_marginal(p)[0] * 100
        print(f"{p:5.1f} {e:9.2f} {paper[p]:9.1f}")


if __name__ == "__main__":
    verify()
