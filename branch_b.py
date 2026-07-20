"""Branch B: PoCA imaging, artifact characterisation, correction (Secs 4.3-4.4).

Four images, all from the COMBINED four-momentum sample (2e6 events):

  unweighted : count
  nominal    : w = (dth_reco / theta_space(p_meas, xx0_ref))^2      Eq. (2)
               -- this is the OBSERVED image and ALREADY CONTAINS the artifact,
                  because the events were sampled from Moliere while the
                  denominator uses uncorrected Highland.
  biased     : same, with theta0 -> theta0 * (1 + eps_M(p_meas))
               -- explicit injection, used only as an analytic cross-check
                  that the artifact scales as (1+eps_M)^-2.
  corrected  : same, with theta_space -> (1 + eps_M(p_meas)) * theta_space
               Eq. (13).  [Note: 'biased' and 'corrected' are the same
               operation with opposite intent; keep them distinct in the
               bookkeeping.]

Cu ROI: the r=3.75 cm central cylinder OVERLAPS the Pb cylinder at (3,2)
with r=2 (closest approach = sqrt(13) - 2 = 1.61 cm < 3.75). Pb-ROI voxels
are therefore EXCLUDED from the Cu ROI by mask difference. The paper's
definition as written is ambiguous on this point -- state the convention.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc

from config import (CU_ROI_R, CU_ROI_ZHALF, MIN_VOX_COUNT, MOMENTA, N_VOX,
                    OUT_DIR, PB_CX, PB_CY, PB_ROI_R, PB_ROI_ZHALF, VOX_HALF,
                    VOX_SIZE)
from kinematics import theta_space_highland

EDGE_Z_BAND = 5.0   # cm, half-width of the z band the radial profile averages

EDGES = np.linspace(-VOX_HALF, VOX_HALF, N_VOX + 1)
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])


# ------------------------------------------------------------------ data
def load_combined(tag="moliere", momenta=MOMENTA):
    dfs = [pd.read_parquet(os.path.join(OUT_DIR, f"events_{tag}_p{p:.1f}.parquet"))
           for p in momenta]
    df = pd.concat(dfs, ignore_index=True)
    return df[df.pass_reco].copy()


# ------------------------------------------------------------------ weights
def weight(df, eps_fn=None):
    """Eq. (2), with the REFERENCE geometry in the denominator.

    eps_fn : callable p -> eps, or None. If given, the denominator is
             scaled by (1 + eps(p_meas)).
    """
    th_pred = theta_space_highland(df.p_meas.values, df.xx0_ref.values)
    if eps_fn is not None:
        th_pred = th_pred * (1.0 + eps_fn(df.p_meas.values))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (df.dth_reco.values / th_pred) ** 2
    return np.where(np.isfinite(w) & (th_pred > 0), w, 0.0)


# ------------------------------------------------------------------ voxelise
def voxelise(df, w=None):
    """Returns (sum_w, sum_w2, counts) on the 50^3 grid."""
    sample = np.stack([df.poca_x.values, df.poca_y.values, df.poca_z.values], 1)
    bins = (EDGES, EDGES, EDGES)
    counts, _ = np.histogramdd(sample, bins=bins)
    if w is None:
        return counts, counts, counts
    sw, _ = np.histogramdd(sample, bins=bins, weights=w)
    sw2, _ = np.histogramdd(sample, bins=bins, weights=w ** 2)
    return sw, sw2, counts


# ------------------------------------------------------------------ ROIs
def _grid_xyz():
    X, Y, Z = np.meshgrid(CENTERS, CENTERS, CENTERS, indexing="ij")
    return X, Y, Z


def roi_masks():
    X, Y, Z = _grid_xyz()
    r_pb = np.hypot(X - PB_CX, Y - PB_CY)
    r_cu = np.hypot(X, Y)
    pb = (r_pb <= PB_ROI_R) & (np.abs(Z) <= PB_ROI_ZHALF)
    cu = (r_cu <= CU_ROI_R) & (np.abs(Z) <= CU_ROI_ZHALF) & (~pb)
    return pb, cu


# ------------------------------------------------------------------ metrics
def metrics(img, sw2=None, counts=None, n_boot=200, rng=None):
    """SNR_Pb, CNR, and their bootstrap errors (voxel-level bootstrap)."""
    rng = rng or np.random.default_rng(7)
    pb, cu = roi_masks()
    vpb = img[pb]
    vcu = img[cu]

    def _m(a, b):
        snr = a.mean() / a.std(ddof=1)
        cnr = (a.mean() - b.mean()) / b.std(ddof=1)
        return snr, cnr

    snr, cnr = _m(vpb, vcu)

    snrs = np.empty(n_boot)
    cnrs = np.empty(n_boot)
    for i in range(n_boot):
        a = vpb[rng.integers(0, vpb.size, vpb.size)]
        b = vcu[rng.integers(0, vcu.size, vcu.size)]
        snrs[i], cnrs[i] = _m(a, b)
    out = dict(SNR_Pb=snr, SNR_Pb_err=snrs.std(ddof=1),
               CNR=cnr, CNR_err=cnrs.std(ddof=1),
               n_pb_vox=int(pb.sum()), n_cu_vox=int(cu.sum()))
    if counts is not None:
        # An unlit ROI makes SNR/CNR meaningless; surface it rather than
        # quietly reporting a number computed over mostly-zero voxels.
        out["pb_empty_frac"] = float((counts[pb] == 0).mean())
        out["cu_empty_frac"] = float((counts[cu] == 0).mean())
    return out


def _erf_edge(x, A, C, x0, sigma):
    return C + 0.5 * A * erfc((x - x0) / (np.sqrt(2.0) * sigma))


def edge_response(img, counts=None):
    """Edge width at the Pb--Cu boundary, from an azimuthally averaged radial
    profile about the inclusion axis.

    The original implementation took a 1-D line along +x from the inclusion
    centre over 12 voxels. That window runs from x=3.3 to x=9.9 and therefore
    contains TWO edges -- Pb->Cu at x=5.0 and Cu->Al at x=7.5 -- plus a floor
    of exact zeros beyond the illuminated region. A single erfc cannot fit
    that, and curve_fit exhausts maxfev.

    Instead: profile in radius about (PB_CX, PB_CY), averaged over azimuth and
    over |z| <= EDGE_Z_BAND. The Pb axis sits 3.0 cm from the Cu block centre
    and the Cu half-width is 7.5 cm, so radii out to CU_HALF - hypot(PB_CX,
    PB_CY) stay inside copper in every direction -- one edge, one material
    either side, and every azimuth contributing statistics.

    Returns sigma_PSF = nan (with a reason) rather than raising: a failed
    edge fit must not destroy an eight-hour run at the metrics stage.
    """
    from config import CU_HALF

    r_max = CU_HALF - np.hypot(PB_CX, PB_CY)      # 3.89 cm: stays inside Cu
    X, Y, Z = _grid_xyz()
    r = np.hypot(X - PB_CX, Y - PB_CY)
    band = np.abs(Z) <= EDGE_Z_BAND

    edges = np.arange(0.0, r_max + 1e-9, VOX_SIZE)
    rc, val, wt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & band
        if counts is not None:
            m = m & (counts > 0)
        if m.sum() < 5:
            continue
        rc.append(0.5 * (lo + hi))
        val.append(img[m].mean())
        wt.append(m.sum())
    rc, val = np.asarray(rc), np.asarray(val)

    fail = dict(sigma_PSF=np.nan, sigma_PSF_err=np.nan, edge_10_90=np.nan)
    if rc.size < 6:
        return dict(fail, edge_fit_status=f"only {rc.size} usable radial bins")

    inner = val[rc < PB_ROI_R].mean() if (rc < PB_ROI_R).any() else val[0]
    outer = val[rc > PB_ROI_R].mean() if (rc > PB_ROI_R).any() else val[-1]
    if not np.isfinite(inner - outer) or abs(inner - outer) < 1e-12:
        return dict(fail, edge_fit_status="no contrast across the boundary")

    p0 = [inner - outer, outer, PB_ROI_R, 0.5]
    bounds = ([-np.inf, -np.inf, 0.5 * PB_ROI_R, 0.05],
              [np.inf, np.inf, 1.5 * PB_ROI_R, 0.5 * r_max])
    try:
        popt, pcov = curve_fit(_erf_edge, rc, val, p0=p0,
                               bounds=bounds, maxfev=20000)
    except Exception as exc:
        return dict(fail, edge_fit_status=f"{type(exc).__name__}: {exc}")

    sig = abs(popt[3])
    err = float(np.sqrt(abs(pcov[3, 3])))
    return dict(sigma_PSF=sig, sigma_PSF_err=err, edge_10_90=2.56 * sig,
                edge_fit_status="ok")


# ------------------------------------------------------------------ artifact map
def artifact_map(img_a, img_b, counts):
    """(a - b)/b, masked to voxels with adequate statistics."""
    mask = counts >= MIN_VOX_COUNT
    d = np.full(img_a.shape, np.nan)
    ok = mask & (np.abs(img_b) > 0)
    d[ok] = (img_a[ok] - img_b[ok]) / img_b[ok]
    return d, mask


def mean_momentum_map(df):
    """Mean p_meas per voxel -- the axis against which the artifact must
    correlate if the Sec. 2.3 mechanism is real."""
    sample = np.stack([df.poca_x.values, df.poca_y.values, df.poca_z.values], 1)
    bins = (EDGES, EDGES, EDGES)
    n, _ = np.histogramdd(sample, bins=bins)
    sp, _ = np.histogramdd(sample, bins=bins, weights=df.p_meas.values)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, sp / np.maximum(n, 1), np.nan), n


# ------------------------------------------------------------------ driver
def run_branch_b():
    from branch_a import eps_M_of

    a, b = np.load(os.path.join(OUT_DIR, "eps_M_fit.npy"))
    eps_fn = lambda p: eps_M_of(p, a, b)

    df = load_combined("moliere")

    w_nom = weight(df, eps_fn=None)
    w_bias = weight_biased(df, eps_fn)          # Sec. 4.3 injection
    w_corr = weight(df, eps_fn=eps_fn)          # Eq. (13)

    img_unw, _, counts = voxelise(df, None)
    img_nom, sw2_nom, _ = voxelise(df, w_nom)
    img_bias, _, _ = voxelise(df, w_bias)
    img_corr, sw2_corr, _ = voxelise(df, w_corr)

    rows = []
    for name, img in [("unweighted", img_unw), ("nominal", img_nom),
                      ("biased", img_bias), ("corrected", img_corr)]:
        m = metrics(img, counts=counts)
        m.update(edge_response(img, counts=counts))
        m["image"] = name
        rows.append(m)
    res = pd.DataFrame(rows)[["image", "SNR_Pb", "SNR_Pb_err", "CNR", "CNR_err",
                              "sigma_PSF", "sigma_PSF_err", "edge_10_90",
                              "pb_empty_frac", "cu_empty_frac",
                              "edge_fit_status"]]
    res.to_csv(os.path.join(OUT_DIR, "branch_b_metrics.csv"), index=False)

    d_bias, mask = artifact_map(img_bias, img_nom, counts)
    d_corr, _ = artifact_map(img_corr, img_nom, counts)
    pmap, _ = mean_momentum_map(df)

    np.savez_compressed(
        os.path.join(OUT_DIR, "images.npz"),
        unweighted=img_unw, nominal=img_nom, biased=img_bias,
        corrected=img_corr, counts=counts, sw2_nom=sw2_nom,
        sw2_corr=sw2_corr, d_bias=d_bias, d_corr=d_corr, pmap=pmap,
        mask=mask, centers=CENTERS,
    )

    summary = dict(
        artifact_rms_biased=float(np.nanstd(d_bias)),
        artifact_rms_corrected=float(np.nanstd(d_corr)),
    )
    pd.Series(summary).to_csv(os.path.join(OUT_DIR, "artifact_summary.csv"))
    return res, summary


def weight_biased(df, eps_fn):
    """Explicit artifact injection: theta0 -> theta0 * (1 + eps_M(p_meas))
    in the weight denominator, per Sec. 4.3. Weight of every event is then
    rescaled by (1 + eps_M)^-2 relative to nominal -- the analytic
    cross-check."""
    th = theta_space_highland(df.p_meas.values, df.xx0_ref.values)
    th = th * (1.0 + eps_fn(df.p_meas.values))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (df.dth_reco.values / th) ** 2
    return np.where(np.isfinite(w) & (th > 0), w, 0.0)


# per-setting diagnostic images (momentum-driven bias -> near-global rescale)
def run_per_setting():
    from branch_a import eps_M_of
    a, b = np.load(os.path.join(OUT_DIR, "eps_M_fit.npy"))
    eps_fn = lambda p: eps_M_of(p, a, b)
    out = {}
    for p in MOMENTA:
        df = pd.read_parquet(os.path.join(OUT_DIR, f"events_moliere_p{p:.1f}.parquet"))
        df = df[df.pass_reco]
        img, _, cnt = voxelise(df, weight(df, None))
        imgc, _, _ = voxelise(df, weight(df, eps_fn))
        out[f"p{p:.1f}_nominal"] = img
        out[f"p{p:.1f}_corrected"] = imgc
        out[f"p{p:.1f}_counts"] = cnt
    np.savez_compressed(os.path.join(OUT_DIR, "images_per_setting.npz"), **out)


if __name__ == "__main__":
    r, s = run_branch_b()
    pd.set_option("display.width", 200)
    print(r.to_string(index=False))
    print(s)
    run_per_setting()