# MST Highland Validation — Simulation Package

Monte Carlo for *Highland Model Validation and Artifact Quantification in Muon
Scattering Tomography*. Implements Branch A (Highland residual decomposition)
and Branch B (tomographic artifact + correction) per the paper's Secs. 4.2–4.4.

## Layout

| File | Role |
|---|---|
| `config.py` | All constants. Single source of truth. Nothing hard-codes numbers elsewhere. |
| `kinematics.py` | β, Highland θ₀ (Eq. 1), θ_space = √2 θ₀. |
| `geometry.py` | Analytic ray tracer. `trace_true` (Al+Cu+Pb) and `trace_ref` (Al+Cu). |
| `moliere.py` | χ_c, χ_a, B, multi-material combination (App. A), truncated n≤2 CDF sampler with bucketed cache. |
| `simulate.py` | Event pipeline → one flat parquet table per momentum setting. |
| `branch_a.py` | Residual decomposition, bootstrap errors, ε̂_M(p) fit. |
| `branch_b.py` | Voxel imaging, SNR/CNR/edge-response, artifact maps, correction. |
| `tests.py` | Pre-flight validation. Run first; abort on failure. |
| `run_all.py` | Driver enforcing execution order. |

## Requirements

numpy, scipy, pandas, pyarrow (parquet). No Geant4, no ROOT.

## Run

```
python tests.py        # must pass
python run_all.py      # stages 0–5, writes to out/
```

Stages are individually runnable: `python simulate.py gauss`,
`python simulate.py moliere`, `python branch_a.py`, `python branch_b.py`.

Cost: 2×10⁶ Molière events + 2×10⁶ Gaussian control events. The Molière CDF
cache (bucketed on p and the areal-density triple, `P_CACHE_STEP` /
`X_CACHE_STEP` in `config.py`) is what makes this tractable — without it you
rebuild a CDF per event.

## Data flow

`simulate.py` writes `out/events_{moliere|gauss}_p{X.X}.parquet`. **Everything
downstream reads those tables. Nothing downstream re-simulates.** One event
table, four images, one residual decomposition.

Columns: `p_true`, `p_meas`, `theta_x/y`, `dth_true`, `dth_reco`, `xx0_true`,
`xx0_ref`, `t_Al/t_Cu/t_Pb`, `poca_x/y/z`, `pass_true`, `pass_reco`.

## The two geometries — do not swap them

- `trace_true` (Al+Cu+**Pb**) → Molière sampling, Branch A prediction (Eq. 14).
  Branch A tests the *model*, so it uses the true material distribution.
- `trace_ref` (Al+Cu, Pb volume filled with Cu) → weight denominator (Eq. 2).
  Excess scattering relative to this reference **is** the imaging signal.

Putting `trace_true` in the weight denominator makes ⟨w⟩ ≈ 1 everywhere and
normalizes away the contrast being imaged. Putting `trace_ref` in Branch A
corrupts the Highland test. `tests.py::test_ref_xx0_less_than_true` guards this.

## Branch A decomposition

Four residuals per momentum, ε = (θ_RMS − θ_pred)/θ_pred, θ_pred **always
per-event** (Eq. 14 — never at ⟨p⟩; Jensen, guarded by
`test_no_jensen_shortcut`):

| | θ_RMS from | p in pred | run |
|---|---|---|---|
| ε_full | reco angles | p_meas | Molière |
| ε_true-ang | true angles | p_meas | Molière |
| ε_true-p | true angles | p_true | Molière |
| ε_gauss | true angles | p_true | Gaussian (n=0) |

- noise = ε_full − ε_true-ang → should **grow** with p (fixed 1.18 mrad noise, θ₀ ∝ 1/p)
- p-resolution = ε_true-ang − ε_true-p → negative (Jensen), largest at 6 GeV/c
- truncation = ε_gauss → negative, largest at 1 GeV/c
- **ε_M = ε_true-p − ε_gauss** → the deliverable

`branch_a.sanity()` asserts each of those sign/trend expectations. If one
fails, do not proceed to Branch B.

Output: `out/branch_a.csv`, `out/eps_M_fit.npy` (log-linear ε_M(p) = a + b ln p).

## Known issues — resolve before quoting any number

0. **BLOCKING, and it is a physics question, not a code bug: does the
   momentum-position correlation Sec. 2.3 relies on actually exist?**

   The dipole steers the beam by 0.3·B·L/p × |z_magnet| = **5.85/p cm** at the
   target. The code applies that kick and never corrects it, so each setting
   lands somewhere different:

   | p (GeV/c) | δ (mrad) | Δx at target (cm) | median PoCA x (cm) |
   |---|---|---|---|
   | 1.0 | 90.0 | 5.85 | 5.85 |
   | 2.0 | 45.0 | 2.92 | 2.93 |
   | 3.5 | 25.7 | 1.67 | 1.65 |
   | 6.0 | 15.0 | 0.97 | 0.95 |

   That displacement **is** Sec. 2.3's mechanism — "muons of different momenta
   sample different regions of the target." But a real tagged beamline is
   retuned per momentum setting so every setting lands on the target. Measured
   spread of median PoCA x across the four settings:

   | beam | steer | spread |
   |---|---|---|
   | pencil | none | **4.90 cm** |
   | pencil | per_setting | **0.04 cm** |
   | uniform | none | 4.68 cm |
   | uniform | per_setting | 0.13 cm |

   Retuning collapses the correlation to well under one voxel (0.6 cm). The
   artifact is then a near-uniform rescaling and the paper's
   spatially-structured claim has no mechanism. `STEER_COMPENSATION` in
   `config.py` makes this explicit; `test_momentum_position_correlation_exists`
   asserts the chosen configuration can support the claim.

   **Either the paper argues the beamline is not retuned — which needs
   justifying, since it is not how such facilities are run — or Sec. 2.3's
   central claim needs restating.** A cosmic-ray exposure, where the momentum
   spectrum and the angular distribution are genuinely correlated at every
   point, is the setting where this mechanism is natural; a four-setting
   tagged-beam exposure may not be.

   Related, and separable: with σ_xy = 1.0 cm the 95% PoCA x-span is 7.7 cm
   against a 15 cm Cu block, so there is no tomogram to reconstruct regardless.
   `BEAM_MODE='uniform'` fixes coverage. This was surfacing as a `curve_fit`
   `maxfev` crash in `edge_response`, the last link in the chain: unlit ROIs →
   profile with two edges plus a zero floor → fit cannot converge.

1. ~~`test_tail_asymptote` fails~~ **RESOLVED — was a convention mismatch,
   not a transcription error.**

1. ~~`test_tail_asymptote` fails~~ **RESOLVED — was a convention mismatch,
   not a transcription error.**

   Bethe (1953) Table II tabulates f⁽ⁿ⁾ for the **space-angle** distribution,
   normalized ∫₀^∞[…]η dη = 1, for which f₀(η) = 2e^(−η²) and f₁(0)=0.8456,
   f₂(0)=2.4929. Appendix A uses the **projected** (1D) normalization,
   f₀(η) = e^(−η²)/√π. The original code converted f₀ but spliced Bethe's
   unconverted space-angle f₁/f₂ into the projected formula. The space→projected
   map is an integral transform, so no rescaling of the tables could fix it.

   Fixed by computing the projected f⁽ⁿ⁾ directly from the generating integral,
   replacing the Hankel kernel J₀ with the Fourier kernel cos:

   f_p⁽ⁿ⁾(η) = (1/π)∫₀^∞ cos(ηu) e^(−u²/4) [(u²/4)ln(u²/4)]ⁿ/n! du

   Tables are gone; values are computed at import and cached to
   `_fn_projected.npz`. Self-check at n=0 reproduces e^(−η²)/√π to 9 digits.

   Two consequences worth stating in the paper:
   - The **projected** tail goes as θ⁻³, not θ⁻⁴. θ⁻⁴ is the space-angle
     (Rutherford) power; projecting a 2D θ⁻⁴ tail onto one axis gives θ⁻³.
     The original test asserted −4 and was itself wrong.
   - f₁ → 1/(2η³) exactly, so F(θ) → χ_c²/(2θ³). **B cancels**, and the tail
     is fixed absolutely by χ_c² alone — the single-scattering limit with no
     free parameter. `test_single_scatter_limit` checks this to 5%; measured
     ratios are 1.010–1.034 across four (p, path) points spanning B = 15.0–16.9.
     This is the strongest available check on the module: it catches a wrong f₁
     normalization, a wrong χ_c, and a wrong prefactor, none of which a slope
     test sees.

   The low-momentum case is not testable this way: at p = 1 GeV/c the 400 mrad
   grid reaches only η ≈ 5.4, short of the asymptotic regime. The test skips
   it explicitly and requires ≥3 of the remaining cases to pass.

2. **`biased` and `corrected` images are the same operation.** Sec. 4.3 says
   θ₀ → θ₀(1+ε_M); Eq. (13) says the same. So `weight_biased(df, eps_fn)` and
   `weight(df, eps_fn=eps_fn)` return identical arrays. The code assumes
   `nominal` (Highland denominator, Molière-sampled events) **already contains
   the artifact** and Eq. (13) removes it. If instead the intent is to inject
   (1+ε_M)⁻² relative to a corrected baseline, the sign in `weight_biased`
   flips. Decide, then fix the paper's wording too.

3. **Cu ROI overlaps the Pb ROI.** Closest approach = √13 − 2 = 1.61 cm <
   3.75 cm. `roi_masks()` excludes Pb voxels from the Cu ROI by mask
   difference. The paper's definition is ambiguous — state the convention.

4. **PoCA-truth vertex approximated at z = 0** rather than the exact in-target
   path midpoint. <1 mm at 2 mrad divergence, but it is an approximation.

5. **Energy loss is not modeled** (by design, Sec. 4.1). ε̂_M(p) is therefore
   not directly transferable to real data at p ≲ 2 GeV/c without an explicit
   energy-loss correction. Do not present it as such.

6. **σ_δ is emergent, not injected.** It arises from smearing hits 1–4, so the
   stations-3/4 correlation between p_meas and Δθ_space (Sec. 3.4) is present
   by construction. `test_sigma_delta_emergent` checks it reproduces
   2σ_hit/Δz ≈ 1.3 mrad. Do not "fix" this by adding a Gaussian to δ.

7. **The 200 mrad cut is applied at event selection, not in the sampler.** The
   CDF grid extends to 400 mrad. ε_M is by construction *cut-dependent* — it
   is a property of the model–acceptance pair, which is the operationally
   relevant quantity.

## Reproducibility

Seeds derive from `SEED_BASE` in `config.py` plus the momentum setting. Same
seed set is used for the Molière and Gaussian runs, so the control differs only
in the sampler.