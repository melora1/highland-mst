# highland-mst

Code and manuscript for *The Highland Constant and the Divergent Second
Moment: Acceptance Dependence of Momentum-Informed Muon Scattering
Tomography*.

Momentum-informed PoCA muon tomography weights each event by the inverse
square of the Highland multiple-Coulomb-scattering prediction. Highland fits
the Gaussian **core** of the Molière distribution, but the weight needs the
**second moment**, which for a single-scatter (θ⁻³) tail grows logarithmically
with the angular acceptance. The mismatch, εM, is momentum- and
path-length-dependent. This repo computes εM deterministically by quadrature,
derives an acceptance-matched estimator and an optimal (adaptive) acceptance,
and propagates the effect through a controlled beamline simulation.

## Layout

### Physics core (event generation, geometry, distributions)

| File | Role |
|---|---|
| `config.py` | All constants: materials, geometry, spectrometer, beam profile, ROIs. Single source of truth — nothing hard-codes numbers elsewhere. |
| `kinematics.py` | β, Highland θ₀ (Eq. 1), θ_space = √2 θ₀. |
| `geometry.py` | Analytic ray tracer. `trace_true` (Al+Cu+Pb) vs. `trace_ref` (Al+Cu, Pb→Cu) — do not swap them; see the module docstring. |
| `moliere.py` | χ_c, χ_a, B (Appendix A), the **radial** Hankel-generated Molière Φ⁽ⁿ⁾ used for quantitative sampling/calibration, plus projected f_p⁽ⁿ⁾ retained only as an appendix diagnostic. The production sampler draws the radial magnitude and a uniform azimuth rather than independent projected marginals. |
| `simulate.py` | Event pipeline: beam → dipole momentum tag → Molière scatter at the target → downstream tracking → PoCA. Writes one flat table per momentum setting. |

### Calibration and detector-level diagnostics

| File | Role |
|---|---|
| `eps_quadrature.py` | **Deterministic radial** εM(p, path), retained fraction, M₂, M₄, and model/path-specific acceptance optimization from the non-factorized Molière density. At the fixed 200 mrad axial reference path, the corrected radial diagnostics are approximately +4.07/+9.77/+13.15/+16.15 % at 1/2/3.5/6 GeV/c. |
| `results_pipeline.py` | Detector-level diagnostic pipeline. Builds I_nom, I_p, I_Q, I_ideal, I_const, and adaptive-I_Q images, runs SNR/CNR/edge/speckle diagnostics, and computes the I_nom−I_Q non-rescaling decomposition. The current manuscript is theory-only; these simulation outputs are diagnostics, not manuscript results by default. |

### Original validation suite (still useful; not on the path that produces the paper's numbers anymore)

| File | Role |
|---|---|
| `branch_a.py` | Residual decomposition (noise / momentum-resolution / truncation / εM) via Monte Carlo + bootstrap. Useful as a sampler-closure and reconstruction-noise diagnostic; its εM fit is no longer what calibrates the imaging weight (see `eps_quadrature.py`). |
| `branch_b.py` | Original imaging driver (nominal/biased/corrected images). Superseded by `results_pipeline.py` for the paper's numbers, but its ROI/metrics/edge-response helpers (`roi_masks`, `metrics`, `edge_response`, `voxelise`) are imported directly by `results_pipeline.py` — keep it in the repo. |
| `run_all.py` | Original end-to-end driver (tests → Gaussian control → Molière run → Branch A → Branch B). Runs under `config.py`'s defaults; optional, diagnostic only. |
| `tests.py` | Pre-flight validation. **Run this before any production run.** Includes radial normalization/tail/isotropy/sampler-closure tests and separate checks for the uncompensated diagnostic steering and the per-setting production steering. |

### Independent checks (literature + closed-form arithmetic, not simulation)

| File | Role |
|---|---|
| `verify_arithmetic.py` | Checks every closed-form number quoted in the manuscript (Highland θ₀, dipole deflections, detectability budget, adaptive-cut table, etc.) against hand computation. Pure stdlib, no dependency on the rest of the repo. |
| `verify_scaling.py` | Independently checks the two divergence laws the argument rests on — the 2nd-moment log law and the 4th-moment quadratic law — using the screened θ⁻³ tail. Pure stdlib. |
| `geant4_compare.py` | Analysis harness for the Sec. 3 Geant4 cross-check. Geant4 itself runs separately (physics-list/macro notes are in the file's header); this script reads its output dump and compares to `eps_quadrature.py`/the quadrature RMS against the published Urban/Wentzel-VI model spread. |

### Manuscript

| File | Role |
|---|---|
| `highland_mst.tex` | Manuscript source. Inline `%` comments carry per-claim verification tags (`[PDF-OK]`, `[WEB-OK]`, `[CODE-REPRODUCED]`, `[UNVERIFIED-CODE]` — see legend below). |
| `references.bib` | Bibliography. Every entry verified against a source PDF or a primary online source; see per-entry comments. |

## Requirements

```bash
pip install numpy scipy pandas pyarrow
```
LaTeX with `bibtex` for the manuscript. No Geant4, no ROOT (Geant4 is only needed if you're doing the Sec. 3 cross-check).

## Run order

```bash
# 1. Validate — must be all-PASS before anything else
python3 tests.py

# 2. Quadrature self-check (seconds, no simulation)
python3 eps_quadrature.py

# 3. Smoke-test the results pipeline (catches integration bugs cheaply)
python3 results_pipeline.py --n 5000

# 4. Convergence scan — check stability before trusting the numbers
python3 results_pipeline.py --n 50000
python3 results_pipeline.py --n 150000

# 5. Manuscript-scale production run (2e6 events total; this is the expensive one)
python3 results_pipeline.py --n 500000
```

Each `results_pipeline.py` run **overwrites** `out/results_images.npz`,
`out/results_metrics.csv`, `out/results_artifact.csv` — rename between runs
if you want to keep intermediate scan points for comparison.

Optional, separate diagnostic (original Branch A/B flow, different
config/εM source, don't mix its numbers into the paper's Results section):
```bash
python3 run_all.py
```

## IMPORTANT: steering is intentionally tested in two configurations

`config.py` deliberately keeps `STEER_COMPENSATION='none'` as the
diagnostic/legacy file default.  In that uncompensated configuration the
dipole produces a large setting-dependent transverse displacement, and
`test_uncompensated_configuration_has_momentum_position_correlation` verifies
that the diagnostic mechanism is actually present.

`results_pipeline.py` is different by design: it forces
`STEER_COMPENSATION='per_setting'` **before importing `simulate.py`**, matching
a beamline re-steered for each nominal momentum setting.
`test_production_configuration_has_no_gross_setting_steering` reproduces that
import-time configuration and verifies that the median setting-to-setting
PoCA-x spread remains below one voxel.  The pipeline also prints the same
production check before an expensive run.

A green test suite therefore validates both intended regimes rather than
conflating them.  Do not change the file default merely to make it match the
production pipeline; the split is deliberate and documented.

## Two configuration decisions that matter — read before changing `config.py`

These are physics/experimental-design decisions, not tuning knobs. Both are
documented at length in `config.py`'s comments.

**`BEAM_MODE = 'raster'`** (current default). A single σ_xy=1cm pencil beam
has a 95% transverse span of 7.7 cm against a 15 cm Cu block —
`test_beam_covers_target_face` fails under `BEAM_MODE='pencil'` by design;
there is no tomogram to reconstruct. The default instead rasters the same
σ_xy=1cm spot over a 7×7 grid spanning the Cu face (`RASTER_NX/NY/HALF`),
equal exposure per node — how tagged-beam facilities actually cover an
extended target. `'uniform'` (flood illumination) and `'pencil'` remain
available if you want to compare.

**`STEER_COMPENSATION`** — `config.py` deliberately defaults to `'none'`
for the uncompensated diagnostic configuration.  `results_pipeline.py`
explicitly forces `'per_setting'` at runtime by patching `config` *before*
importing `simulate`.  This matters because `'none'` manufactures a large
setting-dependent momentum-position correlation through the uncompensated
dipole kick, while `'per_setting'` removes that gross displacement.  The two
steering tests described above validate both behaviors separately.

Earlier versions of `results_pipeline.py` documented the runtime override but
did not actually apply it.  Results generated by those versions under the
unintended `'none'` pathway must not be mixed with the corrected production
results.

## Reading the results

`results_pipeline.py` reports a **non-rescaling orthogonal fraction** in
the detector-level `I_nom - I_Q` decomposition.  It is the RMS fraction of
that difference orthogonal to a uniform rescaling of `I_Q`.  This is a useful
descriptive statistic, but it is not by itself a causal measurement of
"genuine physical structure"; residual momentum/path/reconstruction effects
and finite-statistics structure can also contribute.

A five-seed convergence run at 500,000 events per momentum setting (2 million
events per seed) under the corrected radial model and production
`'per_setting'` steering gave a stable non-rescaling orthogonal fraction of
about 0.213 with a seed-to-seed standard deviation of about 0.007, and a
per-momentum correction reduction of about 0.854 with seed-to-seed standard
deviation below 0.001. These are detector-level constant-momentum simulation
diagnostics, not full systematic uncertainties and not causal proof that the
orthogonal component is physical structure.

## Verification tags used in the manuscript/bib comments

`[PDF-OK]`/`[PDF-VERIFIED]` — checked against a source PDF.
`[WEB-OK]`/`[WEB-VERIFIED]` — checked online against a primary source.
`[CODE-REPRODUCED]` — recomputed independently and matches to rounding.
`[UNVERIFIED-CODE]` — simulation output not yet reproduced/finalized (mainly
the Geant4 cross-check numbers and the Hanson core-offset figures — the
imaging results this repo produces are no longer in this category once run).

## Status

- Theory and radial quadrature (εM, logarithmic second-moment and quadratic
  fourth-moment scaling, acceptance-matched estimator, and path-specific
  acceptance optimization): implemented and internally validated. The projected
  factorized construction remains diagnostic only.
- Beamline + detector-level diagnostic pipeline: all current `tests.py`
  checks pass. Five independent 2M-event seeds show stable fixed-cut artifact
  decomposition metrics; see "Reading the results" above.
- Geant4 cross-check (Sec. 3): harness ready (`geant4_compare.py`); actual
  Geant4 runs are external and not yet done.
- Manuscript: the current theory-only version deliberately excludes
  simulation-dependent image claims. Publication-grade detector/transport
  claims still require the stated remaining systematics: momentum-loss p(X)
  treatment where appreciable, a resolution-conditioned detector calibration
  for w_Q, and independent transport validation (e.g. Geant4). The Cu/Pb ROI
  mask-difference convention should also be stated explicitly in any future
  image-level writeup.