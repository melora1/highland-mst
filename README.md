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
| `moliere.py` | χ_c, χ_a, B (Appendix A), projected Molière f⁽ⁿ⁾ computed from the defining generating integral (not transcribed from Bethe's Table II — see the CONVENTION block in the module), cached CDF sampler. |
| `simulate.py` | Event pipeline: beam → dipole momentum tag → Molière scatter at the target → downstream tracking → PoCA. Writes one flat table per momentum setting. |

### Calibration and results (fills manuscript Sec. 5)

| File | Role |
|---|---|
| `eps_quadrature.py` | **Deterministic** εM(p, path) by 2D acceptance quadrature on this repo's own `moliere.py` — the manuscript's Sec. 2.3 quantity. Reproduces the manuscript's axial εM table (+5.1/+10.8/+14.2/+17.1 % at 1/2/3.5/6 GeV/c) to rounding; run `python3 eps_quadrature.py` to check. This is the calibration source for the imaging weight — **not** `branch_a.py`'s Monte Carlo fit, which answers a different question (reconstruction noise/resolution/truncation decomposition). |
| `results_pipeline.py` | Builds the four manuscript images (I_nom, I_p, I_ideal, I_const — Sec. 4.4), runs SNR/CNR/DP + edge response, and computes the artifact map / null-control decomposition. **This is what fills the previously-empty Results §5.2–5.4.** |

### Original validation suite (still useful; not on the path that produces the paper's numbers anymore)

| File | Role |
|---|---|
| `branch_a.py` | Residual decomposition (noise / momentum-resolution / truncation / εM) via Monte Carlo + bootstrap. Useful as a sampler-closure and reconstruction-noise diagnostic; its εM fit is no longer what calibrates the imaging weight (see `eps_quadrature.py`). |
| `branch_b.py` | Original imaging driver (nominal/biased/corrected images). Superseded by `results_pipeline.py` for the paper's numbers, but its ROI/metrics/edge-response helpers (`roi_masks`, `metrics`, `edge_response`, `voxelise`) are imported directly by `results_pipeline.py` — keep it in the repo. |
| `run_all.py` | Original end-to-end driver (tests → Gaussian control → Molière run → Branch A → Branch B). Runs under `config.py`'s defaults; optional, diagnostic only. |
| `tests.py` | Pre-flight validation. **Run this before any production run.** 17 checks including geometry invariants, the Molière single-scatter tail limit, and the beam/steering assumptions the imaging claim depends on. |

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

**`STEER_COMPENSATION`** — `config.py` default is `'none'`; `results_pipeline.py`
overrides this to `'per_setting'` at runtime (it patches `config` *before*
importing `simulate`, so it never touches the file on disk). This matters
because the uncorrected dipole kick (`'none'`) manufactures a large
momentum-position correlation that would trivially produce the
spatially-structured artifact Sec. 2.3/4.4 claims — `'per_setting'` (a beam
re-steered per momentum setting, as a real tagged beamline is operated) is
what the manuscript's Sec. 3.3 actually specifies, and is the harder,
honest test of whether that claim survives. The repo's own
`test_momentum_position_correlation_exists` documents the difference
(4.90 cm spread with `'none'` vs. 0.04 cm with `'per_setting'`, against a
0.6 cm voxel).

## Reading the results

`results_pipeline.py`'s printed report has one number that actually decides
the paper's central imaging claim: **`orthogonal (genuine structure) fraction`**
in the artifact decomposition. It's the RMS of the component of the
I_nom − I_ideal artifact map that is *not* explainable by a uniform rescale
(the I_const null control) — i.e. genuine momentum/path-length-driven spatial
structure surviving a re-steered beamline. At n=50,000/setting this measured
**0.249** with a 90.4% per-momentum correction reduction, i.e. most of the
artifact is a global rescale but a real ~25% residual structure survives —
run the full convergence scan (step 4 above) before treating that number as
final.

## Verification tags used in the manuscript/bib comments

`[PDF-OK]`/`[PDF-VERIFIED]` — checked against a source PDF.
`[WEB-OK]`/`[WEB-VERIFIED]` — checked online against a primary source.
`[CODE-REPRODUCED]` — recomputed independently and matches to rounding.
`[UNVERIFIED-CODE]` — simulation output not yet reproduced/finalized (mainly
the Geant4 cross-check numbers and the Hanson core-offset figures — the
imaging results this repo produces are no longer in this category once run).

## Status

- Theory and quadrature (εM, √ln and quadratic divergence laws, the
  acceptance-matched estimator, k_opt, η_max): complete, independently
  reproduced by a from-scratch implementation, matches the manuscript to
  rounding.
- Beamline + imaging pipeline: complete, validated (all 17 `tests.py` checks
  pass), and produces real (non-placeholder) numbers as of the raster-beam
  fix. Convergence with event count is still being characterized — see
  "Reading the results" above.
- Geant4 cross-check (Sec. 3): harness ready (`geant4_compare.py`); actual
  Geant4 runs are external and not yet done.
- Manuscript Discussion/Conclusions: not yet written; depend on the
  converged results-pipeline numbers.