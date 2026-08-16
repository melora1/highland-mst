# Condensed revision codebase

This replaces the 20-file legacy tree with one production path.

## Files

- `config.py` — constants and revised ±11 cm raster.
- `physics.py` — Highland kinematics, Bethe/CSDA energy loss, radial Molière n≤2 model, exact reduced-variable formulation, p(X) segmented approximation, cached calibration/sampling.
- `geometry.py` — exact nested-target ray intervals and ordered material segments.
- `simulation.py` — re-steered tagged-beam detector simulation, equal exposure and controlled momentum-gradient exposure.
- `analysis.py` — Steps 1–7 tables, fixed-cut images, off-Cu test, adaptive-cut retention, paired statistics, causal gradient test.
- `plots.py` — reduced-variable and difference-map figures only; legacy five-near-identical panels and unresolved PSF plots are removed.
- `geant4_compare.py` — Step-8 comparison of supplied Geant4 angle dumps at k={2,5,10,20,40}, reporting Fc, truncated RMS and M4 without hard-coded model-error budgets.
- `tests.py` — pre-flight closures: geometry, exact reduced identity, Gaussian kernel, radial normalization/tail coefficient, sampler↔quadrature closure, azimuthal isotropy, CSDA range-table↔RK4 closure, dual p(X) screening conventions, steering, seed uniqueness, and stopping-power minima.
- `run.py` — sole CLI entry point.

## Commands

```bash
python tests.py
python run.py theory --out out/theory
python run.py simulate --n-per-setting 500000 --seed 0 --out out/equal
python run.py gradient --n-per-cell 20000 --seed 0 --out out/gradient
python run.py paired out/seed*/metrics.csv --out out/paired_seed_summary.csv
```

`run.py gradient` uses a **reference-only Al+Cu target by default** so the imposed momentum-mixture gradient is the only deliberate spatial intervention. Total fluence per raster cell is fixed.

Event tables are written as Parquet when a Parquet engine is installed; otherwise `run.py` falls back to a same-stem `.pkl` file rather than failing. `run.py analyze` accepts either format.

## Theoretical separation enforced in code

### Constant-p theory

The radial model obeys exactly, within the chosen n≤2 truncation,

\[
(1+\epsilon_M)^2 = R B\,\mu_2(\eta_{cut};B),\qquad
\eta_{cut}=\frac{\theta_{cut}}{\chi_c\sqrt B}
=\frac{k}{\sqrt{2RB}}.
\]

`analysis.run_theory()` writes the fixed-path collapse, general `epsilon_M(eta;R,B)` design table, matched-k/matched-eta composition tests, n≤1/n≤2 sensitivity, and a split log-law protocol: slope diagnostics at η=8–20, 10–30, and 15–30 entirely within the numerical table, plus deep η=30–100 and 50–500 windows used only to stabilize the asymptotic intercept.

The asymptotic fit is performed as

\[
(1+\epsilon_M)^2-1=m\ln\eta_{cut}+b
                  =m\ln(\eta_{cut}/\eta_1),
\]

with both `m` and `b` free.  The fitted slope is compared with the Rutherford/Moliere prediction `2R` only in windows with η≤`RADIAL_ETA_MAX=30`, so the comparison is not inherited from the analytic Rutherford continuation. Deep windows above η=30 are labeled `eta1_asymptote` and are used only for intercept stability. No universal `eta1` is assumed. Given the exact reduced identity, the fixed-2R pointwise intercept is `eta1 = eta*exp(-(R*B*mu2(eta;B)-1)/(2R))`, so within this model it is determined by `(R,B)` rather than by an additional composition parameter.

### Energy loss

`physics.calibrate_pofx()` propagates energy by CSDA and uses ordered path segments. The varying-p Moliere reduction carries two explicit screening-log continuations: local `dchi_c^2` weighting and slice-wise `Z(Z+1)X/A` serial weighting. They are algebraically identical at constant momentum. Their finite-loss spread is reported as an internal construction systematic; neither makes the one-B representation for a degrading path a theorem. Independent transport benchmarking is still required for publication-level absolute calibration.

The matched quantity

`epsilon_matched = theta_RMS[p(X)] / theta_space[p(X)] - 1`

is never confused with the deployed upstream-tagged quantity

`epsilon_mixed = theta_RMS[p(X)] / theta_space(p_in) - 1`.

## Publication cautions still requiring external verification

1. The Sternheimer material constants in `config.py` are the transcribed values from the supplied code. The test against published minimum stopping powers is only an indirect closure check; verify the constants against the primary PDG/LBL table before publishing derived numbers.
2. Collision stopping power is included; radiative loss is not. At these momenta it should be quantified rather than silently assumed negligible.
3. The segmented p(X) one-B Molière construction is explicit and internally closed but is not promoted to a theorem. `geant4_compare.py` is the independent benchmark hook.
4. The radial n≤2 expansion is an asymptotic truncation. Any clipped negative mass is reported by the calibration cache and theory tables.
5. `I_const` uses the unweighted event-count mean of the per-event mismatch as a pure scale-null control. The nominal-weight-weighted mean is reported separately in `calibration_summary.csv` so it is not confused with the old 12.18% convention.
6. The gradient analysis writes two predictors: the unweighted mean normalization field (the mechanism test) and `w_Q` times that field (an algebraic closure that should reproduce `I_nom-I_Q`; it is not independent evidence).
7. Adaptive-cut retention is reported with an explicit denominator (`all generated` as primary, plus a conditional fixed-cut-accepted diagnostic).

## Legacy files intentionally removed

`branch_a.py`, `branch_b.py`, `run_all.py`, the separate constant-p/p(X) epsilon modules, the old arithmetic verifier, and the old Geant4 plotting script are not part of this tree. Their roles either conflicted after the p(X) sampler patch or are now covered by the unified physics/analysis modules.
