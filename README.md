# Highland-MST revision codebase

This directory is synchronized to the corrected manuscript source `HighlandValidation_rev14.tex`.

## What the production simulation is

The Python production study is a **controlled model-internal detector study**, not full Geant4 transport. `simulation.py` traces each event's ordered material path analytically, then samples the non-factorized radial Moliere `n<=2` distribution using the segmented `p(X)` construction in `physics.py`. The production cache steps are declared in `config.py`. The path is partitioned into equal accumulated `dchi_c^2` intervals and each kink is placed at that interval's `dchi_c^2`-weighted centroid. Tracker hits and the upstream dipole momentum tag are then smeared/reconstructed.

The former 100 mrad form-factor splice is superseded and must not be used. Finite-size theory now inserts `G(q)` inside the screened-Rutherford characteristic exponent and obtains the angular law by an unexpanded Hankel transform. The kernel has `G(0)=1`, transitions to the approximate quasi-elastic nucleon floor `A/[Z(Z+1)]`, and terminates that floor with a dipole proton form factor. Gaussian and uniform-sphere nuclear shapes are both regenerated, with the floor-omitted difference reported as a systematic.

The inverse-CDF sampler is generated from the transformed density. `reduced_cache.py` moves the Hankel inversion into a precomputed table in `(1/B, ln rho)`, applies truncation as `u -> u F(eta_cut)`, and raises on every support miss rather than extrapolating. Empirical five- and 25-kink support has been measured for target and reference paths: together it spans `B=3.251--16.158` and `rho=0.492--934.1`, far beyond the four nominal path values. The production grid covers `B=3.175--18.343` and `rho=0.337--1362.344`. All Gaussian/sphere, floor-on/off tables passed the 56-point off-node `M2`, exhaustive support, and sub-millisecond lookup gates, so finite-size detector production is enabled. Old finite-size detector outputs remain superseded and must be regenerated.

The reference geometry replaces Pb by Cu. `geometry.py` traces exact ordered ray segments `[Al_up, Cu_up, Pb, Cu_down, Al_down]`; energy loss is not reconstructed from unordered totals.

## Current production geometry

- Al cube: 25 cm side.
- Cu cube: 15 cm side.
- Pb cylinder: radius 2 cm, height 15 cm, center `(x,y)=(3,2)` cm.
- Raster: **9 x 9** nodes over `[-11,+11] cm` in both transverse coordinates.
- Four nominal momenta: 1, 2, 3.5, 6 GeV/c.
- 1% Gaussian true-momentum bite; 2 mrad beam divergence; 1 cm Gaussian spot.
- Six tracker planes at `z=(-120,-90,-45,-15,25,65)` cm with 200 um single-hit resolution.
- Point dipole: 1 T, 0.30 m effective length; beam is re-steered separately at each nominal momentum.
- Fixed angular cut: 200 mrad.
- Image grid: 50^3 voxels over `[-15,+15] cm`, 0.6 cm voxels; map statistics require at least 20 entries per voxel.

## Physics levels kept separate

1. **Constant-p radial Moliere model.** The exact reduced identity within the `n<=2` truncation is

   `(1+epsilon_M)^2 = R B mu2(eta_cut;B)`.

2. **Segmented p(X) extension.** Energy is propagated by collision stopping power. Two explicit screening-log continuations are available:
   - `dchi_c2` (production default): local `ln chi_a^2` weighted by local `dchi_c^2`;
   - `serial`: common-p `Z(Z+1)X/A` weighting retained slice by slice.

   They coincide at constant momentum. Their finite-loss spread is written to `energy_loss_calibration.csv` and treated as a construction systematic, not as a theorem.

3. **Detector weights.** `I_nom`, `I_p`, and `I_Q` remain distinct. `I_p` applies the central-path p(X) mismatch factor to the event-specific reconstructed Highland core width; it is not a completely fixed axial denominator.

## Analysis controls now implemented

`analysis.py` writes:

- `metrics.csv`: absolute Pb SNR and Pb-Cu CNR for `I_nom`, `I_const`, `I_scale_opt`, `I_p`, `I_Q`, and `I_ideal`;
- `artifact_summary.csv`: nominal residual, event-mean scalar control, exact voxel-RMS-optimal scalar control, and p(X) residual;
- `calibration_summary.csv`: scalar conventions, p(X) momentum loss, screening mode and clipping diagnostic;
- `path_residuals.csv`: **both truth and reconstructed** Al-only/Cu-bearing path classes;
- `path_class_migration.csv`: truth/reconstruction migration table;
- `split_half_noise.csv`: independent split-half estimate of the map-RMS noise floor for `I_p-I_Q`, including a quadrature residual diagnostic;
- `adaptive_retention.csv`: retention relative to all generated events plus the conditional fixed-cut diagnostic;
- `images.npz`: all image estimators, including `I_scale_opt`.

The exact global scalar control minimizes

`RMS(I_nom/c - I_Q)`

over the same valid voxels used for the map comparison. `I_const` is retained separately as the event-count-mean epsilon control over accepted events with a defined, nonzero reconstructed reference path.

## Commands

```bash
python tests.py
python run.py theory --out out/theory
python run.py simulate --n-per-setting 500000 --seed 0 --n-kinks 25 --form-factor none --out out/equal
python run.py gradient --n-per-cell 20000 --seed 0 --n-kinks 25 --form-factor none --out out/gradient
python run.py paired out/seed*/metrics.csv --out out/paired_seed_summary.csv
python plots.py --root out --all
python validation.py quadrature --n-mc 10000000
python validation.py transform-g1
python validation.py finite-size-transform
python validation.py decision-gates
python validation.py analytic-completion
python validation.py finite-size-sampler --n-mc 10000000
python validation.py task8-summary out/geant4/task8/compare --out out/validation/task8
python validation.py task9-support out/fine/seed0/events.parquet --n-kinks 25 --compact --out out/validation/task9/support_k25
python validation.py task9-build out/fine/seed0/events.parquet out/validation/task9/support_k5 out/validation/task9/support_k25 --out out/validation/task9/cache_gaussian_floor_on_k5_k25_v5.npz --ff-model gaussian --floor on --n-B 70 --n-rho 70 --workers 6
python task9_matrix.py out/fine/seed0/events.parquet out/validation/task9/cache_gaussian_floor_on_k5_k25_v5.npz out/validation/task9/support_k5 out/validation/task9/support_k25 --workers 6
python validation.py task10 --eta-cut 2.713
python validation.py geant4-finite
python validation.py kink-composition --n-events 200000
```

At the small-angle-valid Task 10 acceptance, `eta_cut=2.713`, the matched-cut
`epsilon_M` values at 1 GeV/c are:

| form factor | Al25 | Cu15 | AlCu | Pb15 | Pb15 - AlCu |
|---|---:|---:|---:|---:|---:|
| Gaussian | 1.337% | -0.119% | -0.474% | -1.505% | -1.032 pp |
| sphere | 1.322% | -0.275% | -0.630% | -1.752% | -1.122 pp |

The composition inversion therefore survives the smaller reduced acceptance
for both form factors; it is not only a large-acceptance effect.

The completed Task 9 production-cache gates are:

| form factor | floor | kinks | max relative M2 error | lookup (microseconds/event) | support misses |
|---|---:|---:|---:|---:|---:|
| Gaussian | on | 5 | 0.000212 | 0.336 | 0 / 1,409,615 |
| Gaussian | on | 25 | 0.000304 | 0.334 | 0 / 7,048,075 |
| Gaussian | off | 5 | 0.000069 | 0.426 | 0 / 1,409,615 |
| Gaussian | off | 25 | 0.000059 | 0.561 | 0 / 7,048,075 |
| sphere | on | 5 | 0.000211 | 0.345 | 0 / 1,409,615 |
| sphere | on | 25 | 0.000309 | 0.385 | 0 / 7,048,075 |
| sphere | off | 5 | 0.000042 | 0.330 | 0 / 1,409,615 |
| sphere | off | 25 | 0.000059 | 0.332 | 0 / 7,048,075 |

`tests.py` currently contains 38 physics/geometry and analysis closure tests.

## Geant4 single-slab benchmark

The C++ source under `geant4/` produces single-material Cu or Pb exit-angle dumps. It is separate from the Python production generator.

The corrected executable interface requires an explicit random seed:

```bash
./mstSim <ftfp_bert|ftfp_bert_wvi|wvi_ss> <Cu|Pb> <thickness_cm> <p_GeV> <nEvents> <seed> <outFile>
```

Example:

```bash
./mstSim ftfp_bert Cu 15.0 1.0 1000000 12345 out/Cu_t15_p1_ftfp_bert_s12345.txt
python geant4_compare.py \
  --file ftfp_bert=out/Cu_t15_p1_ftfp_bert_s12345.txt \
  --material Cu --thickness-cm 15.0 --p 1.0 --n-generated 1000000 \
  --out out/Cu_t15_p1_compare.csv
```

`geant4_compare.py` now uses the manuscript sign convention

`theta_rms_model/theta_rms_G4 - 1`,

reports the corresponding quadratic-weight bias, a median/Rayleigh core-width comparison, delta-method sampling intervals from `M4`, and a finite reduced-angle band decomposition of the second-moment numerator. It also accepts `--path AlCu` or `--path Al25` for future layered/reference transport dumps.

`ftfp_bert` and `ftfp_bert_wvi` name the unmodified Geant4 reference lists directly; they are not described as Urban-versus-Wentzel modes because the installed muon model is version dependent (both use WentzelVI in Geant4 11.4.2). `wvi_ss` is an explicit diagnostic configuration that replaces the reference-list muon MSC with Wentzel-VI plus discrete Coulomb scattering. It is **not** assumed a priori to be more physical than the unmodified reference lists. `mstSim` prints the installed muon process names at runtime so the process configuration can be recorded.

## Remaining external work

The following cannot be completed from source code alone:

- verify the transcribed Sternheimer density-effect constants against the primary PDG/LBL tables;
- quantify radiative energy loss if a precision stopping-power uncertainty is claimed;
- archive the final code release and DOI.

The checked-in detector outputs predate the finite-form-factor and equal-`dchi_c^2` kink changes. Regenerate the requested 20-seed ensemble before quoting detector-level values; do not reuse the older five-seed numbers or Geant4 comparators.
