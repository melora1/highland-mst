# Highland-MST revision codebase

This directory is synchronized to the corrected manuscript source `HighlandValidation_rev14.tex`.

## What the production simulation is

The Python production study is a **controlled model-internal detector study**, not full Geant4 transport. `simulation.py` traces each event's ordered material path analytically, then samples the non-factorized radial Moliere `n<=2` distribution using the segmented `p(X)` construction in `physics.py`. The production cache steps are declared in `config.py`. The path is partitioned into equal accumulated `dchi_c^2` intervals and each kink is placed at that interval's `dchi_c^2`-weighted centroid. Tracker hits and the upstream dipole momentum tag are then smeared/reconstructed.

Above 100 mrad the production model splices the Moliere core to a Rutherford tail with a Gaussian nuclear form factor. `--form-factor uniform_sphere` selects `[3 j1(qR)/(qR)]^2`, and `--form-factor none` retains the point-nucleus reference. Both finite-size models keep the explicit incoherent floor `(1 + Z|F|^2)/(Z+1)` for each local material contribution. `form_factor_continuity.csv` is a required validity gate: a row failing the 10% matching criterion must not be presented as a validated absolute prediction.

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
python run.py simulate --n-per-setting 500000 --seed 0 --n-kinks 25 --form-factor gaussian --out out/equal
python run.py gradient --n-per-cell 20000 --seed 0 --n-kinks 25 --form-factor gaussian --out out/gradient
python run.py paired out/seed*/metrics.csv --out out/paired_seed_summary.csv
python plots.py --root out --all
python validation.py quadrature --n-mc 10000000
python validation.py kink-composition --n-events 200000
```

`tests.py` currently contains 23 physics/geometry and analysis closure tests.

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

- rerun the Geant4 slab suite using explicit seeds and retain the angle dumps/metadata;
- retain the Geant4 version and runtime model/process output with every completed transport run;
- verify the transcribed Sternheimer density-effect constants against the primary PDG/LBL tables;
- quantify radiative energy loss if a precision stopping-power uncertainty is claimed;
- archive the final code release and DOI.

The checked-in detector outputs predate the finite-form-factor and equal-`dchi_c^2` kink changes. Regenerate the requested 20-seed ensemble before quoting detector-level values; do not reuse the older five-seed numbers or Geant4 comparators.
