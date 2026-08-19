# Code-aligned revision notes

This package aligns the manuscript source with the uploaded Python/C++ implementation and removes numerical claims that cannot be regenerated from the supplied current code/data.

## Implemented now

- Production generator is identified correctly as the segmented-p(X), radial-Moliere model-internal simulator, not Geant4.
- Production raster is corrected to 9 x 9 nodes over +/-11 cm.
- Ordered target paths are traced exactly as `[Al_up, Cu_up, Pb, Cu_down, Al_down]`; the manuscript no longer describes a symmetric-order approximation.
- The production-default `dchi_c2` screening-log weighting and the alternative serial weighting are documented and their finite-loss spread is reported by `run.py theory`.
- Current p(X) energy-loss/mismatch values are inserted in the TeX source.
- `I_p` is described correctly: the central-path p(X) mismatch factor multiplies the event-specific reconstructed Highland core width.
- The exact image-level global scalar control `min_c RMS(I_nom/c - I_Q)` is implemented.
- `I_const` is retained separately as the event-count-mean mismatch control.
- Path residuals are reported for both truth and reconstructed reference-path classes.
- A split-half map-noise estimator is implemented, including a quadrature residual diagnostic.
- Absolute Pb SNR and Pb-Cu CNR are written for every image construction.
- Voxel grid and ROI definitions are stated explicitly in the TeX source.
- Occupancy discussion now uses the exact p(X) fourth moment; the central 6 GeV/c N=20 standard-error estimate is about 0.71 rather than the older asymptotic 0.65 value.
- The Geant4 comparator now accepts arbitrary Cu/Pb single slabs, uses the manuscript ratio sign convention, reports the implied quadratic-weight bias, produces a median/Rayleigh core diagnostic, provides delta-method sampling intervals, and decomposes the second-moment numerator into finite reduced-angle bands.
- The Geant4 executable interface now requires an explicit seed and documents `urban`, `wentzel`, and `wvi_ss`; it prints installed muon process names at runtime.
- The manuscript no longer states that the unmodified Urban/Wentzel reference lists necessarily contain the same explicit muon single-scattering process.
- Old Geant4 tables and old 7x7 detector numbers were removed from the code-aligned TeX because the supplied current source does not reproduce them without fresh transport/production runs.

## Validation performed here

- All Python files compile.
- `tests.py`: 20/20 tests passed.
- `run.py theory` completed successfully and regenerated the theory CSVs/figures used by the corrected TeX.
- The revised analysis completed successfully on the available 400k-event current-code event sample, including the new scalar and split-half controls.
- A fresh small end-to-end production smoke run completed successfully with the revised analysis pipeline.
- `geant4_compare.py` was exercised on a synthetic angle dump; the corrected ratio/sign, core output, finite-cut moments, and band decomposition were produced.
- The corrected TeX passed `pdflatex` syntax/figure compilation. Bibliography citations remain unresolved in the package because `references.bib` was not among the supplied files.

## Current-code 400k diagnostic only

These numbers are useful for code validation but are not substituted into the manuscript because the stated production exposure is 2e6 events.

- nominal-vs-Q image RMS: 0.7864
- event-mean scalar residual RMS: 0.1121
- exact optimal-scalar residual RMS: 0.04165
- p(X) residual RMS: 0.02657
- optimal image scale: c* = 1.39197
- reconstructed Al-only residual image RMS: 0.04566
- reconstructed Cu-bearing residual image RMS: 0.004684
- split-half noise estimate, reconstructed Al-only: 0.04922 on the stricter common half-occupancy voxel set
- split-half noise estimate, reconstructed Cu-bearing: 0.002191

The Al-only residual is therefore noise-dominated in this lower-statistics diagnostic sample; this is exactly why the final 2e6-event path claim must use the split-half analysis.

## Still requires an actual rerun or external source

No additional information from the user is needed to implement the code, but these scientific outputs cannot be truthfully fabricated from the supplied files:

- full 2e6-event production realization with the revised 9x9 raster;
- matched-seed production ensemble with revised raster and controls;
- fresh Geant4 Cu/Pb angle dumps with explicit seeds and runtime process provenance;
- Geant4 version actually used for final transport runs;
- layered Al+Cu / segmented-p(X) Geant4 benchmark;
- primary-table verification of the Sternheimer constants;
- quantitative radiative-loss systematic if claimed;
- final archival release tag/DOI.
