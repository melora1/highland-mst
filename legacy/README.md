# Preserved legacy implementation

This directory contains the complete pre-condensation Python implementation.
No functions, reports, verification scripts, plotting variants, or command-line
entry points were deleted. Internal imports are package-relative so the legacy
and production implementations cannot accidentally import one another.

Run legacy entry points from the repository root with module syntax, for example:

```bash
python3 -m legacy.run_all
python3 -m legacy.results_pipeline --help
python3 -m legacy.plot_results --help
python3 -m legacy.plotgeant4 --help
python3 -m legacy.plots_paper --help
python3 -m legacy.step1_report
python3 -m legacy.test_pofx
python3 -m legacy.verify_arithmetic
python3 -m legacy.verify_scaling
```

The production implementation remains at the repository root and is invoked
through `python3 run.py ...`.
