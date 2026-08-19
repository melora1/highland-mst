# Geant4 validation (independent transport benchmark)

This documents the external transport benchmark referenced as the
`geant4_compare.py` hook in the main README (caution #3). It cross-checks the
constant-p radial Molière n≤2 model against single-material Geant4 slabs for
Cu and Pb over the four production momenta. It is a standalone stage: the C++
application produces per-event exit-angle dumps; the Python side compares them
against `constant_calibration` from `physics.py`.

## Layout

```
geant4/
  mstSim.cc                 # main(): parses <model> <material> <t_cm> <p_GeV> <N> <outfile>
  include/                  # headers
  src/
    PhysicsListFactory.cc   # BuildPhysicsList(model): urban -> FTFP_BERT, wentzel -> FTFP_BERT_WVI
    DetectorConstruction.cc # single centered slab, material/thickness from CLI
    PrimaryGeneratorAction.cc  # pencil mu- beam, +z, momentum from CLI
    SteppingAction.cc       # records primary space-angle at slab exit
    RunAction.cc            # writes theta_space (rad), one per line; reports non-exiting yield
    DetectorMessenger.cc
  build/mstSim              # compiled binary (not committed)
  out/                      # angle dumps + analysis outputs (not committed)
  plot_geant4.py            # convergence, cu_sweep, pb_sweep, impact_table stages
```

## Environment

Geant4 11.x with data, obtained via conda-forge (no local toolkit build):

```bash
conda install -c conda-forge geant4
conda activate geant4
geant4-config --version        # expect 11.x
echo $G4LEDATA                 # must be non-empty
```

macOS note: the reference lists link `libG4OpenGL`, so the binary needs an
rpath to the conda lib dir even for batch (no-vis) running. This is baked in at
link time below; it is not a runtime environment variable.

## Build (no cmake)

`geant4-config` emits the flags directly. `mstSim.cc` is at the top level; the
other sources are in `src/`, so both must appear on the compile line:

```bash
cd geant4
mkdir -p build
clang++ -std=c++17 -Iinclude mstSim.cc src/*.cc -o build/mstSim \
  $(geant4-config --cflags) $(geant4-config --libs) \
  -Wl,-rpath,$(geant4-config --prefix)/lib
./build/mstSim                 # prints usage line = success
```

## CLI

```
./build/mstSim <urban|wentzel> <Cu|Pb> <thickness_cm> <p_GeV> <nEvents> <outFile>
```

`urban` selects `FTFP_BERT` (Urban muon MSC); `wentzel` selects
`FTFP_BERT_WVI`. Both reference lists already register muon
`G4CoulombScattering`; the single-scatter tail is not suppressed and does not
need to be added. Output is one `theta_space` value (radians) per primary that
reaches the exit plane. `RunAction` reports any primaries that do not exit
(stopped/backscattered); this is <0.05% even for 8 cm Pb at 1 GeV/c and does
not affect truncated moments.

## Reproduce the validation

Production statistics are `N = 1e6`, fixed by the convergence stage below.

Convergence (sets N):

```bash
mkdir -p out
for N in 10000 100000 1000000 3000000; do
  ./build/mstSim urban Cu 15.0 2.0 $N out/conv_Cu15_p2_urban_N${N}.txt
done
python plot_geant4.py convergence      # writes out/convergence.png
```

Cu and Pb sweeps (16 dumps each; parallel across cores):

```bash
# Cu: {3,15} cm x {1,2,3.5,6} GeV x {urban,wentzel}
printf '%s\n' 3.0 15.0 | while read t; do for p in 1.0 2.0 3.5 6.0; do for m in urban wentzel; do
  echo "$m Cu $t $p 1000000 out/Cu_t${t}_p${p}_${m}.txt"
done; done; done | xargs -P 4 -L 1 ./build/mstSim

# Pb: {2,8} cm x {1,2,3.5,6} GeV x {urban,wentzel}
printf '%s\n' 2.0 8.0 | while read t; do for p in 1.0 2.0 3.5 6.0; do for m in urban wentzel; do
  echo "$m Pb $t $p 1000000 out/Pb_t${t}_p${p}_${m}.txt"
done; done; done | xargs -P 4 -L 1 ./build/mstSim
```

Analysis (project venv, where `physics.py`/`analysis.py` import):

```bash
conda deactivate
source /path/to/venv/bin/activate
cd geant4
python plot_geant4.py cu_sweep         # stage1_Cu.csv/.png
python plot_geant4.py pb_sweep         # stage1_Pb.csv/.png
python plot_geant4.py impact_table     # impact_200mrad.csv
python plot_geant4.py impact_figure    # out/geant4_impact_summary.{png,pdf}
```

The two-environment split is intentional: the C++ binary runs in the `geant4`
conda env; the comparison imports the model from the project venv. They
communicate only through the `.txt` dumps on disk, so the model is a single
source of truth (no reimplementation on the conda side).

## What the stages report

- `convergence` — truncated `theta_rms` at k=10 vs N. Drift falls below the
  0.5% target by N=1e6 (0.05%); production N is fixed there.
- `cu_sweep` / `pb_sweep` — per (thickness, momentum, k) fractional truncated-
  RMS difference (quadrature vs Geant4-urban) with a 200-sample bootstrap CI,
  and the Urban/Wentzel spread. k∈{2,5,10,20,40}.
- `impact_table` — the same fractional difference evaluated at the **physical
  200 mrad cut** rather than a reduced k, plus where 200 mrad falls in k for
  each config. This is the production-relevant impact. Writes
  `out/impact_200mrad.csv`.
- `impact_figure` — one-panel summary of `impact_table`: the truncated-RMS
  difference at 200 mrad versus the reduced angle k at which the cut falls,
  colored by material, with bootstrap error bars. Writes
  `out/geant4_impact_summary.{png,pdf}` (the `.pdf` is the paper figure). If
  the impact CSV is absent it is generated first, so this stage is
  self-sufficient given the dumps.

## Findings (conditional on this configuration)

1. **Core agreement.** At k=2 (and at the physical cut for low-k configs) the
   quadrature and Geant4 truncated RMS agree at the few-percent level across
   Al/Cu/Pb and 1–6 GeV/c. The empirical Geant4 core width matches the
   quadrature `theta_space` to <1% (Cu 15 cm, 6 GeV/c: 11.38 vs 11.29 mrad).

2. **Tail divergence, ordered by reduced angle.** The truncated-RMS difference
   grows monotonically with k, from ~0% near the core to −10…−16% by k=20–40.
   The size tracks reduced angle k almost independently of material/thickness,
   and is larger for higher Z and greater thickness — consistent with the
   difference between the model Rutherford Θ⁻³ continuation and Geant4's
   form-factor-suppressed large-angle single scattering.

3. **Model degeneracy at the observable.** At the 200 mrad cut the Urban and
   Wentzel reference lists agree to ~0.2%, so they do not supply a transport
   band there. The two models differ only in the far-tail *slope* (survival
   log-slope in u∈[8,40] roughly −2.5 vs −2.9, vs the Rutherford −2.0), which
   does not propagate strongly to the truncated second moment.

4. **Production relevance.** Thick, low-momentum reference paths sit at low k
   (Cu 15 cm/1 GeV: k≈4; Pb 8 cm/1–2 GeV: k≈3.5–7) and agree to within a few
   percent. Thin, high-momentum paths sit at high k (k≈25–60) and carry the
   full −10…−14% difference. This is the same thin/high-p regime that carries
   the largest reconstructed-path residual in the production analysis.

## Scope and limitations

- Single-material slabs only. The composite Al+Cu production path is not built
  here (the detector places one centered slab); a two-layer geometry is the
  next extension if the energy-loss p(X) construction is to be benchmarked
  directly.
- `SetMscThetaLimit` in the factory is global (affects e± as well); harmless
  for a muon-only slab but revisit before reuse in mixed jobs.
- The far-tail slope is not resolvable at N=1e6: beyond u≈20 the survival
  function is at 1e-5–1e-6, so per-window slope fits are Poisson-noise limited
  and are reported only as an order-of-magnitude bracket, not a fitted power.
- Reproducibility: each `mstSim` invocation uses the Geant4 default engine
  seed. Parallel `xargs` runs are independent; results are statistically
  reproducible at the quoted CIs, not bitwise identical across runs.
```