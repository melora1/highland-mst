# mstSim -- Geant4 cross-check (manuscript Sec. 3)

Fires a monoenergetic mu- beam through a single Cu or Pb slab and records
each primary's exit angle. Feeds directly into `geant4_compare.py` (in the
main project, one level up), which compares the simulated in-acceptance
RMS to the manuscript's deterministic quadrature.

**Not built or run in this environment** -- this is source code, written to
be buildable as-is against an installed Geant4. Verify against your own
Geant4 version before trusting the output; see the two `[VERIFY]` notes
below.

## What it does

- Single slab, material and thickness set at run time (`Cu`/`Pb`, any
  thickness in cm).
- mu- beam, momentum set at run time (converted to kinetic energy
  internally: `mstSim.cc` computes `E_kin = sqrt(p^2 + m_mu^2) - m_mu`).
- Two MSC-model configurations, selected via Geant4's reference physics
  list factory: `urban` -> `FTFP_BERT` (Geant4's default MSC), `wentzel`
  -> `FTFP_BERT_WVI` (Wentzel-VI + single Coulomb scattering). **[VERIFY]**
  the `_WVI` suffix is a documented Geant4 feature but is version-sensitive;
  confirm `FTFP_BERT_WVI` builds without error on your Geant4 install
  before trusting the "wentzel" runs. `PhysicsListFactory.hh` documents the
  manual fallback (`G4EmConfigurator` + `G4WentzelVIModel`) if it doesn't.
- A thin vacuum "scoring plane" immediately downstream of the slab; the
  primary's exit angle relative to the beam axis is written the instant it
  crosses into that plane, then the track is killed.
- Energy loss is **enabled** (unlike the beamline simulation in the main
  project, which deliberately omits it) -- this matches the manuscript's
  Sec. 3 statement that "energy loss is enabled in Geant4 but the incident
  momentum is used in the comparison." **[VERIFY]** if you want the
  no-energy-loss control the manuscript also describes for the low-momentum
  settings, disable energy loss physics processes for mu- in
  `PhysicsListFactory.cc` (e.g. via `G4EmParameters` or a custom process
  table edit) -- not implemented here, since it changes physics-list
  internals in a way that's easy to get subtly wrong without testing
  against a live build.

## Build

```bash
mkdir build && cd build
cmake -DGeant4_DIR=/path/to/geant4-install/lib/Geant4-11.x ..
make -j4
```

Requires Geant4 >= 10.7 (uses `G4RunManagerFactory`; the pattern here
targets 11.x). No Geant4 UI/Vis libraries are required (batch-only, no
macros, no visualization) -- a minimal Geant4 build without Qt/OpenGL is
sufficient.

## Run one configuration

```bash
./build/mstSim urban Cu 15.0 1.0 500000 out/Cu_t15.0_p1.0_urban.txt
```

Arguments: `<urban|wentzel> <Cu|Pb> <thickness_cm> <p_GeV> <nEvents> <outFile>`.

Output: one `theta_space` value (rad) per line -- the primary muon's exit
angle relative to the beam axis, for every event whose primary reached the
scoring plane. `RunAction` prints how many events were written vs.
generated at the end of the run; if this drops noticeably below 100% at low
momentum, check whether primaries are ranging out or backscattering in the
thicker slabs before trusting the tail of the distribution.

## Run the full sweep

```bash
cd build   # or wherever mstSim was built
../run_sweep.sh 500000 out
```

32 configurations (2 materials x 2 thicknesses x 4 momenta x 2 models),
matching the manuscript's per-material path lengths (Sec. 5.2's k_opt
table: Cu at x/X0 = 2.08, 10.42; Pb at x/X0 = 3.57, 14.29). Each
configuration is fully independent -- for anything beyond a quick check,
run the 32 jobs in parallel (e.g. `GNU parallel`, a Slurm array, or similar)
rather than serially inside the loop as written.

## Analyze against the quadrature

From the main project directory:

```bash
python3 geant4_compare.py --file geant4/out/Cu_t15.0_p1.0_urban.txt \
    --material Cu --thickness_cm 15.0 --p 1.0 --model urban
```

Repeat for each of the 32 output files. `geant4_compare.py` computes the
in-acceptance (200 mrad) RMS from the Geant4 dump, compares it to
`quadrature.theta_rms` for the same configuration, and checks the
difference against the published Urban/Wentzel-VI model spread
(Makarova et al. 2017) plus the manuscript's ~1% planar-correlation
systematic -- see that script's docstring for the pass criterion.

## Layout

| File | Role |
|---|---|
| `mstSim.cc` | Main driver; fully argument-driven (no macros needed for the sweep). |
| `include/`, `src/DetectorConstruction.*` | Slab + scoring-plane geometry. |
| `include/`, `src/DetectorMessenger.*` | `/det/setMaterial`, `/det/setThickness` UI commands. |
| `include/`, `src/PhysicsListFactory.*` | Urban vs. Wentzel-VI selection. |
| `include/`, `src/PrimaryGeneratorAction.*` | mu- gun. |
| `include/`, `src/RunAction.*` | Output file lifecycle. |
| `include/`, `src/SteppingAction.*` | Scores the primary's exit angle. |
| `run_sweep.sh` | Drives the full 32-configuration sweep. |
| `CMakeLists.txt` | Build configuration. |
