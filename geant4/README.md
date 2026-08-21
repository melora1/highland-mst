# mstSim

Minimal Geant4 single-slab muon multiple-scattering benchmark.

## Requirements

- CMake 3.16+
- A Geant4 installation with CMake package files available
- A C++17 compiler
- Python 3 only if using `run_benchmark_suite.sh`
- `geant4_compare.py` in the parent directory of the run location if using the benchmark suite as originally written

## Build

From this directory:

```bash
cmake -S . -B build
cmake --build build -j
```

If CMake cannot find Geant4, first source the Geant4 setup script, for example:

```bash
source /path/to/geant4-install/bin/geant4.sh
```

or provide its CMake package location:

```bash
cmake -S . -B build -DGeant4_DIR=/path/to/lib/cmake/Geant4
cmake --build build -j
```

## Run one simulation

```bash
./build/mstSim urban Cu 15.0 1.0 10000 12345 out.txt
```

Arguments:

```text
./mstSim <urban|wentzel|wvi_ss> <Cu|Pb> <thickness_cm> <p_GeV> <nEvents> <seed> <outFile>
```

The output is one primary exit polar angle per line, in radians.

## Benchmark suite

The supplied `run_benchmark_suite.sh` expects `EXE=./mstSim` by default and calls `../geant4_compare.py`. A convenient layout is therefore either to copy/symlink the executable next to the script or override `EXE`:

```bash
EXE=./build/mstSim N_EVENTS=10000 ./run_benchmark_suite.sh
```

The comparison stage still requires `../geant4_compare.py`. That Python program was not among the supplied files, so it cannot be reconstructed exactly from the Geant4 source alone.
