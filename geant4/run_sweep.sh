#!/usr/bin/env bash
# run_sweep.sh -- the full Sec. 3 Geant4 cross-check sweep.
#
# Material/thickness pairs match the manuscript's own per-material k_opt
# table (Sec. 5.2): Cu at x/X0 = 2.08 and 10.42 (t = 3.0, 15.0 cm, using
# PDG X0 = 1.44 cm); Pb at x/X0 = 3.57 and 14.29 (t = 2.0, 8.0 cm, X0 =
# 0.56 cm). Four momenta and both MSC models, as in the manuscript.
#
# Usage:
#   ./run_sweep.sh [N_EVENTS] [OUT_DIR]
# Defaults: N_EVENTS=100000 (fast pass -- raise to 500000+ for final
# numbers, matching the beamline simulation's per-setting statistics),
# OUT_DIR=out
#
# Runtime: 2 materials x 2 thicknesses x 4 momenta x 2 models = 32 runs.
# At N=500000 this is a substantial batch job -- consider running the
# 32 configs in parallel (e.g. GNU parallel or a job array) rather than
# serially, since each run is fully independent.

set -euo pipefail

N=${1:-100000}
OUT=${2:-out}
mkdir -p "$OUT"

MOMENTA=(1.0 2.0 3.5 6.0)
MODELS=(urban wentzel)
# material  thickness_cm
CONFIGS=(
  "Cu 3.0"
  "Cu 15.0"
  "Pb 2.0"
  "Pb 8.0"
)

BIN=./build/mstSim
if [ ! -x "$BIN" ]; then
  echo "mstSim not found at $BIN -- build first (see README.md)." >&2
  exit 1
fi

for cfg in "${CONFIGS[@]}"; do
  read -r MAT THICK <<< "$cfg"
  for P in "${MOMENTA[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      OUTFILE="$OUT/${MAT}_t${THICK}_p${P}_${MODEL}.txt"
      echo "=== $MAT t=${THICK}cm p=${P}GeV/c model=$MODEL -> $OUTFILE ==="
      "$BIN" "$MODEL" "$MAT" "$THICK" "$P" "$N" "$OUTFILE"
    done
  done
done

echo "done. Analyze each file with:"
echo "  python3 ../geant4_compare.py --file $OUT/<name>.txt --material <Cu|Pb> \\"
echo "      --thickness_cm <t> --p <p> --model <urban|wentzel>"
