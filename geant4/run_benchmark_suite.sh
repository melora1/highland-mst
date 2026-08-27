#!/usr/bin/env bash
set -euo pipefail

EXE=${EXE:-./mstSim}
PYTHON=${PYTHON:-python}
N_EVENTS=${N_EVENTS:-1000000}
N_SEEDS=${N_SEEDS:-3}
OUT=${OUT:-out/geant4}
mkdir -p "$OUT/raw" "$OUT/compare" "$OUT/logs"

if command -v geant4-config >/dev/null 2>&1; then
  geant4-config --version > "$OUT/geant4_version.txt"
else
  printf '%s\n' 'geant4-config not found; record Geant4 version manually.' > "$OUT/geant4_version.txt"
fi

counter=0
for MAT in Cu Pb; do
  if [[ "$MAT" == "Cu" ]]; then
    T=15.0
  else
    T=15.0
  fi
  for P in 1.0 2.0 3.5 6.0; do
    for SEED_INDEX in $(seq 1 "$N_SEEDS"); do
      counter=$((counter + 1))
      SU=$((510000 + counter * 10 + 1))
      SW=$((510000 + counter * 10 + 2))
      SS=$((510000 + counter * 10 + 3))
      FU="$OUT/raw/${MAT}_t${T}_p${P}_ftfp_bert_s${SU}.txt"
      FW="$OUT/raw/${MAT}_t${T}_p${P}_ftfp_bert_wvi_s${SW}.txt"
      FS="$OUT/raw/${MAT}_t${T}_p${P}_wvi_ss_s${SS}.txt"
      "$EXE" ftfp_bert "$MAT" "$T" "$P" "$N_EVENTS" "$SU" "$FU" | tee "$OUT/logs/${MAT}_t${T}_p${P}_ftfp_bert_s${SU}.log"
      "$EXE" ftfp_bert_wvi "$MAT" "$T" "$P" "$N_EVENTS" "$SW" "$FW" | tee "$OUT/logs/${MAT}_t${T}_p${P}_ftfp_bert_wvi_s${SW}.log"
      "$EXE" wvi_ss "$MAT" "$T" "$P" "$N_EVENTS" "$SS" "$FS" | tee "$OUT/logs/${MAT}_t${T}_p${P}_wvi_ss_s${SS}.log"
      FILE_ARGS=(--file "ftfp_bert=$FU" --file "ftfp_bert_wvi=$FW" --file "wvi_ss=$FS")
      "$PYTHON" ../geant4_compare.py \
        "${FILE_ARGS[@]}" \
        --material "$MAT" \
        --thickness-cm "$T" \
        --p "$P" \
        --n-generated "$N_EVENTS" \
        --theta-cut-mrad 200 \
        --out "$OUT/compare/${MAT}_t${T}_p${P}_seed${SEED_INDEX}_compare.csv"
    done
  done
done
