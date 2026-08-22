#!/usr/bin/env bash
set -euo pipefail

EXE=${EXE:-./mstSim}
PYTHON=${PYTHON:-python}
N_EVENTS=${N_EVENTS:-1000000}
OUT=${OUT:-out/geant4}
INCLUDE_WVI_SS=${INCLUDE_WVI_SS:-0}
mkdir -p "$OUT/raw" "$OUT/compare" "$OUT/logs"

if command -v geant4-config >/dev/null 2>&1; then
  geant4-config --version > "$OUT/geant4_version.txt"
else
  printf '%s\n' 'geant4-config not found; record Geant4 version manually.' > "$OUT/geant4_version.txt"
fi

counter=0
for MAT in Cu Pb; do
  if [[ "$MAT" == "Cu" ]]; then
    THICKS=(3.0 15.0)
  else
    THICKS=(2.0 8.0)
  fi
  for T in "${THICKS[@]}"; do
    for P in 1.0 2.0 3.5 6.0; do
      counter=$((counter + 1))
      SU=$((510000 + counter * 10 + 1))
      SW=$((510000 + counter * 10 + 2))
      FU="$OUT/raw/${MAT}_t${T}_p${P}_ftfp_bert_s${SU}.txt"
      FW="$OUT/raw/${MAT}_t${T}_p${P}_ftfp_bert_wvi_s${SW}.txt"
      "$EXE" ftfp_bert "$MAT" "$T" "$P" "$N_EVENTS" "$SU" "$FU" | tee "$OUT/logs/${MAT}_t${T}_p${P}_ftfp_bert_s${SU}.log"
      "$EXE" ftfp_bert_wvi "$MAT" "$T" "$P" "$N_EVENTS" "$SW" "$FW" | tee "$OUT/logs/${MAT}_t${T}_p${P}_ftfp_bert_wvi_s${SW}.log"
      FILE_ARGS=(--file "ftfp_bert=$FU" --file "ftfp_bert_wvi=$FW")
      if [[ "$INCLUDE_WVI_SS" == "1" ]]; then
        SS=$((510000 + counter * 10 + 3))
        FS="$OUT/raw/${MAT}_t${T}_p${P}_wvi_ss_s${SS}.txt"
        "$EXE" wvi_ss "$MAT" "$T" "$P" "$N_EVENTS" "$SS" "$FS" | tee "$OUT/logs/${MAT}_t${T}_p${P}_wvi_ss_s${SS}.log"
        FILE_ARGS+=(--file "wvi_ss=$FS")
      fi
      "$PYTHON" ../geant4_compare.py \
        "${FILE_ARGS[@]}" \
        --material "$MAT" \
        --thickness-cm "$T" \
        --p "$P" \
        --n-generated "$N_EVENTS" \
        --theta-cut-mrad 200 \
        --u-bands 0 3 10 20 40 80 \
        --out "$OUT/compare/${MAT}_t${T}_p${P}_compare.csv"
    done
  done
done
