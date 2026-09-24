#!/usr/bin/env bash
set -euo pipefail

EXE=${EXE:-./mstSim}
PYTHON=${PYTHON:-python}
N_EVENTS=${N_EVENTS:-1000000}
N_SEEDS=${N_SEEDS:-3}
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
OUT=${OUT:-out/geant4}
mkdir -p "$OUT/raw" "$OUT/compare" "$OUT/logs"

if command -v geant4-config >/dev/null 2>&1; then
  geant4-config --version > "$OUT/geant4_version.txt"
else
  printf '%s\n' 'geant4-config not found; record Geant4 version manually.' > "$OUT/geant4_version.txt"
fi

run_transport() {
  local CONFIG=$1 MAT=$2 T=$3 P=$4 SEED=$5 FILE=$6 LOG=$7
  if [[ -s "$FILE" && -s "$LOG" ]] && grep -q '\[RunAction\] wrote' "$LOG"; then
    return
  fi
  "$EXE" "$CONFIG" "$MAT" "$T" "$P" "$N_EVENTS" "$SEED" "$FILE" > "$LOG" 2>&1
}

generated_from_log() {
  local LOG=$1 FALLBACK=$2 VALUE
  VALUE=$(sed -n 's/.*wrote [0-9][0-9]* \/ \([0-9][0-9]*\) primary.*/\1/p' "$LOG" | tail -n 1)
  printf '%s\n' "${VALUE:-$FALLBACK}"
}

run_seed_set() {
  local MAT=$1 T=$2 P=$3 SEED_INDEX=$4 COUNTER=$5
  local SU=$((510000 + COUNTER * 10 + 1))
  local SW=$((510000 + COUNTER * 10 + 2))
  local SS=$((510000 + COUNTER * 10 + 3))
  local FU="$OUT/raw/${MAT}_t${T}_p${P}_ftfp_bert_s${SU}.txt"
  local FW="$OUT/raw/${MAT}_t${T}_p${P}_ftfp_bert_wvi_s${SW}.txt"
  local FS="$OUT/raw/${MAT}_t${T}_p${P}_wvi_ss_s${SS}.txt"
  local LU="$OUT/logs/${MAT}_t${T}_p${P}_ftfp_bert_s${SU}.log"
  local LW="$OUT/logs/${MAT}_t${T}_p${P}_ftfp_bert_wvi_s${SW}.log"
  local LS="$OUT/logs/${MAT}_t${T}_p${P}_wvi_ss_s${SS}.log"
  printf 'Running %s t=%s p=%s seed-set=%s\n' "$MAT" "$T" "$P" "$SEED_INDEX"
  run_transport ftfp_bert "$MAT" "$T" "$P" "$SU" "$FU" "$LU"
  run_transport ftfp_bert_wvi "$MAT" "$T" "$P" "$SW" "$FW" "$LW"
  run_transport wvi_ss "$MAT" "$T" "$P" "$SS" "$FS" "$LS"
  local NU NW NS
  NU=$(generated_from_log "$LU" "$N_EVENTS")
  NW=$(generated_from_log "$LW" "$N_EVENTS")
  NS=$(generated_from_log "$LS" "$N_EVENTS")
  local FILE_ARGS=(--file "ftfp_bert=$FU" --file "ftfp_bert_wvi=$FW" --file "wvi_ss=$FS")
  local GENERATED_ARGS=(--n-generated-label "ftfp_bert=$NU" --n-generated-label "ftfp_bert_wvi=$NW" --n-generated-label "wvi_ss=$NS")
  local VARIANT FF_MODEL="" FLOOR=""
  for VARIANT in "point off" "gauss on" "gauss off" "sphere on" "sphere off"; do
    FF_MODEL=${VARIANT%% *}
    FLOOR=${VARIANT##* }
    "$PYTHON" ../geant4_compare.py \
      "${FILE_ARGS[@]}" \
      "${GENERATED_ARGS[@]}" \
      --material "$MAT" \
      --thickness-cm "$T" \
      --p "$P" \
      --ff-model "$FF_MODEL" \
      --floor "$FLOOR" \
      --theta-cut-mrad 200 \
      --out "$OUT/compare/${MAT}_t${T}_p${P}_seed${SEED_INDEX}_${FF_MODEL}_${FLOOR}_compare.csv" \
      > "$OUT/logs/${MAT}_t${T}_p${P}_seed${SEED_INDEX}_${FF_MODEL}_${FLOOR}_compare.log" 2>&1
  done
  printf 'Completed %s t=%s p=%s seed-set=%s\n' "$MAT" "$T" "$P" "$SEED_INDEX"
}

counter=0
launched=0
for MAT in Cu Pb; do
  T=15.0
  for P in 1.0 2.0 3.5 6.0; do
    for SEED_INDEX in $(seq 1 "$N_SEEDS"); do
      counter=$((counter + 1))
      run_seed_set "$MAT" "$T" "$P" "$SEED_INDEX" "$counter" &
      launched=$((launched + 1))
      if (( launched % PARALLEL_JOBS == 0 )); then
        wait
      fi
    done
  done
done
wait
