#!/usr/bin/env bash
# Kalan tur-ilac ciftlerini sirayla kosar.
set -u
cd ~/projects/maldi-shift

run () {
  SP="$1"; DR="$2"; TAG="$3"
  LOG="outputs/cv_${TAG}.log"
  if [ -f "outputs/cv/${SP// /_}__${DR// /_}.csv" ]; then
    echo "[$(date +%H:%M)] $TAG zaten var, atlandi"; return
  fi
  echo "[$(date +%H:%M)] BASLADI: $SP / $DR"
  python nested_cv.py --species "$SP" --drug "$DR" --seeds 3 --folds 5 > "$LOG" 2>&1
  echo "[$(date +%H:%M)] BITTI: $TAG"
  tail -6 "$LOG"
  echo "----------------------------------------"
}

run "Escherichia coli" "Ciprofloxacin" "ecoli_cip"
run "Klebsiella pneumoniae" "Ceftriaxone" "kpneu_cro"
run "Pseudomonas aeruginosa" "Ciprofloxacin" "paer_cip"

echo "[$(date +%H:%M)] TUM DENEYLER TAMAM"
