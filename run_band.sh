#!/usr/bin/env bash
# Bant ablasyonu: dusuk kutle (2-4 kDa) bolgesini atinca ne oluyor?
set -u
cd ~/projects/maldi-shift
mkdir -p outputs/cv_band

run () {
  SP="$1"; DR="$2"; TAG="$3"
  OUTF="outputs/cv_band/${SP// /_}__${DR// /_}.csv"
  if [ -f "$OUTF" ]; then
    echo "[$(date +%H:%M)] $TAG zaten var, atlandi"; return
  fi
  echo "[$(date +%H:%M)] BASLADI: $SP / $DR  (mz 4000+)"
  python nested_cv.py --species "$SP" --drug "$DR" --mz-min 4000 \
    --out outputs/cv_band --seeds 3 --folds 5 > "outputs/cv_band_${TAG}.log" 2>&1
  echo "[$(date +%H:%M)] BITTI: $TAG"
  tail -8 "outputs/cv_band_${TAG}.log"
  echo "----------------------------------------"
}

run "Klebsiella pneumoniae" "Ceftriaxone" "kpneu_cro"
run "Escherichia coli" "Ceftriaxone" "ecoli_cro"
run "Escherichia coli" "Ciprofloxacin" "ecoli_cip"
run "Pseudomonas aeruginosa" "Ciprofloxacin" "paer_cip"

echo "[$(date +%H:%M)] BANT ABLASYONU TAMAM"
