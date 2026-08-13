#!/usr/bin/env bash
# Yeniden kalibrasyon: kalan dort cift
set -u
cd ~/projects/maldi-shift

run () {
  SP="$1"; DR="$2"; TAG="$3"
  OUTF="outputs/recal/${SP// /_}__${DR}.csv"
  if [ -f "$OUTF" ]; then
    echo "[$(date +%H:%M)] $TAG zaten var, atlandi"; return
  fi
  echo "[$(date +%H:%M)] BASLADI: $SP / $DR"
  python recalibrate.py --species "$SP" --drug "$DR" \
    --ncal 25,50,100,200,400 --reps 30 > "outputs/recal_${TAG}.log" 2>&1
  echo "[$(date +%H:%M)] BITTI: $TAG"
  tail -22 "outputs/recal_${TAG}.log"
  echo "----------------------------------------"
}

run "Escherichia coli" "Ceftriaxone" "ecoli_cro"
run "Escherichia coli" "Ciprofloxacin" "ecoli_cip"
run "Klebsiella pneumoniae" "Ceftriaxone" "kpneu_cro"
run "Pseudomonas aeruginosa" "Ciprofloxacin" "paer_cip"

echo "[$(date +%H:%M)] YENIDEN KALIBRASYON TAMAM"
