#!/usr/bin/env bash
set -u
cd ~/projects/maldi-shift

run () {
  SP="$1"; DR="$2"; TG="$3"; NS="$4"; TAG="$5"
  OUTF="outputs/curve2/${SP// /_}__${DR}__${TG}.csv"
  if [ -f "$OUTF" ]; then echo "[$(date +%H:%M)] $TAG var, atlandi"; return; fi
  echo "[$(date +%H:%M)] BASLADI: $SP / $DR -> $TG"
  python learning_curve.py --species "$SP" --drug "$DR" --target "$TG" \
    --ns "$NS" --reps 10 --out outputs/curve2 > "outputs/curve2_${TAG}.log" 2>&1
  echo "[$(date +%H:%M)] BITTI: $TAG"
  sed -n '/PRAUC/,/^$/p' "outputs/curve2_${TAG}.log"
  echo "----------------------------------------"
}

run "Escherichia coli" "Ciprofloxacin" "DRIAMS-D" "25,50,100,200,500,1000" "ecoli_cip_D"
run "Klebsiella pneumoniae" "Ceftriaxone" "DRIAMS-D" "25,50,100,200,500,1000" "kpneu_cro_D"
run "Escherichia coli" "Ceftriaxone" "DRIAMS-C" "25,50,100,200,400" "ecoli_cro_C"
run "Escherichia coli" "Ciprofloxacin" "DRIAMS-C" "25,50,100,200,400" "ecoli_cip_C"
run "Staphylococcus aureus" "Oxacillin" "DRIAMS-C" "25,50,100,200,300" "saureus_oxa_C"
run "Escherichia coli" "Ceftriaxone" "DRIAMS-B" "25,50,100" "ecoli_cro_B"

echo "[$(date +%H:%M)] TUM EGRILER TAMAM"
