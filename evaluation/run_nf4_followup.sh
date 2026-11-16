#!/usr/bin/env bash
# Build NF4 copies and score bf16 vs NF4. Usage, from the repository root: evaluation/run_nf4_followup.sh c0 npo
# Outputs go to runs/nf4_followup/ (ignored by Git: it holds model answers).
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=1
OUT=runs/nf4_followup
mkdir -p $OUT
for m in "$@"; do
  case $m in c0) src=HF/models/c0v2 ;; npo) src=HF/models/c2 ;; mansu) src=$OUT/floor/checkpoint ;; *) echo "unknown $m"; exit 2 ;; esac
  sha=$(sha256sum "$(readlink -f $src/model.safetensors)" | cut -d' ' -f1)
  echo "[$(date +%H:%M)] $m source sha256 $sha"
  $PY -B -u compression/build_mansu_nf4.py $src $OUT/nf4_$m --expect-source-sha $sha > $OUT/nf4_build_$m.log 2>&1 \
    || { echo "NF4 build of $m failed, see $OUT/nf4_build_$m.log"; continue; }
  for p in bf16 nf4; do
    ckpt=$src; [ $p = nf4 ] && ckpt=$OUT/nf4_$m
    $PY -B -u evaluation/nf4_followup.py mcq --checkpoint $ckpt --precision $p --label ${m}_$p > $OUT/mcq_${m}_$p.log 2>&1 \
      || echo "MCQ scoring of ${m}_$p failed"
    $PY -B -u evaluation/run_mansu_comparison.py --checkpoint $ckpt --tokenizer HF/models/c0v2 \
      --inputs runs/mansu_20261115/inputs --label ${m}_$p --precision $p --out $OUT/evaluation > $OUT/eval_${m}_$p.log 2>&1 \
      || echo "scoring of ${m}_$p failed"
    echo "[$(date +%H:%M)] ${m}_$p done: $(grep -h forget_mcq $OUT/mcq_${m}_$p.log | cut -c1-200)"
  done
done
$PY -B evaluation/nf4_followup.py contrast --methods "$@"
