#!/bin/bash
# Full fresh TOFU rescore of C1 and C3 (qint4) on GPU 0 with the unchanged harness, via Ganesh's reviewed wrapper (md5 c66903dc).
# Usage: run.sh <out_dir> [--limit N]
cd ~/Lexsi
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:~/miniconda3/envs/lexsi/bin:$PATH
PY=~/miniconda3/envs/lexsi/bin/python
OUT=$1; shift
echo "start $(date '+%F %H:%M:%S') out=$OUT args=$*"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
md5sum Ganesh/scripts/r2_rescore.py Ganesh/scripts/compare_regen.py Vedika/scripts/tofu_eval.py Vedika/scripts/evaluate.py common/addons/inmemory_model.py
declare -A SHA=([c1]=1e3422875f548ccb2947c92efb0fa1ad3216b74c4dec0db2d52247e06a4798d3 [c3]=f4a3b4a6edd3c2d347af324acd12ff1f28d97ebfca8db98db909d1fcf5f1a02f)
for c in c1 c3; do
  s=$(sha256sum Mudit/out/$c-q4/model.safetensors | cut -d' ' -f1)
  [ "$s" = "${SHA[$c]}" ] || { echo "$c sha256 $s != R1 reference; stop"; exit 1; }
  echo "$c sha256 matches R1 (Mudit.md 02:04)"
  $PY Ganesh/scripts/r2_rescore.py $c Mudit/out/$c-q4 --out $OUT "$@"; rc=$?; echo "$c exit=$rc $(date '+%H:%M:%S')"; [ $rc -eq 0 ] || fail=1
done
echo "end $(date '+%F %H:%M:%S')"
exit ${fail:-0}
