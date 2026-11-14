#!/bin/bash
# R4a (PLAN, proposed): fresh full TOFU rescore of the original bf16 model on GPU 0, via Ganesh's reviewed wrapper (md5 c66903dc) unchanged.
# Starts only after Vedika's OK and a fresh GPU 0 reservation. Usage: run.sh <out_dir> [--limit N]
cd ~/Lexsi
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
PY=~/miniconda3/envs/lexsi/bin/python
SNAP=~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324
OUT=$1; shift
[ -e "$OUT/original" ] && { echo "$OUT/original exists; use a fresh out dir"; exit 1; }
echo "start $(date '+%F %H:%M:%S') out=$OUT args=$*"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
md5sum Ganesh/scripts/r2_rescore.py Ganesh/scripts/compare_regen.py Vedika/scripts/tofu_eval.py Vedika/scripts/evaluate.py common/addons/inmemory_model.py
# Pinned identity: base shard sha256 as recorded in R1 (Mudit/out/r1/plain-q4.json).
declare -A SHA=([model-00001-of-00002.safetensors]=0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1
                [model-00002-of-00002.safetensors]=2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab)
for f in "${!SHA[@]}"; do
  s=$(sha256sum "$SNAP/$f" | cut -d' ' -f1)
  [ "$s" = "${SHA[$f]}" ] || { echo "$f sha256 $s != R1 reference; stop"; exit 1; }
  echo "$f sha256 matches R1"
done
$PY Ganesh/scripts/r2_rescore.py original "$SNAP" --out "$OUT" "$@"; rc=$?
echo "original exit=$rc $(date '+%H:%M:%S')"
echo "end $(date '+%F %H:%M:%S')"
exit $rc
