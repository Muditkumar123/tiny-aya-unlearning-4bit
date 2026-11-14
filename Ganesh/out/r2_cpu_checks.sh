#!/bin/bash
# R2 CPU checks (PLAN v1 section 4): merge bitwise + identity + reload for C0v2 and C2. No GPU, no download.
cd ~/Lexsi
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
PY=~/miniconda3/envs/lexsi/bin/python
echo "start $(date '+%F %H:%M:%S')"
$PY Ganesh/scripts/verify_merge.py CohereLabs/tiny-aya-global Ganesh/out/c0v2-adapter Ganesh/out/c0v2-target-merged \
  --revision af89d219b53ed9b13b8a4645f9c8028973510324 --expect-linears 252 --out Ganesh/out/r2_merge_c0v2.json; echo "c0v2 exit=$? $(date '+%H:%M:%S')"
$PY Ganesh/scripts/verify_merge.py Ganesh/out/c0v2-target-merged Ganesh/out/c2-lr3e-05-adapter Ganesh/out/c2-lr3e-05-merged \
  --expect-linears 252 --out Ganesh/out/r2_merge_c2.json; echo "c2 exit=$? $(date '+%H:%M:%S')"
