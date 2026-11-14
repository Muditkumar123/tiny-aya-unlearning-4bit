#!/bin/bash
# R2 full rescore (PLAN v1 section 4 fallback; Codex 01:54) on GPU 1: C0v2 then C2, all rows, unchanged harness.
cd ~/Lexsi
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
PY=~/miniconda3/envs/lexsi/bin/python
echo "start $(date '+%F %H:%M:%S')"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
$PY Ganesh/scripts/r2_rescore.py c0v2 Ganesh/out/c0v2-target-merged; echo "c0v2 exit=$? $(date '+%H:%M:%S')"
$PY Ganesh/scripts/r2_rescore.py c2 Ganesh/out/c2-lr3e-05-merged; echo "c2 exit=$? $(date '+%H:%M:%S')"
# Descriptive only: the fresh rows replace the saved ones; this shows how far the saved ones differ.
$PY Ganesh/scripts/compare_regen.py Ganesh/out/r2_rescore/c0v2 Vedika/out/tofu_v2/c0v2 --out Ganesh/out/r2_rescore_vs_saved_c0v2.json > /dev/null; echo "c0v2 compare exit=$?"
$PY Ganesh/scripts/compare_regen.py Ganesh/out/r2_rescore/c2 Vedika/out/tofu_v2/c2 --out Ganesh/out/r2_rescore_vs_saved_c2.json > /dev/null; echo "c2 compare exit=$?"
echo "end $(date '+%F %H:%M:%S')"
