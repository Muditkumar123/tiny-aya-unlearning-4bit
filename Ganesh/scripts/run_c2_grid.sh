#!/bin/bash
# C2 grid on GPU 0 (Mudit, board 09:42): train + merge + score each lr in ascending order, stop at the first
# that meets the frozen rule (Ganesh/configs/c2.yaml). Waits for the GPU-0 job given as $1 (a pid) to end first.
set -e
cd ~/Lexsi
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$HOME/miniconda3/envs/lexsi/bin:$PATH CUDA_VISIBLE_DEVICES=0
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done
echo "GPU 0 start $(date '+%F %H:%M')"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for lr in 1e-05 3e-05 1e-04; do
  echo "=== lr $lr train $(date '+%H:%M')"
  python Ganesh/scripts/unlearn_c2.py --lr $lr
  echo "=== lr $lr score $(date '+%H:%M')"
  # the harness writes summary.json before its table step; only a missing summary is fatal
  python Vedika/scripts/tofu_eval.py c2-lr$lr Ganesh/out/c2-lr$lr-merged --out Ganesh/out/tofu_c2grid \
    || [ -f Ganesh/out/tofu_c2grid/c2-lr$lr/summary.json ] || exit 1
  set +e; python Ganesh/scripts/select_c2.py; code=$?; set -e
  if [ $code -eq 0 ]; then
    chosen=$(python -c "import json; print(f\"{json.load(open('Ganesh/out/c2_selection.json'))['chosen_lr']:.0e}\")")
    ln -sfn "c2-lr$chosen-merged" Ganesh/out/c2-unlearned-merged
    echo "C2_CHOSEN lr=$chosen $(date '+%H:%M')"; exit 0
  fi
  [ $code -eq 4 ] && { echo "C2_STOP: every lr collapsed retain retention $(date '+%H:%M')"; exit 4; }
done
