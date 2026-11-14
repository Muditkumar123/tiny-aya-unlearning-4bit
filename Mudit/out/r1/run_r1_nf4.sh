#!/usr/bin/env bash
# R1 NF4 reload checks on GPU 0, with the audited verify_quant.py.
cd ~/Lexsi
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:~/miniconda3/envs/lexsi/bin:$PATH
md5sum Mudit/scripts/verify_quant.py
run() { echo "== $1 $(date '+%F %T')"; python Mudit/scripts/verify_quant.py "$2" --source "$3" --kind nf4 --expect-nf4 252 --source-sha "$4" --out "Mudit/out/r1/$1.json"; echo "exit $? $(date '+%F %T')"; }
run c1-nf4 Mudit/out/c1-nf4 Ganesh/out/c0v2-target-merged 4375fe12a7af731525019cffb9932d98a96d0f12567cbe72d757809959ff10a6
run c3-nf4 Mudit/out/c3-nf4 Ganesh/out/c2-lr3e-05-merged a1cfe9ea440e0ab4d7f0eddc377d3625b20ceb3cdfa9c635647c42694cc795ca
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
