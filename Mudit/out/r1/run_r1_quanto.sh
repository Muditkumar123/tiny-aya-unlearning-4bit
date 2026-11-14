#!/usr/bin/env bash
# R1 v2: identity, source lineage, loader/precision and excluded-weight checks on the five quanto checkpoints (CPU only).
cd ~/Lexsi
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:~/miniconda3/envs/lexsi/bin:$PATH
B=~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324
run() { echo "== $1 $(date '+%F %T')"; python Mudit/scripts/verify_quant.py "$2" --source "$3" --kind quanto --expect-q4 "$4" --expect-bf16 "$5" ${6:+--source-sha $6} --out "Mudit/out/r1/$1.json"; echo "exit $? $(date '+%F %T')"; }
run c1-q4 Mudit/out/c1-q4 Ganesh/out/c0v2-target-merged 252 0 4375fe12a7af731525019cffb9932d98a96d0f12567cbe72d757809959ff10a6
run c3-q4 Mudit/out/c3-q4 Ganesh/out/c2-lr3e-05-merged 252 0 a1cfe9ea440e0ab4d7f0eddc377d3625b20ceb3cdfa9c635647c42694cc795ca
run plain-q4 Mudit/out/plain-q4 $B 252 0
run random30-q4 Mudit/out/random30-q4 $B 175 77
run protected-q4 Mudit/out/refusal_circuit_matched/tiny-aya-refusal-protected-q4 $B 175 77
