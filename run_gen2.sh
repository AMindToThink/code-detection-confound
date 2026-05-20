#!/bin/bash
cd /home/cs29824/matthew/code-detection-confound
for M in Qwen/Qwen2.5-3B-Instruct microsoft/Phi-3.5-mini-instruct; do
  echo "===== $(date +%H:%M:%S) generating $M ====="
  CUDA_VISIBLE_DEVICES=1 python3 scripts/02_generate_llm.py --model "$M" 2>&1 | grep -vE "Loading|Materializing|Downloading|Fetching" || echo "FAILED $M"
done
echo "===== ALL GEN DONE $(date +%H:%M:%S) ====="
