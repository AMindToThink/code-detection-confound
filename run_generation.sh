#!/bin/bash
set -e
cd /home/cs29824/matthew/code-detection-confound
for M in google/gemma-2-2b-it Qwen/Qwen2.5-3B-Instruct microsoft/Phi-3.5-mini-instruct meta-llama/Llama-3.1-8B-Instruct; do
  echo "===== $(date +%H:%M:%S) generating $M ====="
  CUDA_VISIBLE_DEVICES=1 python3 scripts/02_generate_llm.py --model "$M" 2>&1 | grep -vE "Loading|Materializing|Downloading|Fetching" || echo "FAILED $M"
done
echo "===== ALL GENERATION DONE $(date +%H:%M:%S) ====="
