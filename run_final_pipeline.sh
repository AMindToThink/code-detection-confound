#!/bin/bash
cd /home/cs29824/matthew/code-detection-confound
# wait for OpenAI generation to finish (parquet present + no openai gen proc)
until [ -f data/gen/openai__gpt-5_4-nano.parquet ] && ! pgrep -f 02c_generate_openai >/dev/null; do sleep 10; done
echo "openai gen ready: $(python3 -c "import pandas as pd;print(len(pd.read_parquet('data/gen/openai__gpt-5_4-nano.parquet')))" 2>/dev/null) rows"
# wait for any in-flight detector/analysis procs (2-model pipeline) to clear -> free GPU
until ! pgrep -f 03_run_detectors >/dev/null && ! pgrep -f 04_analyze >/dev/null; do sleep 10; done
echo "===== FINAL 3-MODEL PIPELINE $(date +%H:%M:%S) ====="
CUDA_VISIBLE_DEVICES=1 python3 scripts/03_run_detectors.py 2>&1 | grep -vE "Loading|Materializing|Downloading|Fetching" | tail -12
python3 scripts/04_analyze.py 2>&1 | tail -4
python3 scripts/05_figures.py 2>&1 | tail -2
python3 scripts/06_report.py 2>&1 | tail -2
echo "===== FINAL PIPELINE DONE $(date +%H:%M:%S) ====="
