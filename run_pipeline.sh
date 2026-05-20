#!/bin/bash
# Reproducible end-to-end pipeline for the skill-vs-species code-detection experiment.
# Idempotent: data-collection / generation stages are skipped if their output exists;
# the scoring -> analysis -> figures -> report stages always re-run.
#
# Usage:
#   ./run_pipeline.sh            # full pipeline (skips existing data artifacts)
#   ./run_pipeline.sh analyze    # only scoring -> analysis -> figures -> report
#   FORCE_GEN=1 ./run_pipeline.sh   # also (re)generate LLM code
#
# GPU: uses GPU1 only (CUDA_VISIBLE_DEVICES=1). Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
PY=python3
STAGE="${1:-all}"

log() { echo "===== $(date +%H:%M:%S) $* ====="; }

if [ "$STAGE" = "all" ]; then
  # ---- 1. human Codeforces cohort
  if [ ! -f data/human_code.parquet ]; then
    log "01 human data"; $PY scripts/01_build_human_data.py
  else log "01 human data (cached)"; fi

  # ---- 2. LLM generation (open-weight panel). Slow; only with FORCE_GEN=1 or if empty.
  if [ "${FORCE_GEN:-0}" = "1" ] || [ -z "$(ls data/gen/*.parquet 2>/dev/null)" ]; then
    log "02 LLM generation (open-weight panel)"
    for M in google/gemma-2-2b-it Qwen/Qwen2.5-3B-Instruct; do
      $PY scripts/02_generate_llm.py --model "$M"
    done
  else log "02 LLM generation (cached: $(ls data/gen/*.parquet | wc -l) models)"; fi

  # ---- 2c. frontier model via OpenAI API (needs OPENAI_API_KEY in .env)
  if [ -f .env ] && grep -q OPENAI_API_KEY .env && [ ! -f data/gen/openai__gpt-5_4-nano.parquet ]; then
    log "02c OpenAI gpt-5.4-nano generation"; $PY scripts/02c_generate_openai.py --model gpt-5.4-nano || true
  fi

  # ---- 2b. legendary pre-LLM code
  if [ ! -f data/legendary_code.parquet ]; then
    log "02b legendary pre-LLM extraction"; $PY scripts/02b_legendary_extract.py
  else log "02b legendary (cached)"; fi

  # ---- 7. AtCoder author-rating confirmatory cohort (scrape; slow, network)
  if [ ! -f data/atcoder_human.parquet ]; then
    log "07 AtCoder author-rating scrape"; $PY scripts/07_atcoder_human.py || true
  else log "07 AtCoder (cached)"; fi
fi

# ---- 3-6. scoring -> analysis -> figures -> report (always)
log "03 detector scoring";   $PY scripts/03_run_detectors.py
log "04 analysis";           $PY scripts/04_analyze.py
log "05 figures";            $PY scripts/05_figures.py
log "06 build macros (variables only)"; $PY scripts/06_build_macros.py
log "07 compile paper.tex -> paper.pdf"; ( cd paper && latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null 2>&1 ) \
  && echo "paper/paper.pdf built" || echo "WARN: latex compile failed (edit paper/paper.tex; macros are in paper/macros.tex)"
log "DONE — edit prose in paper/paper.tex; numbers come from paper/macros.tex; figures in results/figures/"
