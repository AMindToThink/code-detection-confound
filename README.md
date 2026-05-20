# Skill vs. Species in AI Code Detection

Does an "AI code detector" detect author **species** (human vs LLM) or author
**skill/style**? Post-RLHF code models imitate idiomatic "senior-engineer" style, so a
detector trained on AI-vs-human labels may really be an expert-vs-novice classifier.
This repo is a preliminary autonomous experiment testing that.

See **`DESIGN.md`** for the full design and the deviations from `initial_plans.html`
(decided after three fresh-context critique agents + the real 8h / 1×RTX8000 budget).
The headline deviation: no public dataset joins human *source code* to *author Elo*, so
the human skill axis is **problem difficulty** (a noisy proxy); the **clean** skill axis
is the LLM-side natural/novice/expert prompt manipulation.

## Pipeline

```
scripts/01_build_human_data.py   # Codeforces (MatrixStudio) -> human_code + problems
scripts/02_generate_llm.py       # HF batched gen (one model/proc) -> data/gen/<model>.parquet
scripts/02b_legendary_extract.py # pre-LLM code: Torvalds/antirez/Hipp/CPython -> legendary_code
scripts/03_run_detectors.py      # assemble + style features + 3 detectors -> scored_samples
scripts/04_analyze.py            # AUROC cells, Δ_confound, within-species slopes, mediation
scripts/05_figures.py            # figures -> results/figures/
scripts/06_report.py             # results/report.md (numbers from analysis.json only)
```

Run order: `01 → 02 (./run_generation.sh) + 02b → 03 → 04 → 05 → 06`.

## Detectors (higher score = more "AI")
- **DroidDetect-Base-Binary** (trained family) — faithful TLModel reconstruction.
- **Fast-DetectGPT**, **Binoculars** (statistical family) — **vendored verbatim** from the
  official repos (`src/vendor/`), not re-derivations.

## Environment notes (shared `dfc` conda env)
- `src/_env.py` shims an ABI-broken torchvision so transformers/ModernBERT import.
- vLLM unusable here (wheel/torch ABI mismatch); generation uses HF transformers.
- GPU1 only; ~34 GB free disk → prefer cached models, no large downloads.

## Key results
Generated into `results/report.md`, `results/figures/`, `results/tables/paper_macros.json`.
