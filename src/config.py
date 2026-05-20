"""Shared configuration for the skill-vs-species code-detection experiment.

Single source of truth for paths, the skill axis, the prompt conditions, the model
panel, and the cell taxonomy. See DESIGN.md for the rationale behind each choice.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _d in (DATA, RESULTS, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

HUMAN_PARQUET = DATA / "human_code.parquet"
LLM_PARQUET = DATA / "llm_code.parquet"
ALL_SAMPLES_PARQUET = DATA / "all_samples.parquet"          # human+llm, pre-detector
SCORED_PARQUET = DATA / "scored_samples.parquet"            # + detector scores + features

# ---------------------------------------------------------------- GPU / disk discipline
# We were granted GPU1 only (GPU0 belongs to another user) for an 8h window.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# Reuse the shared HF cache; never download more than necessary (~34 GB free disk).

# ---------------------------------------------------------------- skill axis
# DEVIATION (see DESIGN.md #1): the ideal axis is *author Elo*, but no public dataset
# co-locates human source code with author rating within budget. We use *problem
# difficulty* as a noisy, asymmetric proxy. SKILL_SOURCE is recorded in every artifact
# so a future author-rating run can be swapped in without changing downstream code.
SKILL_SOURCE = "codeforces_problem_difficulty"

# Difficulty bands (Codeforces problem rating). Bands are deliberately separated by a
# gap to make EASY/HARD stylistically distinct; mid-range problems are dropped.
EASY_MAX = 1200      # <= EASY_MAX  -> "easy" band
HARD_MIN = 1900      # >= HARD_MIN  -> "hard" band
BANDS = ("easy", "hard")

def band_of(problem_rating: int | float | None) -> str | None:
    if problem_rating is None:
        return None
    r = float(problem_rating)
    if r <= EASY_MAX:
        return "easy"
    if r >= HARD_MIN:
        return "hard"
    return None  # mid-range, excluded

# ---------------------------------------------------------------- prompt conditions
# DEVIATION (DESIGN.md #4): expert is load-bearing, not an afterthought.
CONDITIONS = ("natural", "novice", "expert")

PROMPT_TEMPLATES: dict[str, str] = {
    "natural": (
        "Solve this competitive programming problem in Python 3. "
        "Read from standard input and write to standard output.\n\n{problem}\n\n"
        "Return only the Python code."
    ),
    "novice": (
        "Solve this competitive programming problem in Python 3. "
        "Read from standard input and write to standard output.\n\n{problem}\n\n"
        "Write it the way a beginner who is still learning Python would: use only basic "
        "constructs, no type hints, short single-letter variable names where natural, and "
        "minimal or no comments. Return only the Python code."
    ),
    "expert": (
        "Solve this competitive programming problem in Python 3. "
        "Read from standard input and write to standard output.\n\n{problem}\n\n"
        "Write production-quality, idiomatic Python: clear descriptive names, type hints, "
        "helper functions, and concise docstrings/comments. Return only the Python code."
    ),
}

# ---------------------------------------------------------------- LLM generation panel
# All cached open-weight INSTRUCT (RLHF'd) models -> sit toward the "assistant style"
# attractor the hypothesis is about. Ordered cheapest-first for the preliminary run.
LLM_PANEL: list[str] = [
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
# Qwen/Qwen2.5-14B-Instruct is cached and can be added if time permits (slower on Turing).

# ---------------------------------------------------------------- detectors
DROIDDETECT_REPO = "project-droid/DroidDetect-Base-Binary"   # trained family
# Statistical family base models (cached, small, fast on one RTX 8000):
FASTDETECT_SCORING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BINOCULARS_OBSERVER = "Qwen/Qwen2.5-0.5B-Instruct"           # performer / scorer
BINOCULARS_PERFORMER = "Qwen/Qwen2.5-1.5B-Instruct"          # reference

DETECTOR_FAMILY = {
    "DroidDetect": "trained",
    "FastDetectGPT": "statistical",
    "Binoculars": "statistical",
}

# ---------------------------------------------------------------- preliminary scale
# "Clear evidence after a smaller number of samples" (Matthew). Scale up if time remains.
N_PROBLEMS_PER_BAND = 60
MAX_HUMAN_PER_PROBLEM = 6
N_GEN_PER_MODEL_COND = 2          # generations per (problem, model, condition)
GEN_SEED = 0

# ---------------------------------------------------------------- cell taxonomy
# A "cell" pairs a human difficulty band with an LLM prompt condition for binary
# (human vs LLM) discrimination. matched := the two sides share a skill/style register.
def cell_kind(band: str, condition: str) -> str:
    matched = {("hard", "natural"), ("easy", "novice")}
    mismatched = {("easy", "natural"), ("hard", "novice")}
    if (band, condition) in matched:
        return "matched"
    if (band, condition) in mismatched:
        return "mismatched"
    return "other"   # expert-prompt cells: symmetry control, analysed separately
