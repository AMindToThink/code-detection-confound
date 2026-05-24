# Archived scripts

These scripts represent earlier experiment paths that were superseded but are kept for
traceability and to ensure a future Claude session doesn't accidentally re-attempt them
without context.

## The `11_*` family — abandoned 2026-05-22

A custom ModernBERT-base fine-tune on DroidCollection-Python with `black`-normalized
training data. The intent was the same as the current line of work (test the
formatting confound), but the *implementation* violated the project's "import, don't
reimplement" rule because DroidDetect's training code is not public so we built our
own training recipe.

User feedback (verbatim): *"This is comical. I have rules, I ask you so many times, do
not reimplement, and you reimplement, and you don't notice until I ask."*

Pivoted to **CodeGPTSensor (Xu et al., TOSEM 2025)**, which IS fully open-source. The
canonical training entry point is now `scripts/18_train_cgs_amp.py`.

Do not resurrect these scripts without explicit user authorization. If a future
session is tempted to use `11_train.py`'s telemetry / watchdog patterns, they are
already extracted into `~/.claude/skills/supervising-training-runs/SKILL.md`.

## Files

- `11_prepare_data.py` — DroidCollection-Python parquet prep with class balancing.
- `11_train.py` — ModernBERT-base sequence classification trainer.
- `11b_make_variants.py` — `black`-formatted, AST-roundtripped, and comment-stripped
  test-set variants.
- `11c_eval.py` — per-variant eval driver.
- `11d_watchdog.py` — JSONL-tail watchdog (the pattern survives in the skill).
- `11e_plot.py` — matplotlib summary of training curves.
