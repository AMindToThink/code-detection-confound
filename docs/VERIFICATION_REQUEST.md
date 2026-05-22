# Request for human verification

> **This repository was produced autonomously by an AI agent (Claude Code) and has NOT been
> verified by a human.** This document lists what a human reviewer should check before any
> result here is trusted, cited, or acted on. Use it as the body of a tracking issue.

## Headline claim to verify
**"DroidDetect's verdict is gated by code formatting."** Running its own in-distribution human
test data through `black` flips the fraction it labels "machine" from ~1% to ~83%.

- [ ] **Reproduce the ablation.** Run `scripts/08_format_ablation.py` and confirm
      `results/format_ablation.json` shows DroidCollection-human `original` ≈ 0.01 and `black` ≈ 0.83,
      with machine code ≈ unchanged. Confirm `black` was actually applied (not a no-op on unparseable
      files) and that the `transform_success` rate is high.
- [ ] **Check "semantics-preserving" is fair.** Confirm identifiers/literals/docstrings are preserved
      (`identifier_preservation_ast_canon` ≈ 1.000) and decide whether `black`'s quote-style/trailing-comma
      normalization should count as "formatting" or as a legitimate authorship signal.
- [ ] **In-distribution control.** Confirm `scripts/00_validate_droiddetect.py` reproduces the published
      AUROC (~0.999) on the authors' own data — i.e. the load is faithful and the flip is real model behavior.

## Family-wide claim (Finding 7)
- [ ] Confirm `results/other_detectors.json` shows all three DroidDetect siblings flipping ~1%→~80% under
      `black` (Table in `paper/paper.tex` §"family"). Confirm the load of the 4-class and Large checkpoints
      is correct (`strict=True`, 0 missing keys).

## Statistics
- [ ] The skill result is a **null**. Verify it is presented as a *bounded* null (CI upper bound stated),
      not as proven absence, and that the floor effect (mediocre AUROC ~0.75) and the accepted-only collider
      are acknowledged.
- [ ] AtCoder high band is n=9. Confirm it is treated as suggestive only (slope CI includes 0).
- [ ] Check the bootstrap is clustered over problems and that no single CI's exclusion of zero is load-bearing.

## Numbers and citations
- [ ] Every number in the prose is a `\res…` macro from `paper/macros.tex` (run `pytest tests/test_paper_macros.py`).
- [ ] Every citation resolves and the metadata is correct (`scripts/verify_cites.py`; spot-check `paper/refs.bib`
      against the real papers — esp. author lists).
- [ ] `gpt-5.4-nano` is the model actually used (OpenAI API, May 2026), not a typo.

## Known open questions (NOT yet resolved)
- The confound's **generality beyond DroidCollection's data recipe** is untested: all detectors share authors,
  data, and the ModernBERT backbone. The decisive experiment (an *independent*-architecture detector trained on
  DroidCollection, and/or DroidDetect retrained on `black`-normalized data) is planned but not run —
  see `docs/TRAINING_PLAN.md`.
- The training-data formatting-cleanliness metric (15% vs 44% black-compliant) is not length-controlled and is
  labeled "suggestive."
