# Skill vs. Species in AI Code Detection — Design (preliminary run)

Derived from `initial_plans.html`, revised after three fresh-context critique agents
(design validity, statistics/power, feasibility) and the real resource constraints on
this machine (8h, single RTX 8000 / GPU1, ~34 GB free disk, no paid frontier APIs,
HF cache only). Matthew asked for **preliminary results fast** with smaller samples.

## Core question (unchanged)
Do AI-code detectors detect author **species** (human vs LLM) or author **skill/style**?
If post-RLHF code models sit at the "senior-engineer style attractor," detectors trained
on AI-vs-human labels may really be expert-vs-novice classifiers.

## Material deviations from `initial_plans.html` (with rationale)

1. **Skill axis = problem difficulty, NOT author Elo.** No public dataset co-locates
   human *source code* with *author rating*. `MatrixStudio/Codeforces-Python-Submissions`
   has code + problem `rating` + `verdict` + `test_cases` but no author handle;
   `denkCF/...` has `rating_at_submission` but no source. Scraping source page-by-page
   (Cloudflare, ~1 req/2s) does not fit 8h. Problem difficulty is a **noisy, asymmetric**
   skill proxy (hard accepted solution ⇒ skilled author; easy solution ⇒ uninformative).
   The cleaner axis is the LLM-side **prompt manipulation** (experimental, not observational).
   *This is the single biggest caveat and is flagged prominently in the report.*

2. **Primary metric = AUROC, not F1.** Both stats critiques: F1 is threshold- and
   prevalence-dependent, and prevalence differs across cells by construction, so a raw
   F1 gap conflates skill-signal collapse with base-rate drift. AUROC is threshold-free
   and base-rate-robust. F1 heatmap demoted to illustration. No per-cell threshold tuning.

3. **Within-species slope is the mechanism test (new primary).** Regress
   logit(detector score) on the style axis *within each species*. Pure species classifier
   ⇒ within-species slope ≈ 0. Skill classifier ⇒ large slopes of predicted sign
   (humans: harder→more "AI"; LLMs: novice→less "AI"). Robust to base rates.

4. **Expert-prompt promoted to load-bearing** (was "optional sanity check"). Needed to
   separate "detector tracks skill" from "novice-prompted output is merely OOD." Conditions:
   `natural`, `novice`, `expert`.

5. **Surface-feature ablation added.** Regress detector score on hand-coded style features
   (comment density, type-hint presence, mean identifier length, line length, blank-line
   ratio). Tests the mundane alternative: detector = comment/type-hint detector. Also
   measure the difficulty→style relationship directly on humans (does hard-problem code
   actually look more "expert"?).

6. **Cluster-bootstrap over PROBLEMS for all CIs** (not over samples or detectors).
   Samples are clustered within problems; report Δ_confound **per detector**, never averaged.

7. **No acceptance-filtering of LLM code in the main analysis.** Acceptance is a collider
   (rating → accept ← difficulty); humans are accepted-only (MatrixStudio), LLMs kept
   unfiltered. We *record* LLM acceptance via local `test_cases` judging as a covariate and
   report with/without, but do not condition the primary analysis on it.

## Cells (2 difficulty bands x 3 prompt conditions, + species)
Human difficulty bands: EASY (rating <= 1200), HARD (rating >= 1900).
LLM conditions: natural / novice / expert.

Skill-MATCHED cells (detector should struggle if it is a skill classifier):
  - HARD-problem humans  vs  natural-prompt LLM   (both "expert-ish")
  - EASY-problem humans  vs  novice-prompt LLM     (both "novice-ish")
Skill-MISMATCHED cells (detector should do well under both hypotheses):
  - EASY-problem humans  vs  natural-prompt LLM
  - HARD-problem humans  vs  novice-prompt LLM
Headline: Δ_confound = AUROC_mismatched − AUROC_matched, per detector, problem-bootstrapped.

## Detectors
- Trained family: `project-droid/DroidDetect-Base` if it exists on the hub (verify);
  collapse multiclass to binary P(AI). Fallback: fine-tune a small encoder on a DroidCollection
  slice, or a TF-IDF+logreg trained detector, to still represent the "trained" family.
- Statistical family: Fast-DetectGPT and Binoculars, using cached small base LMs
  (Qwen2.5-0.5B/1.5B-Instruct pair for Binoculars; one scoring model for Fast-DetectGPT).
- Between-family prediction: trained detectors show larger Δ_confound than statistical.

## LLM panel (cached open-weight instruct models; all RLHF'd)
Qwen2.5-3B-Instruct, Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct, gemma-2-2b-it, Phi-3.5-mini.
(Frontier models via OpenRouter optional if a key is provided — strengthens "RLHF attractor"
claim but not required for preliminary evidence.)

## Preliminary scale (fast, clear-signal-first)
~60 problems/band x 2 bands; up to N human solutions/problem from MatrixStudio;
LLM: subset of panel x 3 conditions x n=2 per problem. Scale up only if time remains.

## Hard constraints
GPU1 only (`CUDA_VISIBLE_DEVICES=1`). Edit nothing outside this folder. Commit as we go,
identity AMindToThink. Watch disk (~34 GB free) — prefer cached models, avoid large downloads.
