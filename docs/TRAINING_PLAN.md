# Plan: train our own detector to prove the formatting confound is *in the data*

> Status: **proposed, not run.** Needs a GPU window (GPU1 only) and your go-ahead.
> Done by AI, not verified by human.

## Why train anything at all?

The current evidence (the `black` ablation + the family-wide Finding 7) shows the *shipped*
DroidDetect checkpoints are gated by formatting. It does **not** yet prove *where* the confound
lives. Two alternative explanations survive everything we've run, because every detector we tested
is the same family (same authors, same DroidCollection data, same ModernBERT backbone):

1. **Data explanation (our headline):** the training corpus has a formatting imbalance — machine
   code is born auto-formatted, human code is scraped as-is — so *any* model minimizing loss on it
   learns "clean layout ⇒ machine." If true, the confound is a property of the data and will appear
   in models we train ourselves.
2. **Recipe / tokenizer explanation (the alternative we can't currently rule out):** the flip is an
   artifact of *how DroidDetect specifically processes code* — most plausibly its tokenizer (more on
   this below) — rather than the human/machine split in the data.

Training our own models is the only way to separate these. Two experiments do it.

---

## Experiment A — retrain DroidDetect on `black`-normalized data (your proposal)

**Idea.** If the confound is the train-time formatting *imbalance* between the two classes, then
making the two classes formatting-identical at training time should destroy the shortcut. Run
`black` over **both** the human and machine training code so neither class has a formatting
signature, then train the same architecture and see what's left.

**Procedure.**
1. **Pipeline-validation control (must run first).** Train a ModernBERT sequence classifier on the
   *original, unmodified* DroidCollection (a fixed subset, e.g. 80k/class). Confirm we reproduce
   DroidDetect-like in-distribution AUROC (~0.99). *Without this control, a low AUROC from the
   treatment model below is uninterpretable — it could just mean our training loop is bad.*
2. **Treatment.** `black`-format every training example (both classes), train the identical setup.
3. **Evaluate** both models on a held-out test split, each in two formatting regimes (raw and
   `black`-normalized), and measure: (a) AUROC, (b) the human→`black` flip rate (our headline probe).

**Decision rules (what each outcome proves).**
- If the control reproduces ~0.99 **and** the treatment's AUROC on formatting-normalized test data
  **drops sharply** (e.g. 0.99 → ~0.6): formatting was carrying most of the discriminative signal.
  This is the cleanest possible quantification of "how much of DroidDetect is formatting."
- If the treatment model **no longer flips** human code under `black` at test time: we've shown the
  confound is *removable by fixing the data*, i.e. it was a data-collection artifact, not an
  intrinsic limit of the task. (This is the constructive, "here's the fix" version of the result.)
- If the treatment **keeps** high AUROC and **doesn't** flip: there is a real authorship signal
  underneath, and DroidDetect simply failed to prioritize it — also a strong, publishable result.

This experiment is the most *direct* test of our headline mechanism, and it's the one you proposed.

---

## Experiment B — independent architecture / **different tokenizer** (the control I was urging)

**What "tokenizer artifact" means, concretely.** A transformer never sees characters; it sees
*tokens*. ModernBERT splits text into subword tokens using one specific learned BPE vocabulary.
When `black` reformats code, it changes whitespace, indentation, quote characters, and line breaks —
and **that changes how the text gets chopped into tokens.** It is *possible* (this is the alternative
we can't currently exclude) that `black`'s output happens to produce token patterns that
*ModernBERT's particular vocabulary* maps toward "machine," for reasons that are an accident of that
tokenizer rather than a fact about the human/machine data. All three siblings in Finding 7 share the
**exact same ModernBERT tokenizer**, so they cannot tell us whether this is happening — they'd all
inherit the same quirk.

**Why a different tokenizer is informative.** Train a model with a *different* tokenizer —
e.g. `roberta-base` (generic BPE) or `microsoft/codebert-base` (code BPE) — on the **original**
DroidCollection, then run the same `black` ablation:
- If it **also flips** under `black`: the effect cannot be a ModernBERT-tokenization quirk, because a
  different tokenizer reproduced it. The cause must be in the data. ✅ kills the alternative.
- If it **does not flip**: part of our headline really is a ModernBERT-specific tokenization story,
  and we'd have to re-scope the claim. (This would itself be an important, surprising finding.)

It also doubles as the "independent architecture" generality test: a non-DroidDetect model, our own
training loop, confirming the confound is not specific to their recipe.

> Note: RoBERTa is *not code-pretrained*; CodeBERT is. Running both is cheap and brackets the
> question "does this need a code-aware encoder?" I'd run CodeBERT as primary, RoBERTa as a bonus.

---

## How A and B fit together

| | trained on | tokenizer | answers |
|---|---|---|---|
| **A-control** | original data | ModernBERT | does our training loop reproduce DroidDetect? (sanity) |
| **A-treatment** | `black`-normalized data | ModernBERT | is the confound *removable* / how much is formatting? |
| **B** | original data | RoBERTa/CodeBERT (different) | is the flip data-driven or a ModernBERT-tokenizer artifact? |

A answers "is formatting the cause and can we fix it?"; B answers "is the cause the *data* (vs the
*recipe/tokenizer*)?" Together they discharge both surviving alternatives. If budget is tight, **A
is the more direct test of the headline; B is the more rigorous generality control.**

## Implementation notes (importing, not reimplementing)
- Use HuggingFace `Trainer` + `AutoModelForSequenceClassification` for B (standard, off-the-shelf).
  For A's same-architecture control, reuse the vendored `src/vendor/droiddetect_model.py` `TLModel`
  so we match their head exactly.
- Data: stream `project-droid/DroidCollection`, Python split, balanced classes; fixed seed; a held-out
  test split disjoint from train. Reuse the streaming/`black` harness already in
  `scripts/10_other_detectors.py` and `scripts/08_format_ablation.py`.
- `black` the corpus once and cache (formatting 100k+ files via `uvx black` is the slow step — batch it).
- Compute: ModernBERT-base / CodeBERT-base classifier, ~1 epoch on ~80k/class ≈ 1–3 GPU-hours each on
  GPU1. Whole plan fits comfortably in a single session.
- Add as `scripts/11_train_independent.py`, emit metrics to `results/`, and feed new macros through
  `scripts/06_build_macros.py` so any paper numbers stay script-sourced.

## Risks / failure modes
- **A-treatment low AUROC could be a bad training run, not a real drop** → the A-control gate prevents
  this misread.
- **`black` can't parse some scraped human files** → exclude unparseable items consistently across
  train and eval, and report the exclusion rate (same policy as the existing ablation).
- **Class/label polarity bugs** → assert on a labeled holdout before trusting any ablation number.
- **Train/test leakage** by problem or repo → split on the coarsest available grouping key.
