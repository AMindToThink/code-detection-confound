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

2. **Recipe / architecture explanation:** the confound is specific to *how DroidDetect is trained*
   (the ModernBERT recipe), not to the data, so a different architecture/training-loop on the same
   data would not inherit it.

Training our own models is the only way to separate these. Two experiments do it.

> **Retracted framing (kept for the record).** An earlier draft of this plan motivated Experiment B
> as ruling out a "tokenizer artifact" — the idea that `black`'s output produces token sequences
> that *ModernBERT's specific BPE vocabulary* accidentally maps to "machine." A fresh-context review
> (and Matthew) showed this framing is **confused**: the formatting signal lives in the *text*, and
> byte-level BPE tokenizers (RoBERTa/CodeBERT) preserve whitespace/newlines, so a different-tokenizer
> model trained on the same non-normalized data flips under `black` *identically* — both the data
> hypothesis and the "tokenizer" hypothesis predict the same result, so the experiment cannot
> separate them and there is no coherent tokenizer-only artifact to isolate. Experiment B is retained
> below purely as a **recipe/architecture generality** check, not a tokenizer test. The clean test of
> formatting-as-input is direct input-level normalization (which the `black`/`ast` ablation already is).

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

## Experiment B — independent architecture (recipe-generality check)

Train a detector with a *different architecture and training loop* — e.g. `microsoft/codebert-base`
(code-pretrained) or `roberta-base` (generic) via standard `AutoModelForSequenceClassification` — on
the **original** DroidCollection, then run the same `black` ablation.
- If it **also flips** under `black`: the confound is inherited by an independent recipe, so it is a
  property of the **data**, not of the DroidDetect recipe. (Strongly expected — see the retracted-framing
  note; this is the data hypothesis's prediction.)
- If it **does not flip**: the confound is somehow specific to the DroidDetect recipe — a surprising
  result worth chasing.

This is a generality control, **not** a tokenizer test. Because both hypotheses predict a flip, B's
main value is confirming the confound is recipe-agnostic; it is strictly secondary to Experiment A,
which is the direct, decisive test. RoBERTa is not code-pretrained and CodeBERT is, so running both
also brackets "does this need a code-aware encoder?"

---

## How A and B fit together

| | trained on | architecture | answers |
|---|---|---|---|
| **A-control** | original data | ModernBERT (DroidDetect head) | does our training loop reproduce DroidDetect? (sanity gate) |
| **A-treatment** | `black`-normalized data | ModernBERT (DroidDetect head) | is the confound *removable* / how much of the score is formatting? |
| **B** | original data | CodeBERT/RoBERTa (independent) | does an independent recipe inherit the confound? (data vs recipe) |

A answers "is formatting the cause and can we fix it?"; B answers "is the cause the *data* (vs the
DroidDetect *recipe*)?" If budget is tight, **A is the direct, decisive test of the headline; B is a
secondary generality control whose outcome the data hypothesis already predicts.**

## Compute / time estimate (this machine)
Measured: ModernBERT-base fine-tuning on **GPU1 (Quadro RTX 8000, 46 GB)** runs at **~55 samples/sec**
at seq-len 512, fp16 (batch 32 uses 12.5 GB; the recipe's batch 64 fits). DroidDetect's recipe is
**3 epochs**; the DroidCollection Python *train* subset is **not published** but is ≈130k samples
(~167k Python rows × 0.8 train split).

| Python train size | one 3-epoch run | Experiment A (control + treatment) |
|---|---|---|
| ~90k | ~1.4 h | ~2.7 h + preprocessing |
| **~130k (best guess)** | **~2.0 h** | **~3.9 h + preprocessing** |
| ~170k | ~2.6 h | ~5.2 h + preprocessing |

So **~2 hours per full 3-epoch run**, and **~4 hours of GPU time** for the complete Experiment A
(both runs), plus a one-time `black` pass over the corpus (~15–30 min) and minutes for eval. Adding
Experiment B is one more ~2 h run. The whole program fits in a single session. Caveats: the exact
Python count and DroidDetect's training `max_len`/precision are unpublished (pin them from the
project's GitHub training scripts for a precise figure); the contrastive triplet term adds only
minor per-step overhead; fp16 assumed (fp32 on Turing would be ~2× slower, but our reproduction would
use AMP).

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
