# Reproducing this repo on a fresh machine

This repo was developed on a shared machine (`cs29824`) that Matthew is losing access to.
`README.md` documents the pipeline *shape* (`01` → `06`) but omits several things a fresh
machine actually needs. This file fills those gaps. Read `README.md` and `DESIGN.md` first
for the scientific context; this file is purely mechanical "how do I run this."

> None of the commands below were re-run to produce this file — it was written by auditing
> the code (scripts, `.gitignore`, `src/config.py`, `pyproject.toml`) and inspecting the
> live package environment on `cs29824`, not by executing the pipeline. Where the source
> was ambiguous or absent, it's marked `TODO (Matthew)` rather than guessed.

## 1. Environment

**Declared env is incomplete.** `pyproject.toml` only pins the Python floor:

```toml
requires-python = ">=3.11"
dependencies = []
```

There is no `uv.lock`, no `requirements.txt`, no `environment.yml` in the repo. Running
`uv sync` or `uv pip install -e .` installs **nothing** beyond stdlib — it will not get you
a working environment. `.python-version` pins `3.11` (the machine's plain `python3` is
3.10.12 — do not use it; the actual runtime used a Python-3.11 conda env).

The environment actually used was a **shared conda env named `dfc`** (see the "Environment
notes" section of `README.md`), not anything built from `pyproject.toml`. That env cannot be
handed over. Below are the package versions actually present in `dfc` on `cs29824`, captured
2026-07-02 via `pip list --format=freeze` — use these to reconstruct an equivalent env with
`uv` (`uv add <pkg>==<version>` for each, or a `uv pip install` line), since this is the
combination the code was actually run and tested against:

```
torch==2.7.1+cu128        torchvision==0.24.1        torchaudio==2.9.1
transformers==5.2.0       tokenizers==0.22.2         accelerate==1.12.0
datasets==4.5.0           huggingface_hub==1.4.1     sentence-transformers==5.2.2
sentencepiece==0.2.1      scikit-learn==1.8.0        scipy==1.17.0
pandas==3.0.0              pyarrow==23.0.0             numpy==1.26.4
matplotlib==3.10.8        wandb==0.24.2              openai==2.18.0
javalang==0.13.0          vllm==0.15.1 (installed but NOT usable, see below)
```

`black` was not present via `pip freeze` in `dfc` — the two scripts that need it
(`scripts/15_format_hmcorp.py`, `scripts/_archived/11b_make_variants.py`) are documented in
their own docstrings to be run through an **ephemeral uv environment**, not `dfc`:

```bash
uv run --no-project --with black --with javalang --with pandas python -u scripts/15_format_hmcorp.py
```

So this repo actually spans **two different environments**: the long-lived `dfc` conda env
(torch/transformers/GPU work) and one-off `uv run --no-project --with ...` invocations for
the CPU-only formatting scripts. TODO (Matthew): confirm exact `black` version used (not
recorded anywhere in the repo or in `dfc`'s freeze).

**GPU/CUDA:** `torch==2.7.1+cu128` requires a CUDA-12.8-compatible driver. `vllm==0.15.1` is
installed in `dfc` but multiple script docstrings (`scripts/02_generate_llm.py`,
`README.md`) state it is **unusable** on this machine: `vllm._C undefined symbol
_ZN3c104cuda9SetDeviceEab` (torch/vLLM ABI mismatch). All generation therefore goes through
plain HF `transformers` batched generation, not vLLM.

**torchvision ABI shim:** `src/_env.py` must be imported before any `transformers` import
(scripts already do this via `from src import _env`). It stubs `torchvision.transforms` if
the real import raises `torchvision::nms does not exist`, which happened with the specific
torch/torchvision pair on `cs29824`. Harmless no-op on a matched pair.

**Java toolchain:** `scripts/15_format_hmcorp.py` shells out to `java -jar
external/google-java-format.jar`, so a JRE must be on `PATH`. TODO (Matthew): record which
JRE version was used on `cs29824`.

## 2. Data (all gitignored — nothing under `data/` or `external/CodeGPTSensor/dataset/` is in git)

`.gitignore` excludes `data/*.parquet`, `data/repos/`, `data/hmcorp/`, `data/droid_py/`,
and all of `external/`. Every one of the following must be (re)built or downloaded on a
fresh machine. Real sources, verified against the code that produces each file:

| Output | Real source | Producing script |
|---|---|---|
| `data/human_code.parquet`, `data/problems.parquet` | HF dataset `MatrixStudio/Codeforces-Python-Submissions`, streamed (`datasets.load_dataset(..., streaming=True)`) | `scripts/01_build_human_data.py` |
| `data/gen/<model>.parquet` | Local HF `transformers` generation over the open-weight panel (`google/gemma-2-2b-it`, `Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct`, `meta-llama/Llama-3.1-8B-Instruct`) | `scripts/02_generate_llm.py --model <hf-id>`, driven by `./run_generation.sh` (all 4) / `./run_gen2.sh` (2 of them) |
| `data/gen/openai__gpt-5_4-nano.parquet` | OpenAI Responses API, `gpt-5.4-nano`, needs `OPENAI_API_KEY` | `scripts/02c_generate_openai.py --model gpt-5.4-nano` |
| `data/legendary_code.parquet` | Public git clones at pinned refs: `git/git` @ commit `e83c5163…` (Torvalds' initial commit, 2005), `redis/redis` @ last commit before 2012-06-01 (antirez), `sqlite/sqlite` @ last commit before 2013-06-01 (D.R. Hipp), `python/cpython` @ tag `v3.3.0`. Cloned into `data/repos/` (gitignored) then deleted. | `scripts/02b_legendary_extract.py` |
| `data/atcoder_human.parquet` | Scraped: submission stream from `kenkoooo.com` API + rating history + source from `atcoder.jp` (politeness delay ~1.2s) | `scripts/07_atcoder_human.py` |
| `data/droid_py_train.parquet`, `data/droid_py_test.parquet` | HF dataset `project-droid/DroidCollection` — `huggingface_hub.hf_hub_download(repo="project-droid/DroidCollection", repo_type="dataset")` on the 3 train shards + 1 test shard, filtered to `Language=="Python"` and label in `{HUMAN_GENERATED, MACHINE_GENERATED}` | `scripts/_archived/11_prepare_data.py` — **archived but still the real, load-bearing data-prep source**; see note below |
| `data/droid_py_train_bal.parquet`, `data/droid_py_train_bal_black.parquet`, `data/droid_py_test_black.parquet` (overwrites `droid_py_test.parquet` filtered) | Derived from the above: ast-parseable + black-formattable rows only, balanced 60k/class | `scripts/_archived/11b_make_variants.py` |
| `data/droid_py/{train,valid,test,test_formatted}.jsonl` | Converted from `droid_py_train_bal*`/`droid_py_test*` parquet into CodeGPTSensor's JSONL schema | `scripts/29_dc_prepare.py` |
| `external/CodeGPTSensor/dataset/{python,java}/{train,valid,test}.jsonl` (HMCorp) | Clone `https://github.com/doriscullen/CodeGPTSensor` into `external/CodeGPTSensor/` — the raw JSONL is that repo's own `dataset.zip`, already unzipped in the checkout used here. Upstream paper: Xu et al., *"Distinguishing LLM-generated from Human-written Code by Contrastive Learning,"* TOSEM 2025. | not produced by any script here — obtain via `git clone` |
| `data/hmcorp/{python,java}/{train,valid,test}[_formatted].jsonl` (~1.3 GB, ~30 min to rebuild) | Parse-filtered + `black`/`google-java-format`-formatted HMCorp | `scripts/15_format_hmcorp.py` (needs `external/google-java-format.jar`, see below) |

**Important nuance on the `11_*` scripts:** `scripts/_archived/README.md` says the `11_*`
family was "abandoned" — but that refers specifically to the *from-scratch ModernBERT
trainer* (`11_train.py`), which was pivoted away from in favor of CodeGPTSensor. The **data
prep** half of that family (`11_prepare_data.py`, `11b_make_variants.py`) is still the actual
source of `data/droid_py_train.parquet` etc., which `scripts/26_compliance_audit.py` and
`scripts/29_dc_prepare.py` both depend on today. Do not skip them just because the folder is
named `_archived`.

**`external/google-java-format.jar`** (3.5 MB, referenced by `scripts/15_format_hmcorp.py`
and `scripts/26_compliance_audit.py`) has no download script in this repo — the filename
matches the `google/google-java-format` project's release jars. TODO (Matthew): confirm the
exact release/version used and add a fetch step (currently must be placed at that path
manually).

**Fine-tuned checkpoints (this project's own trained detectors):** three CodeGPTSensor-recipe
(UniXcoder-base-nine, cross-entropy-only) fine-tunes, ~481 MB each:

```
results/cgs/java_raw_ce/checkpoint-best-f1/model.bin        # trained on HMCorp/java
results/cgs/python_raw_ce/checkpoint-best-f1/model.bin      # trained on HMCorp/python
results/cgs/unixcoder_dc_ce/checkpoint-best-f1/model.bin    # trained on DroidCollection-Python
```

These are gitignored (`.gitignore`: `results/cgs/*/checkpoint-best-f1/`) and are being
uploaded to the **public HuggingFace repo `AMindToThink/code-detection-confound-checkpoints`**.
Download the weights from there instead of retraining, unless you specifically want to
reproduce the training run (see §3). The small JSON/JSONL metrics/eval files alongside each
checkpoint (`metrics.jsonl`, `final_val_metrics.json`, `eval_phase_b.json`, `eval_q1.json`,
`eval_phase_d.json`) **are** tracked in git — only the `.bin` is excluded.

## 3. Running

Standard pipeline (skill-vs-species confound study, per `README.md`):

```bash
./run_pipeline.sh              # full pipeline, skips existing data artifacts
FORCE_GEN=1 ./run_pipeline.sh  # also (re)generate LLM code
./run_pipeline.sh analyze      # only re-run scoring -> analysis -> figures -> report
```
Hardcodes `CUDA_VISIBLE_DEVICES=1` (`README.md`: "GPU1 only" on the original 2-GPU machine —
adjust or unset on a different machine).

**Discrepancy found:** both `README.md` (line 32, `scripts/06_report.py`) and
`run_final_pipeline.sh` (which calls `python3 scripts/06_report.py`) reference a script that
**does not exist** in the repo — only `scripts/06_build_macros.py` is present, and it is what
`run_pipeline.sh` actually calls. `run_final_pipeline.sh` will fail at that line as committed.
TODO (Matthew): confirm whether `06_report.py` was renamed/merged into `06_build_macros.py`
and fix `README.md` / `run_final_pipeline.sh` accordingly, or restore the missing script.

CodeGPTSensor-recipe training (produces the 3 checkpoints above), faithful to Xu et al.'s
own `run.py` usage in `external/CodeGPTSensor/README.md` but AMP-enabled and with our
torchvision shim applied first:

```bash
# java_raw_ce / python_raw_ce (trained on HMCorp; swap python<->java, data/hmcorp/<lang>/*)
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/18_train_cgs_amp.py \
    --do_train --amp --model_name_or_path microsoft/unixcoder-base-nine \
    --train_data_file data/hmcorp/python/train.jsonl \
    --eval_data_file  data/hmcorp/python/valid.jsonl \
    --output_dir results/cgs/python_raw_ce \
    --num_train_epochs 20 --block_size 400 \
    --train_batch_size 8 --eval_batch_size 16 \
    --learning_rate 2e-5 --max_grad_norm 1.0 --seed 99 --contrast

# unixcoder_dc_ce (trained on DroidCollection-Python; CE-only, no --contrast)
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/30_train_v2.py \
    --do_train --amp --model_name_or_path microsoft/unixcoder-base-nine \
    --train_data_file data/droid_py/train.jsonl \
    --eval_data_file  data/droid_py/valid.jsonl \
    --output_dir results/cgs/unixcoder_dc_ce \
    --num_train_epochs 20 --block_size 400 \
    --train_batch_size 8 --eval_batch_size 16 \
    --learning_rate 2e-5 --max_grad_norm 1.0 --seed 99
```

Eval of a trained checkpoint against the 4-corpus battery (in-dist / formatted /
length-matched / length-matched+formatted):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/27_eval_phase_d.py \
    --output_dir results/cgs/python_raw_ce --eval_batch_size 16
```

Validate that the vendored `DroidDetect-Base-Binary` checkpoint loads faithfully (control,
should reproduce the published ~0.999 AUROC on the authors' own data):

```bash
python3 scripts/00_validate_droiddetect.py
```

Confound / ablation scripts referenced by the "Key results" in `FINDINGS.md` (each is a
standalone `python3 scripts/<name>.py` run reading the parquet/JSONL artifacts above and
writing to `results/*.json` or `results/phase_*/`): `08_format_ablation.py`,
`09_formatting_confound.py`, `10_other_detectors.py`, `15_format_hmcorp.py`,
`19_length_baseline.py`, `20_build_lenmatched.py`, `26_compliance_audit.py`,
`32_cross_cell_q1.py`. See each script's module docstring for its exact inputs/outputs and
pre-registered thresholds — they are detailed and not repeated here to avoid drift from the
source of truth.

Tests: `pytest` is present (`tests/test_detectors.py`, `tests/test_bib_pipeline.py`,
`tests/test_paper_macros.py`); no CI config or documented invocation was found — TODO
(Matthew): confirm `pytest` (bare) is sufficient, or whether specific fixtures/env need
`dfc` active first (likely yes, since `test_detectors.py` imports `src.vendor.*`, which
imports `torch`/`transformers`).

Paper: `paper/paper.tex` → `paper/paper.pdf`, built via `latexmk -pdf paper.tex` from the
`paper/` directory (see the last stage of `run_pipeline.sh`). Numbers are `\res...` macros
generated into `paper/macros.tex` by `scripts/06_build_macros.py` — never hand-typed.

## 4. External dependencies

- **Base HF models actually referenced in code:** `microsoft/unixcoder-base-nine` (CodeGPTSensor
  backbone), `Qwen/Qwen2.5-0.5B-Instruct` / `Qwen/Qwen2.5-1.5B-Instruct` (Binoculars
  observer/performer + Fast-DetectGPT scorer), `google/gemma-2-2b-it`,
  `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
  `microsoft/Phi-3.5-mini-instruct` (LLM generation panel; `Qwen/Qwen2.5-14B-Instruct`
  mentioned as optional in `src/config.py` but not run by default).
- **DroidDetect checkpoint (vendored, not ours):** HF repo
  `project-droid/DroidDetect-Base-Binary` (`src/config.py:DROIDDETECT_REPO`), loaded via
  `src/detectors.py` + `src/vendor/droiddetect_model.py`.
- **Fast-DetectGPT / Binoculars:** code (not weights) vendored verbatim in `src/vendor/`
  from `https://github.com/baoguangsheng/fast-detect-gpt` and
  `https://github.com/ahans30/Binoculars` respectively — see file headers for exact source
  paths and license (MIT / BSD-3-Clause).
- **This project's fine-tuned checkpoints:** public HF repo
  **`AMindToThink/code-detection-confound-checkpoints`** — the 3 `model.bin` files described
  in §2, once uploaded.
- **`gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct` are gated on HF** — the account
  used to generate must have accepted both licenses. No explicit `HF_TOKEN` handling was
  found in the code; it relies on `huggingface_hub`'s default auth (`huggingface-cli login`
  or `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env var).
- **wandb:** project `code-detection-confound`, entity
  `matthewkhoriaty-northwestern-university` (see `scripts/25_wandb_tailer.py` usage and the
  run links in `FINDINGS.md`). `scripts/25_wandb_tailer.py` is an optional sidecar that tails
  a training run's `metrics.jsonl` and logs to wandb — training itself
  (`18_train_cgs_amp.py`, `30_train_v2.py`) writes `metrics.jsonl` regardless of whether
  wandb is used.
- **Required env vars** (`.env` at repo root, gitignored): `OPENAI_API_KEY` (for
  `scripts/02c_generate_openai.py`), `WANDB_API_KEY` (for `scripts/25_wandb_tailer.py`; that
  script also accepts the legacy name `WANDB_KEY` and bridges it to `WANDB_API_KEY` if
  present). Neither is required to run the core detection/analysis pipeline (`01`-`06`),
  only the OpenAI-generation and wandb-logging side paths.

## 5. Hardware

- Developed on a machine with **2x Quadro RTX 8000 (46 GB each)**; the project was granted
  **GPU1 only** (`CUDA_VISIBLE_DEVICES=1` is hardcoded in `run_pipeline.sh`, `src/config.py`
  default, and `run_generation.sh`/`run_gen2.sh`). CodeGPTSensor training scripts
  (`17_run_cgs.py`, `18_train_cgs_amp.py`, `27_eval_phase_d.py`, `30_train_v2.py`) instead
  default to `CUDA_VISIBLE_DEVICES=0` in their own usage examples — pick whichever GPU is
  free on your machine.
- Disk: `README.md` notes only **~34 GB free** was available and instructs preferring cached
  HF models over new downloads. A fresh machine reproducing everything from scratch (LLM
  panel + DroidCollection + HMCorp + generated data) will need substantially more; no
  measured total was found in the repo — TODO (Matthew): estimate total disk footprint if
  you want to give future reproducers a number.
- No multi-GPU / distributed training code was found — everything is single-GPU.

---
*Written 2026-07-02 during the `cs29824` machine migration, auditing `README.md`,
`DESIGN.md`, `FINDINGS.md`, `pyproject.toml`, `.gitignore`, `scripts/`, `src/`, and
`external/CodeGPTSensor/` against the code as committed. Supersedes the earlier
uncommitted `REPRODUCIBILITY.md` draft in this checkout.*
