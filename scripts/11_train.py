"""Experiment A: fine-tune ModernBERT-base for human-vs-machine code classification.

Trains on either the ORIGINAL or the BLACK-normalized DroidCollection-Python balanced
subset, following DroidDetect's recipe as closely as a standard HF setup allows
(3 epochs, batch 64, AdamW lr 5e-5, cosine schedule, 0.1 warmup, fp16). We use the
canonical `AutoModelForSequenceClassification` head + cross-entropy (we DROP DroidDetect's
auxiliary contrastive-triplet term and custom projection head; neither bears on the
formatting question, and the A-control reproducing high AUROC validates the setup).

Monitoring: appends one JSON line per logging step to <out>/metrics.jsonl (loss, lr,
grad_norm, samples/s) and per eval (val AUROC/acc). Fail-loud: aborts on non-finite loss.
Checkpoints best (by val AUROC) + last.

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u scripts/11_train.py --variant original --out results/train/original
  CUDA_VISIBLE_DEVICES=1 python -u scripts/11_train.py --variant black    --out results/train/black
  add --smoke for a fast preflight (tiny subset, ~150 steps, frequent eval).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401  torchvision shim so transformers imports
from src import config as C

import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          TrainerCallback)
from datasets import Dataset

BASE = "answerdotai/ModernBERT-base"
MAX_LEN = 512
VAL_N = 3000  # small held-out slice of test for periodic monitoring


class JsonlMonitor(TrainerCallback):
    """Append compact telemetry to metrics.jsonl; abort loudly on non-finite loss."""
    def __init__(self, path: Path):
        self.path = path; self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(""); self._t = time.time(); self._last_step = 0
        self._lt = self._t

    def _write(self, d: dict):
        with self.path.open("a") as f:
            f.write(json.dumps(d) + "\n"); f.flush()

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        rec = {"step": state.global_step, "t": round(time.time() - self._t, 1)}
        if "loss" in logs:
            loss = logs["loss"]
            if not np.isfinite(loss):
                self._write({**rec, "FATAL": "non-finite loss", "loss": str(loss)})
                raise RuntimeError(f"non-finite loss {loss} at step {state.global_step}")
            now = time.time(); dstep = state.global_step - self._last_step
            sps = (dstep * args.per_device_train_batch_size) / max(now - getattr(self, "_lt", now), 1e-6)
            self._lt = now; self._last_step = state.global_step
            rec.update(loss=round(loss, 4), lr=logs.get("learning_rate"),
                       grad_norm=round(logs.get("grad_norm", float("nan")), 3),
                       sps=round(sps, 1))
            self._write(rec)
        if "eval_auroc" in logs:
            self._write({**rec, "eval_auroc": round(logs["eval_auroc"], 4),
                         "eval_acc": round(logs.get("eval_accuracy", float("nan")), 4)})


def compute_metrics(p):
    logits, labels = p
    prob1 = torch.softmax(torch.tensor(logits).float(), dim=-1)[:, 1].numpy()
    return {"auroc": roc_auc_score(labels, prob1),
            "accuracy": accuracy_score(labels, prob1 > 0.5)}


def make_ds(df: pd.DataFrame, tok) -> Dataset:
    ds = Dataset.from_pandas(df[["code", "label"]], preserve_index=False)
    return ds.map(lambda b: tok(b["code"], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=["code"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "black"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    train_pq = ("droid_py_train_bal.parquet" if args.variant == "original"
                else "droid_py_train_bal_black.parquet")
    test_pq = "droid_py_test.parquet" if args.variant == "original" else "droid_py_test_black.parquet"
    train_df = pd.read_parquet(C.DATA / train_pq)
    test_df = pd.read_parquet(C.DATA / test_pq)

    # label-polarity / data audit (fail loud BEFORE the long run)
    assert set(train_df.label.unique()) == {0, 1}, train_df.label.unique()
    print(f"[{args.variant}] train={len(train_df)} "
          f"(h={int((train_df.label==0).sum())}/m={int((train_df.label==1).sum())}) "
          f"test={len(test_df)} | label map 0=human,1=machine", flush=True)

    rng = np.random.default_rng(0)
    val_idx = rng.permutation(len(test_df))[:VAL_N]
    val_df = test_df.iloc[val_idx].reset_index(drop=True)
    if args.smoke:
        train_df = train_df.sample(2000, random_state=0).reset_index(drop=True)
        val_df = val_df.sample(800, random_state=0).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, id2label={0: "human", 1: "machine"},
        label2id={"human": 0, "machine": 1})
    train_ds, val_ds = make_ds(train_df, tok), make_ds(val_df, tok)

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=1 if args.smoke else 3,
        per_device_train_batch_size=64, per_device_eval_batch_size=128,
        learning_rate=5e-5, warmup_ratio=0.1, lr_scheduler_type="cosine",
        weight_decay=0.01, fp16=True, max_grad_norm=1.0,
        eval_strategy="steps", eval_steps=20 if args.smoke else 250,
        save_strategy="steps", save_steps=20 if args.smoke else 250, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="auroc", greater_is_better=True,
        logging_strategy="steps", logging_steps=5 if args.smoke else 10,
        # wandb is the human's dashboard (set WANDB_API_KEY in .env, WANDB_PROJECT env);
        # JSONL telemetry remains as the script-readable backup. TensorBoard is optional —
        # add "tensorboard" to this list if `pip install tensorboard` is in the env.
        report_to=["wandb"],
        run_name=f"{args.variant}_modernbert" + ("_smoke" if args.smoke else ""),
        seed=0, dataloader_num_workers=4, disable_tqdm=True)

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                      callbacks=[JsonlMonitor(out / "metrics.jsonl")])
    trainer.train()
    metrics = trainer.evaluate()
    (out / "final_val_metrics.json").write_text(json.dumps(metrics, indent=2))
    trainer.save_model(str(out / "best"))
    tok.save_pretrained(str(out / "best"))
    print(f"[{args.variant}] DONE. final val: {metrics}", flush=True)


if __name__ == "__main__":
    main()
