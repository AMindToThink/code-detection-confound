"""CodeGPTSensor training with AMP-FP16, faithful to the published recipe.

This script is a near-verbatim copy of `external/CodeGPTSensor/CodeGPTSensor/run.py`
(Xu et al., TOSEM 2025, github.com/doriscullen/CodeGPTSensor). The DIFF vs their code is
small and contained here so `external/CodeGPTSensor/` stays unmodified:

  1. `Model.forward` is replaced by an AMP-safe forward that wraps the encoder pass in
     `torch.amp.autocast('cuda', dtype=torch.float16)`. KL_div and cosine_embedding
     inputs are cast back to FP32 inside autocast to avoid R-drop / cosine underflow
     under FP16 — a known FP16 hazard for these two loss components. Cross-entropy is
     safe under autocast (HF autocasts softmax/log_softmax internally).
  2. The training loop uses `torch.amp.GradScaler` around `loss.backward()` and
     `optimizer.step()` so gradients stay representable in FP16.
  3. Per-step JSONL telemetry is appended to `<output_dir>/metrics.jsonl` (loss, lr,
     grad_norm, samples/s) and per-epoch eval rows. The completion sentinel
     `<output_dir>/final_val_metrics.json` is written at end so the watchdog can detect
     a healthy finish.
  4. Tokenized eval dataset is cached across epochs — their `evaluate()` re-tokenizes
     the full valid set every call, which is wasteful at 20-epoch scale.

EVERYTHING ELSE is bit-identical to their `run.py`: batch=8 default, lr=2e-5 default,
block_size=400 default, EarlyStopping(patience=5) on val_loss, F1-best checkpointing,
SequentialSampler, contrastive loss weights (0.1 KL + 0.2 cosine), seed handling.

The `--amp` flag toggles AMP; when off the script behaves identically to their `run.py`
(useful for the AMP-vs-FP32 parity check).

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/18_train_cgs_amp.py \\
      --do_train --amp --model_name_or_path microsoft/unixcoder-base-nine \\
      --train_data_file data/hmcorp/python/train.jsonl \\
      --eval_data_file  data/hmcorp/python/valid.jsonl \\
      --output_dir results/cgs/python_raw \\
      --num_train_epochs 20 --block_size 400 \\
      --train_batch_size 8 --eval_batch_size 16 \\
      --learning_rate 2e-5 --max_grad_norm 1.0 --seed 99 --contrast
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── one-time env shim + path setup (must run before transformers import) ─────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401  torchvision shim

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

# numpy>=2 removed `np.Inf` (their early_stopping.py uses it). Restore the alias here
# rather than modifying their file. Pure compat shim, no semantic change.
import numpy as _np  # noqa: E402
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf

# Their unmodified utilities:
import model as cgs_model  # noqa: E402  for `get_kl_loss`
from utils.early_stopping import EarlyStopping  # noqa: E402

from torch.utils.data import DataLoader, Dataset, SequentialSampler  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from transformers import (  # noqa: E402
    RobertaConfig, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ────────────────────────────── data ──────────────────────────────────────────────────
# Verbatim from their run.py (lines 49-118), no changes.

class InputFeatures:
    def __init__(self, input_tokens, input_ids, contrast_tokens, contrast_ids, label, index):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.contrast_tokens = contrast_tokens
        self.contrast_ids = contrast_ids
        self.label = label
        self.index = index


def convert_examples_to_features(js, tokenizer, args):
    code = " ".join(js["code"].split())
    code_tokens = tokenizer.tokenize(code)[: args.block_size - 4]
    source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_ids += [tokenizer.pad_token_id] * (args.block_size - len(source_ids))

    contrast = " ".join(js["contrast"].split())
    contrast_tokens = tokenizer.tokenize(contrast)[: args.block_size - 4]
    contrast_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + contrast_tokens + [tokenizer.sep_token]
    contrast_ids = tokenizer.convert_tokens_to_ids(contrast_tokens)
    contrast_ids += [tokenizer.pad_token_id] * (args.block_size - len(contrast_ids))

    # Their `js['index'] = int(js["index"][-6:])` — preserved (index format e.g. "gp223799").
    js["index"] = int(js["index"][-6:])

    return InputFeatures(source_tokens, source_ids, contrast_tokens, contrast_ids,
                         js["label"], js["index"])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path: str, max_rows: int | None = None):
        self.examples = []
        with open(file_path) as f:
            data = [json.loads(line) for line in f]
        if max_rows is not None:
            data = data[:max_rows]
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return (
            torch.tensor(ex.input_ids),
            torch.tensor(ex.contrast_ids),
            torch.tensor(ex.label),
            torch.tensor(ex.index),
        )


# ────────────────────────────── model: AMP-safe forward ───────────────────────────────
# Identical math to cgs_model.Model.forward — the ONLY changes are:
#   - encoder/classifier wrapped in autocast(FP16)
#   - KL_div and cosine_embedding inputs explicitly .float() to stay in FP32

class AMPModel(torch.nn.Module):
    """Drop-in for their Model, with AMP-safe loss math when --amp is on."""

    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = torch.nn.Linear(config.hidden_size, 2)

    def get_xcode_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.encoder(
            source_ids,
            attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2),
            output_hidden_states=True,
        )
        tok_emb = out[0]
        return (tok_emb * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)

    def _forward_core(self, input_ids, contrast_ids, labels):
        vec = self.get_xcode_vec(input_ids)
        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)
        ce = F.cross_entropy(logits, labels)
        if self.args.contrast and self.args.do_train:
            vec2 = self.get_xcode_vec(input_ids)
            logits2 = self.classifier(vec2)
            kl = cgs_model.get_kl_loss(logits.float(), logits2.float())
            vec3 = self.get_xcode_vec(contrast_ids)
            label_new = torch.full(labels.size(), -1, device=self.args.device)
            cos = F.cosine_embedding_loss(vec.float(), vec3.float(), label_new)
            loss = ce + 0.1 * kl + 0.2 * cos
        else:
            loss = ce
        return loss, prob

    def forward(self, input_ids, contrast_ids=None, labels=None):
        # AMP is on only during training, never during eval. `model.eval()` clears
        # self.training to False, and the per-paper Table-6 F1 numbers depend on
        # logits crossing a 0.5 threshold — FP16 noise can flip borderline rows, so
        # eval stays FP32 for parity even when training is FP16.
        if self.args.amp and self.training:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self._forward_core(input_ids, contrast_ids, labels)
        return self._forward_core(input_ids, contrast_ids, labels)


# ────────────────────────────── telemetry ─────────────────────────────────────────────

class JsonlMonitor:
    """Per-skill `supervising-training-runs` pattern: per-step + per-eval JSONL telemetry."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_step = 0
        self.cum_step = 0

    def log_step(self, loss: float, lr: float, grad_norm: float | None, batch_size: int):
        self.cum_step += 1
        if self.cum_step % 10 != 0:
            return
        now = time.time()
        dstep = self.cum_step - self.last_step
        sps = (dstep * batch_size) / max(now - self.last_t, 1e-6)
        self.last_t = now
        self.last_step = self.cum_step
        if not np.isfinite(loss):
            self._append({"FATAL": "non-finite loss", "step": self.cum_step, "loss": str(loss)})
            raise RuntimeError(f"non-finite loss {loss} at step {self.cum_step}")
        rec = {
            "step": self.cum_step,
            "t": round(now - self.t0, 1),
            "loss": round(loss, 4),
            "lr": round(lr, 8),
            "grad_norm": (round(grad_norm, 3) if grad_norm is not None else None),
            "sps": round(sps, 2),
        }
        self._append(rec)

    def log_eval(self, epoch: int, results: dict, val_loss: float):
        rec = {
            "epoch": epoch,
            "step": self.cum_step,
            "t": round(time.time() - self.t0, 1),
            "eval_loss": round(val_loss, 4),
            "eval_acc": round(results["acc"], 4),
            "eval_f1": round(results["f1"], 4),
            "eval_auroc": round(results["roc_auc"], 4),
        }
        self._append(rec)

    def _append(self, rec: dict):
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


# ────────────────────────────── train / eval ─────────────────────────────────────────
# Their train()/evaluate() with three small additions: GradScaler, telemetry, eval cache.

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer, monitor: JsonlMonitor):
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    args.max_steps = args.num_train_epochs * len(train_dataloader)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.max_steps * 0.1), num_training_steps=args.max_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    early_stopping = EarlyStopping(patience=5)

    logger.info("***** Running training *****")
    logger.info("  Num examples=%d  Num Epochs=%d  Batch=%d  Total steps=%d  AMP=%s",
                len(train_dataset), args.num_train_epochs, args.train_batch_size,
                args.max_steps, args.amp)

    # Cache the eval dataset (single tokenization across all epochs)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    best_f1 = 0.0
    losses: list[float] = []
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device, non_blocking=True)
            contrasts = batch[1].to(args.device, non_blocking=True)
            labels = batch[2].to(args.device, non_blocking=True)

            model.train()
            loss, _ = model(inputs, contrasts, labels)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item())
            monitor.log_step(
                loss=loss.item(), lr=scheduler.get_last_lr()[0],
                grad_norm=float(grad_norm) if grad_norm is not None else None,
                batch_size=args.train_batch_size,
            )

            if (step + 1) % 100 == 0:
                logger.info("epoch %d step %d loss %.4f scale=%.0f",
                            epoch, step + 1, float(np.mean(losses[-100:])),
                            float(scaler.get_scale()) if args.amp else 1.0)

        # ── per-epoch eval ─────
        results, eval_loss = evaluate(args, model, tokenizer, eval_dataset)
        monitor.log_eval(epoch, results, eval_loss)
        logger.info("***** valid epoch=%d ***** acc=%.4f f1=%.4f auc=%.4f eval_loss=%.4f",
                    epoch, results["acc"], results["f1"], results["roc_auc"], eval_loss)

        if results["f1"] > best_f1:
            best_f1 = results["f1"]
            ckpt_dir = Path(args.output_dir) / "checkpoint-best-f1"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), ckpt_dir / "model.bin")
            logger.info("  *** new best F1 %.4f saved to %s ***", best_f1, ckpt_dir)

        early_stopping(eval_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping at epoch %d (patience=5 exhausted on val_loss)", epoch)
            break


def evaluate(args, model, tokenizer, eval_dataset: TextDataset | None = None):
    if eval_dataset is None:
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset),
        batch_size=args.eval_batch_size, num_workers=4, pin_memory=True,
    )
    eval_loss = 0.0
    n_eval_steps = 0
    model.eval()
    logits_all, labels_all = [], []
    for batch in eval_dataloader:
        inp = batch[0].to(args.device, non_blocking=True)
        con = batch[1].to(args.device, non_blocking=True)
        lab = batch[2].to(args.device, non_blocking=True)
        with torch.inference_mode():
            # AMPModel.forward gates autocast on self.training; model.eval() above
            # forces FP32 here regardless of args.amp. See C2 fix.
            lm_loss, logit = model(inp, con, lab)
            eval_loss += lm_loss.mean().item()
            logits_all.append(logit.float().cpu().numpy())
            labels_all.append(lab.cpu().numpy())
        n_eval_steps += 1
    logits = np.concatenate(logits_all, 0)
    labels = np.concatenate(labels_all, 0)
    preds = logits[:, 1] > 0.5
    results = {
        "acc": float(accuracy_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, logits[:, 1])),
    }
    return results, eval_loss / max(n_eval_steps, 1)


# ────────────────────────────── CLI ──────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_data_file")
    p.add_argument("--eval_data_file")
    p.add_argument("--test_data_file")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--block_size", type=int, default=400)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_test", action="store_true")
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--amp", action="store_true", help="enable FP16 autocast + GradScaler")
    p.add_argument("--smoke_max_rows", type=int, default=0,
                   help="if >0, train on the first N rows only (smoke test)")
    args = p.parse_args()

    # First thing: truncate metrics.jsonl so the watchdog doesn't trip on a stale file.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics.jsonl").write_text("")

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    logger.info("device=%s n_gpu=%d amp=%s seed=%d", device, args.n_gpu, args.amp, args.seed)
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    model = AMPModel(encoder, config, tokenizer, args).to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    logger.info("args: %s", vars(args))

    monitor = JsonlMonitor(Path(args.output_dir) / "metrics.jsonl")

    if args.do_train:
        train_ds = TextDataset(
            tokenizer, args, args.train_data_file,
            max_rows=(args.smoke_max_rows or None),
        )
        train(args, train_ds, model, tokenizer, monitor)

    if args.do_test:
        # C1 fix: their published run.py evaluates on `args.test_data_file` for
        # --do_test; the earlier version of this file silently fell back to
        # args.eval_data_file (= valid) via the cached eval dataset. Build the test
        # dataset explicitly and pass it through.
        if not args.test_data_file:
            raise SystemExit("--do_test requires --test_data_file")
        ckpt = Path(args.output_dir) / "checkpoint-best-f1" / "model.bin"
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(torch.load(ckpt, map_location=device))
        test_ds = TextDataset(tokenizer, args, args.test_data_file)
        results, _ = evaluate(args, model, tokenizer, eval_dataset=test_ds)
        (Path(args.output_dir) / "test_results.json").write_text(json.dumps(results, indent=2))
        logger.info("***** TEST ***** %s", json.dumps(results, indent=2))

    # Completion sentinel for the watchdog
    final = {"amp": args.amp, "smoke": bool(args.smoke_max_rows)}
    if args.do_train:
        final_results, final_loss = evaluate(args, model, tokenizer)
        final.update({"final_val": final_results, "final_val_loss": final_loss})
    (Path(args.output_dir) / "final_val_metrics.json").write_text(json.dumps(final, indent=2))
    logger.info("DONE → %s/final_val_metrics.json", args.output_dir)


if __name__ == "__main__":
    main()
