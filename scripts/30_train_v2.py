"""Model-agnostic CodeGPTSensor-recipe trainer.

Like scripts/18_train_cgs_amp.py but uses AutoModel/AutoConfig/AutoTokenizer so we
can swap between unixcoder-base-nine and ModernBERT-base (or any other encoder)
with the SAME recipe — CE-only, lr 2e-5, batch 8, seq 400, 20 ep cap, patience-5
ES, AMP-FP16, seed 99. This is the script for the 2×2 architecture × data
experiment.

DIFFs vs scripts/18_train_cgs_amp.py:
  - AutoModel/AutoConfig/AutoTokenizer (not RobertaModel-specific)
  - 2D attention_mask (standard format; both RoBERTa and ModernBERT accept)
  - `<encoder_only>` token only inserted if it's in the tokenizer's vocab
    (UniXcoder has it; ModernBERT doesn't)
  - Otherwise IDENTICAL: same JsonlMonitor, same EarlyStopping(patience=5) on val
    loss, same best-F1 checkpoint policy, same loss components

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/30_train_v2.py \\
      --do_train --amp --model_name_or_path microsoft/unixcoder-base-nine \\
      --train_data_file data/droid_py/train.jsonl \\
      --eval_data_file  data/droid_py/valid.jsonl \\
      --output_dir results/cgs/unixcoder_dc_ce \\
      --num_train_epochs 20 --block_size 400 \\
      --train_batch_size 8 --eval_batch_size 16 \\
      --learning_rate 2e-5 --max_grad_norm 1.0 --seed 99
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

import numpy as _np  # numpy 2 shim for cgs's EarlyStopping
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

from utils.early_stopping import EarlyStopping  # noqa: E402

from torch.utils.data import DataLoader, Dataset, SequentialSampler  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------- data -------------------------

ENCODER_ONLY_TOK = "<encoder_only>"  # UniXcoder-specific; tokenizer.tokenize() handles it


def tokenize_one(text: str, tok, block_size: int) -> list[int]:
    """Match CGS's `convert_examples_to_features` layout exactly:
        [CLS, <encoder_only>, SEP] + body + [SEP] + padding
    Note: `<encoder_only>` is NOT in UniXcoder's tokenizer vocab — CGS's
    `convert_tokens_to_ids` maps it to UNK (id=3). We preserve this quirk
    bit-for-bit for parity with the published recipe; for ModernBERT it also
    becomes that tokenizer's UNK token at position 1. Either way the prefix
    is a 3-token fixed pattern that's architecture-agnostic."""
    text = " ".join(text.split())
    body = tok.tokenize(text)[: block_size - 4]
    toks = [tok.cls_token, ENCODER_ONLY_TOK, tok.sep_token] + body + [tok.sep_token]
    ids = tok.convert_tokens_to_ids(toks)
    ids += [tok.pad_token_id] * (block_size - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, tok, block_size: int, file_path: str, max_rows: int | None = None):
        self.block_size = block_size
        with open(file_path) as f:
            rows = [json.loads(l) for l in f]
        if max_rows is not None:
            rows = rows[:max_rows]
        self.examples: list[dict] = []
        for r in rows:
            self.examples.append({
                "input_ids": tokenize_one(r["code"], tok, block_size),
                "contrast_ids": tokenize_one(r["contrast"], tok, block_size),
                "label": int(r["label"]),
                "index": int(str(r["index"])[-6:]),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return (
            torch.tensor(ex["input_ids"]),
            torch.tensor(ex["contrast_ids"]),
            torch.tensor(ex["label"]),
            torch.tensor(ex["index"]),
        )


# ------------------------- model -------------------------

class AMPModelV2(torch.nn.Module):
    """Encoder-agnostic head: masked mean pool of token embeddings → Linear(hidden→2).
    AMP-FP16 gated on self.training so eval is always FP32 (parity)."""

    def __init__(self, encoder, hidden_size: int, pad_token_id: int, args):
        super().__init__()
        self.encoder = encoder
        self.pad_token_id = pad_token_id
        self.args = args
        self.classifier = torch.nn.Linear(hidden_size, 2)

    def _pool(self, source_ids: torch.Tensor) -> torch.Tensor:
        mask = source_ids.ne(self.pad_token_id)  # (B, S) bool
        out = self.encoder(source_ids, attention_mask=mask.long())
        # Both RobertaModel and ModernBERT return last_hidden_state at index 0
        tok_emb = out[0]
        masked = tok_emb * mask.unsqueeze(-1)
        denom = mask.sum(-1).unsqueeze(-1).clamp(min=1)
        return masked.sum(1) / denom

    def _forward_core(self, input_ids, contrast_ids, labels):
        vec = self._pool(input_ids)
        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels)
        # CE-only — no contrastive auxiliary in this script
        return loss, prob

    def forward(self, input_ids, contrast_ids=None, labels=None):
        if self.args.amp and self.training:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self._forward_core(input_ids, contrast_ids, labels)
        return self._forward_core(input_ids, contrast_ids, labels)


# ------------------------- telemetry -------------------------

class JsonlMonitor:
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


# ------------------------- train + eval -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tok, monitor: JsonlMonitor):
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    args.max_steps = args.num_train_epochs * len(train_loader)
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.max_steps * 0.1),
        num_training_steps=args.max_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    early_stopping = EarlyStopping(patience=5)

    logger.info("Examples=%d Epochs=%d Batch=%d Steps=%d AMP=%s",
                len(train_dataset), args.num_train_epochs, args.train_batch_size,
                args.max_steps, args.amp)

    # Cache the eval dataset (re-tokenizing each epoch is wasteful)
    eval_dataset = TextDataset(tok, args.block_size, args.eval_data_file)

    best_f1 = 0.0
    losses: list[float] = []
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
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

        # per-epoch eval
        results, eval_loss = evaluate(args, model, tok, eval_dataset)
        monitor.log_eval(epoch, results, eval_loss)
        logger.info("valid epoch=%d acc=%.4f f1=%.4f auc=%.4f eval_loss=%.4f",
                    epoch, results["acc"], results["f1"], results["roc_auc"], eval_loss)
        if results["f1"] > best_f1:
            best_f1 = results["f1"]
            ckpt_dir = Path(args.output_dir) / "checkpoint-best-f1"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            target = model.module if hasattr(model, "module") else model
            torch.save(target.state_dict(), ckpt_dir / "model.bin")
            logger.info("new best F1=%.4f saved to %s", best_f1, ckpt_dir)
        early_stopping(eval_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping at epoch %d (patience=5 exhausted on val_loss)", epoch)
            break


def evaluate(args, model, tok, eval_dataset: TextDataset):
    loader = DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset),
        batch_size=args.eval_batch_size, num_workers=4, pin_memory=True,
    )
    eval_loss = 0.0
    n_steps = 0
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        inp = batch[0].to(args.device, non_blocking=True)
        con = batch[1].to(args.device, non_blocking=True)
        lab = batch[2].to(args.device, non_blocking=True)
        with torch.inference_mode():
            lm_loss, logit = model(inp, con, lab)
            eval_loss += lm_loss.mean().item()
            logits_all.append(logit.float().cpu().numpy())
            labels_all.append(lab.cpu().numpy())
        n_steps += 1
    logits = np.concatenate(logits_all, 0)
    labels = np.concatenate(labels_all, 0)
    preds = logits[:, 1] > 0.5
    res = {
        "acc": float(accuracy_score(labels, preds)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, logits[:, 1])),
    }
    return res, eval_loss / max(n_steps, 1)


# ------------------------- main -------------------------

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
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke_max_rows", type=int, default=0)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics.jsonl").write_text("")

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    logger.info("device=%s n_gpu=%d amp=%s seed=%d model=%s",
                device, args.n_gpu, args.amp, args.seed, args.model_name_or_path)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    # ModernBERT pads with -1; RoBERTa with 1. Use the tokenizer's actual pad id.
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise SystemExit(f"tokenizer {args.model_name_or_path} has no pad_token_id")
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise SystemExit(f"config for {args.model_name_or_path} has no hidden_size")
    model = AMPModelV2(encoder, hidden_size, pad_id, args).to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    logger.info("hidden_size=%d pad_id=%d has_<encoder_only>=%s",
                hidden_size, pad_id, ENCODER_ONLY_TOK in tok.get_vocab())
    monitor = JsonlMonitor(Path(args.output_dir) / "metrics.jsonl")

    if args.do_train:
        train_ds = TextDataset(tok, args.block_size, args.train_data_file,
                               max_rows=(args.smoke_max_rows or None))
        train(args, train_ds, model, tok, monitor)

    final = {"amp": args.amp, "smoke": bool(args.smoke_max_rows),
             "model": args.model_name_or_path}
    if args.do_train:
        eval_ds = TextDataset(tok, args.block_size, args.eval_data_file)
        final_results, final_loss = evaluate(args, model, tok, eval_ds)
        final.update({"final_val": final_results, "final_val_loss": final_loss})
    (Path(args.output_dir) / "final_val_metrics.json").write_text(json.dumps(final, indent=2))
    logger.info("DONE → %s/final_val_metrics.json", args.output_dir)


if __name__ == "__main__":
    main()
