"""Generic Q1 eval driver for scripts/30_train_v2.py checkpoints.

Takes a (model_name_or_path, checkpoint_dir, raw_path, formatted_path) and applies
the pre-registered Q1 thresholds:
  - test vs test_formatted: machine-prob drop ≥ 0.10 OR flag-rate drop ≥ 10 pp → Q1=yes
  Both fail → Q1=no.

Writes results/cgs/<checkpoint_name>/eval_q1.json keyed by corpus, with per-row p1
arrays for downstream paired analysis.

Usage:
  python3 -u scripts/31_eval_q1_v2.py \\
      --model_name_or_path answerdotai/ModernBERT-base \\
      --output_dir results/cgs/modernbert_hmcorp_ce \\
      --raw_path  data/hmcorp/python/test.jsonl \\
      --formatted_path data/hmcorp/python/test_formatted.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

import numpy as _np  # numpy 2 shim
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

# Reuse model + dataset from 30_train_v2.py via importlib (no source duplication)
_spec = importlib.util.spec_from_file_location("train_v2", ROOT / "scripts" / "30_train_v2.py")
train_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_v2)
AMPModelV2 = train_v2.AMPModelV2
TextDataset = train_v2.TextDataset

BLOCK = 400
N_BOOT = 500
SEED = 0


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(y)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    arr = np.asarray(aucs)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def eval_corpus(model, tok, args, path: Path) -> dict:
    ds = TextDataset(tok, BLOCK, str(path))
    dl = DataLoader(ds, sampler=SequentialSampler(ds),
                    batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    model.eval()
    all_idx, all_lab, all_p1 = [], [], []
    with torch.inference_mode():
        for batch in dl:
            inp = batch[0].to(args.device, non_blocking=True)
            con = batch[1].to(args.device, non_blocking=True)
            lab = batch[2].to(args.device, non_blocking=True)
            idx = batch[3]
            _, logit = model(inp, con, lab)
            prob = F.softmax(logit.float(), dim=-1)
            all_p1.append(prob[:, 1].cpu().numpy())
            all_lab.append(lab.cpu().numpy())
            all_idx.append(idx.numpy())
    p1 = np.concatenate(all_p1); y = np.concatenate(all_lab); ids = np.concatenate(all_idx)
    preds = p1 > 0.5
    out = {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y, p1)),
        "machine_prob_mean_on_machine_rows": float(p1[y == 1].mean() if (y == 1).any() else float("nan")),
        "machine_prob_mean_on_human_rows":   float(p1[y == 0].mean() if (y == 0).any() else float("nan")),
        "flag_rate_machine": float((preds[y == 1]).mean() if (y == 1).any() else float("nan")),
        "flag_rate_human":   float((preds[y == 0]).mean() if (y == 0).any() else float("nan")),
        "_idx": ids.tolist(),
        "_y": y.tolist(),
        "_p1": p1.tolist(),
    }
    lo, hi = bootstrap_auc(y, p1)
    out["auroc_ci"] = [lo, hi]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--raw_path", required=True)
    ap.add_argument("--formatted_path", required=True)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--label", help="optional label for the report (e.g. 'modernbert_hmcorp')")
    a = ap.parse_args()

    args = argparse.Namespace(
        eval_batch_size=a.eval_batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amp=False, do_train=False,
    )

    print(f"Loading {a.model_name_or_path} + {a.output_dir}/checkpoint-best-f1/model.bin",
          flush=True)
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path)
    cfg = AutoConfig.from_pretrained(a.model_name_or_path)
    enc = AutoModel.from_pretrained(a.model_name_or_path)
    model = AMPModelV2(enc, cfg.hidden_size, tok.pad_token_id, args).to(args.device)
    ckpt = Path(a.output_dir) / "checkpoint-best-f1" / "model.bin"
    model.load_state_dict(torch.load(ckpt, map_location=args.device))

    results: dict = {
        "checkpoint": str(ckpt),
        "model": a.model_name_or_path,
        "label": a.label or Path(a.output_dir).name,
        "per_corpus": {},
    }
    for name, path in [("test", Path(a.raw_path)),
                       ("test_formatted", Path(a.formatted_path))]:
        print(f"\n[{name}] {path}", flush=True)
        r = eval_corpus(model, tok, args, path)
        print(f"  n={r['n']} acc={r['acc']:.4f} f1={r['f1']:.4f} "
              f"AUROC={r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]",
              flush=True)
        print(f"  mean_p1: human={r['machine_prob_mean_on_human_rows']:.4f} "
              f"machine={r['machine_prob_mean_on_machine_rows']:.4f}  "
              f"TPR={r['flag_rate_machine']:.4f}  FPR={r['flag_rate_human']:.4f}", flush=True)
        results["per_corpus"][name] = r

    rt = results["per_corpus"]["test"]
    rf = results["per_corpus"]["test_formatted"]
    prob_drop = rt["machine_prob_mean_on_machine_rows"] - rf["machine_prob_mean_on_machine_rows"]
    flag_drop = rt["flag_rate_machine"] - rf["flag_rate_machine"]
    results["q1_metrics"] = {
        "machine_prob_drop_test_vs_formatted": prob_drop,
        "flag_rate_drop_test_vs_formatted":    flag_drop,
    }
    q1 = "yes" if (prob_drop >= 0.10 or flag_drop >= 0.10) else "no"
    results["q1_decision"] = q1

    print(f"\n=== Q1 pre-registered decision ===", flush=True)
    print(f"  machine-prob drop test→formatted = {prob_drop:+.4f}  (threshold ≥ 0.10)",
          flush=True)
    print(f"  flag-rate    drop test→formatted = {flag_drop:+.4f}  (threshold ≥ 0.10)",
          flush=True)
    print(f"  Q1 = {q1.upper()}", flush=True)

    out_p = Path(a.output_dir) / "eval_q1.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
