"""Phase B eval: load a trained CodeGPTSensor checkpoint, evaluate on the 4-corpus
battery for one language, compute the pre-registered Q1 metric.

Battery (per language):
  (a) HMCorp test.jsonl              — in-distribution reproduction target
  (b) HMCorp test_formatted.jsonl    — Q1 formatting-flip
  (c) test_lenmatched.jsonl          — length-controlled diagnostic
  (d) test_lenmatched_formatted.jsonl — length-and-formatting joint control

Pre-registered Q1 positive criterion (revised plan 2026-05-24):
  On (a) vs (b), comparing same rows by index:
    (i) machine-row mean predicted-machine-probability drops by ≥ 0.10 (absolute), OR
    (ii) per-row binary-flag rate on machine class drops by ≥ 10 percentage points.
  Either fires → Q1 = yes → Phase C launches. Both fail → Q1 = no → paper pivots.

Outputs:
  results/cgs/{lang}_raw_ce/eval_phase_b.json
    {
      "checkpoint": "...",
      "per_corpus": {
        "test":                 {"n": ..., "acc": ..., "f1": ..., "auroc": ..., "auroc_ci": [lo, hi], "machine_prob_mean": ..., "flag_rate_machine": ...},
        "test_formatted":       {...},
        "test_lenmatched":      {...},
        "test_lenmatched_formatted": {...},
      },
      "q1_metrics": {
        "machine_prob_drop_test_vs_formatted":    <float>,
        "flag_rate_drop_test_vs_formatted":       <float>,
        "machine_prob_drop_lenmatched_vs_lenmatched_formatted": <float>,
        "flag_rate_drop_lenmatched_vs_lenmatched_formatted":    <float>,
      },
      "q1_decision": "yes"|"no"|"ambiguous",
      "length_tiebreaker": "use formatting"|"length explains"|"ambiguous"|"n/a",
    }

Usage:
  python3 -u scripts/23_eval_phase_b.py --lang python --output_dir results/cgs/python_raw_ce
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

# numpy 2 shim for their EarlyStopping import path
import numpy as _np  # noqa: E402
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, SequentialSampler  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer  # noqa: E402

# Reuse the AMP-safe model + TextDataset from the training script
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("train_mod", ROOT / "scripts" / "18_train_cgs_amp.py")
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)
AMPModel = train_mod.AMPModel
TextDataset = train_mod.TextDataset

BASE = "microsoft/unixcoder-base-nine"
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
    ds = TextDataset(tok, args, str(path))
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=args.eval_batch_size,
                    num_workers=4, pin_memory=True)
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
    p1 = np.concatenate(all_p1)
    y = np.concatenate(all_lab)
    idx_arr = np.concatenate(all_idx)
    preds = p1 > 0.5
    auroc = float(roc_auc_score(y, p1))
    lo, hi = bootstrap_auc(y, p1)
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "auroc": auroc,
        "auroc_ci": [lo, hi],
        "machine_prob_mean": float(p1.mean()),
        "machine_prob_mean_on_machine_rows": float(p1[y == 1].mean()),
        "machine_prob_mean_on_human_rows":   float(p1[y == 0].mean()),
        "flag_rate_machine": float((preds[y == 1]).mean()),  # TPR
        "flag_rate_human":   float((preds[y == 0]).mean()),  # FPR
        # Per-row scores for downstream paired analysis
        "_idx": idx_arr.tolist(),
        "_y": y.tolist(),
        "_p1": p1.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["python", "java"], required=True)
    ap.add_argument("--output_dir", required=True, help="trained checkpoint dir")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    a = ap.parse_args()

    args = argparse.Namespace(
        block_size=BLOCK, eval_batch_size=a.eval_batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amp=False, do_train=False, contrast=False,
    )

    print(f"Loading {BASE} + checkpoint from {a.output_dir}/checkpoint-best-f1/model.bin", flush=True)
    tok = RobertaTokenizer.from_pretrained(BASE)
    config = RobertaConfig.from_pretrained(BASE)
    encoder = RobertaModel.from_pretrained(BASE)
    model = AMPModel(encoder, config, tok, args).to(args.device)
    ckpt_p = Path(a.output_dir) / "checkpoint-best-f1" / "model.bin"
    if not ckpt_p.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_p}")
    model.load_state_dict(torch.load(ckpt_p, map_location=args.device))

    data = ROOT / "data" / "hmcorp" / a.lang
    corpora = {
        "test":                       data / "test.jsonl",
        "test_formatted":             data / "test_formatted.jsonl",
        "test_lenmatched":            data / "test_lenmatched.jsonl",
        "test_lenmatched_formatted":  data / "test_lenmatched_formatted.jsonl",
    }
    results: dict = {"checkpoint": str(ckpt_p), "lang": a.lang, "per_corpus": {}}
    for name, p in corpora.items():
        print(f"\n[{name}] {p}", flush=True)
        r = eval_corpus(model, tok, args, p)
        print(f"  n={r['n']} acc={r['acc']:.4f} f1={r['f1']:.4f} AUROC={r['auroc']:.4f} "
              f"[{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]", flush=True)
        print(f"  mean_p1: human={r['machine_prob_mean_on_human_rows']:.4f} "
              f"machine={r['machine_prob_mean_on_machine_rows']:.4f}  "
              f"TPR={r['flag_rate_machine']:.4f}  FPR={r['flag_rate_human']:.4f}", flush=True)
        results["per_corpus"][name] = r

    # Q1 metric on test vs test_formatted (paired by index — both share same rows)
    a_t = results["per_corpus"]["test"]
    a_f = results["per_corpus"]["test_formatted"]
    prob_drop = a_t["machine_prob_mean_on_machine_rows"] - a_f["machine_prob_mean_on_machine_rows"]
    flag_drop = a_t["flag_rate_machine"] - a_f["flag_rate_machine"]
    # Length-matched analog
    l_t = results["per_corpus"]["test_lenmatched"]
    l_f = results["per_corpus"]["test_lenmatched_formatted"]
    prob_drop_lm = l_t["machine_prob_mean_on_machine_rows"] - l_f["machine_prob_mean_on_machine_rows"]
    flag_drop_lm = l_t["flag_rate_machine"] - l_f["flag_rate_machine"]

    results["q1_metrics"] = {
        "machine_prob_drop_test_vs_formatted":                    prob_drop,
        "flag_rate_drop_test_vs_formatted":                       flag_drop,
        "machine_prob_drop_lenmatched_vs_lenmatched_formatted":   prob_drop_lm,
        "flag_rate_drop_lenmatched_vs_lenmatched_formatted":      flag_drop_lm,
    }
    # Pre-registered Q1 criterion
    q1 = "yes" if (prob_drop >= 0.10 or flag_drop >= 0.10) else "no"
    results["q1_decision"] = q1
    print(f"\n=== Q1 pre-registered decision ===", flush=True)
    print(f"  machine-prob drop test→formatted = {prob_drop:+.4f}  (threshold ≥ 0.10)", flush=True)
    print(f"  flag-rate    drop test→formatted = {flag_drop:+.4f}  (threshold ≥ 0.10)", flush=True)
    print(f"  Q1 = {q1.upper()}", flush=True)

    # Length-tiebreaker: per-quartile AUC on the length-matched slice (single bucket
    # here — already length-matched; if AUC stays high, formatting is the lever, not
    # length). Treat the length-matched corpus AUC as the headline diagnostic.
    lm_auc = l_t["auroc"]
    if q1 == "yes":
        if lm_auc >= 0.90:
            tiebreaker = "use formatting (length-matched AUC stays high)"
        elif lm_auc <= 0.65:
            tiebreaker = "length explains it (length-matched AUC collapses; Phase C cannot fix)"
        else:
            tiebreaker = "ambiguous — surface to PI"
    else:
        tiebreaker = "n/a (Q1 = no; paper pivots to 'no formatting confound')"
    results["length_tiebreaker"] = tiebreaker
    print(f"  length-matched AUROC = {lm_auc:.4f}  →  {tiebreaker}", flush=True)

    out_p = Path(a.output_dir) / "eval_phase_b.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
