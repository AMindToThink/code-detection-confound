"""Phase D — OOD generalization eval for a Python CodeGPTSensor checkpoint.

Corpora (Python-only; we have no Java OOD data):
  1. data/human_code.parquet         — MatrixStudio Codeforces humans-only (614)
  2. data/legendary_code.parquet     — pre-LLM Python "legendary" humans (135)
                                       (Sanfilippo/Hipp/CPython/Torvalds, 2005-2013)
  3. data/droid_py_test.parquet      — DroidCollection Python test (mixed, 23,405)

Headline metric: human-flag-rate (i.e. FPR on humans-only corpora). Compare against
DroidDetect-Base-Binary's ~62% FPR on MatrixStudio (vendored result) and ~78.5% on
legendary. If CodeGPTSensor stays low, the formatting-vs-generalization story is:
DroidDetect failed due to formatting; CodeGPTSensor doesn't have either failure mode.

Also reports ECE (Expected Calibration Error, 10 bins) and a fixed-FPR operating point
(threshold at HMCorp-test FPR=5%) so the comparison isn't only at the default 0.5
threshold.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/27_eval_phase_d.py \\
      --output_dir results/cgs/python_raw_ce --eval_batch_size 16
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

import numpy as _np  # numpy 2 shim for cgs's early_stopping (imported transitively)
if not hasattr(_np, "Inf"):
    _np.Inf = _np.inf

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score  # noqa: E402

# Reuse AMPModel from training script
_spec = importlib.util.spec_from_file_location("train_mod", ROOT / "scripts" / "18_train_cgs_amp.py")
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)
AMPModel = train_mod.AMPModel

BASE = "microsoft/unixcoder-base-nine"
BLOCK = 400
N_BOOT = 500
SEED = 0


class CodeOnlyDataset(Dataset):
    """Eval-only dataset: tokenize one `code` column from arbitrary rows. Mirrors
    convert_examples_to_features whitespace-collapse + special tokens layout but
    skips the `contrast` field (returns a dummy contrast for the model signature).
    All rows assumed label=0 unless explicit labels supplied."""

    def __init__(self, codes: list[str], labels: list[int] | None, tokenizer):
        self.codes = codes
        self.labels = labels if labels is not None else [0] * len(codes)
        self.tok = tokenizer

    def __len__(self):
        return len(self.codes)

    def _ids(self, text: str) -> list[int]:
        text = " ".join(text.split())
        toks = self.tok.tokenize(text)[: BLOCK - 4]
        toks = [self.tok.cls_token, "<encoder_only>", self.tok.sep_token] + toks + [self.tok.sep_token]
        ids = self.tok.convert_tokens_to_ids(toks)
        ids += [self.tok.pad_token_id] * (BLOCK - len(ids))
        return ids

    def __getitem__(self, i):
        ids = self._ids(self.codes[i])
        # contrast is unused at eval (args.contrast=False, args.do_train=False)
        dummy = ids
        return (torch.tensor(ids), torch.tensor(dummy),
                torch.tensor(self.labels[i]), torch.tensor(i))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard binary-classification ECE: for each row, confidence in the model's
    PREDICTED class = max(p1, 1-p1), and accuracy in the bin = fraction of rows
    correctly classified. Bins are on confidence (not on p1), so an over-confident
    50:50 prediction lands in the lowest-confidence bin regardless of which class
    it predicts.

    ECE = sum_b (n_b / N) * |mean_confidence_b - accuracy_b|
    """
    conf = np.maximum(probs, 1.0 - probs)              # confidence in chosen class
    correct = (labels == (probs > 0.5)).astype(float)  # 1 if right, 0 if wrong
    bins = np.linspace(0.5, 1.0, n_bins + 1)            # confidence ranges from 0.5 to 1.0
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


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
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def predict(model, ds: CodeOnlyDataset, batch: int, device) -> tuple[np.ndarray, np.ndarray]:
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=batch,
                    num_workers=4, pin_memory=True)
    model.eval()
    p1, ys = [], []
    with torch.inference_mode():
        for b in dl:
            inp = b[0].to(device, non_blocking=True)
            con = b[1].to(device, non_blocking=True)
            lab = b[2].to(device, non_blocking=True)
            _, logit = model(inp, con, lab)
            prob = F.softmax(logit.float(), dim=-1)
            p1.append(prob[:, 1].cpu().numpy())
            ys.append(lab.cpu().numpy())
    return np.concatenate(p1), np.concatenate(ys)


def find_threshold_at_fpr(y: np.ndarray, p1: np.ndarray, target_fpr: float) -> float:
    """Return the score threshold s.t. FPR on the human (label=0) class is ≤ target."""
    hum = p1[y == 0]
    if len(hum) == 0:
        return 0.5
    # threshold = (1 - target)-th percentile of human scores: above it, target_fpr humans
    return float(np.quantile(hum, 1.0 - target_fpr))


def report_corpus(name: str, p1: np.ndarray, y: np.ndarray, fixed_thr: float | None) -> dict:
    out = {"n": int(len(y)), "machine_prob_mean": float(p1.mean())}
    # Human-only corpora carry only y=0; AUROC undefined
    has_both = len(np.unique(y)) > 1
    preds_05 = p1 > 0.5
    out["flag_rate_at_0.5"] = float(preds_05.mean())  # fraction predicted machine
    out["flag_rate_human_at_0.5"] = float(preds_05[y == 0].mean()) if (y == 0).any() else None
    out["flag_rate_machine_at_0.5"] = float(preds_05[y == 1].mean()) if (y == 1).any() else None
    if has_both:
        auc = float(roc_auc_score(y, p1))
        lo, hi = bootstrap_auc(y, p1)
        out["auroc"] = auc
        out["auroc_ci"] = [lo, hi]
        out["acc"] = float(accuracy_score(y, preds_05))
        out["f1"] = float(f1_score(y, preds_05))
        out["ece"] = expected_calibration_error(p1, y)
    if fixed_thr is not None:
        preds_t = p1 > fixed_thr
        out["fixed_threshold"] = fixed_thr
        out["flag_rate_at_fixed"] = float(preds_t.mean())
        out["flag_rate_human_at_fixed"] = float(preds_t[y == 0].mean()) if (y == 0).any() else None
        out["flag_rate_machine_at_fixed"] = float(preds_t[y == 1].mean()) if (y == 1).any() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    a = ap.parse_args()

    args = argparse.Namespace(
        block_size=BLOCK, eval_batch_size=a.eval_batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amp=False, do_train=False, contrast=False,
    )

    print(f"Loading {BASE} + {a.output_dir}/checkpoint-best-f1/model.bin", flush=True)
    tok = RobertaTokenizer.from_pretrained(BASE)
    cfg = RobertaConfig.from_pretrained(BASE)
    enc = RobertaModel.from_pretrained(BASE)
    model = AMPModel(enc, cfg, tok, args).to(args.device)
    ckpt = Path(a.output_dir) / "checkpoint-best-f1" / "model.bin"
    model.load_state_dict(torch.load(ckpt, map_location=args.device))

    # First: HMCorp-test calibration → 5% FPR threshold on the in-distribution human
    # distribution. We then apply this threshold on OOD corpora to get a "calibrated"
    # FPR on OOD humans.
    import json as _json
    hmcorp_test = ROOT / "data" / "hmcorp" / "python" / "test.jsonl"
    rows = [_json.loads(l) for l in hmcorp_test.open()]
    codes = [r["code"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    print(f"\n[HMCorp-test calibration] n={len(codes)}", flush=True)
    p1, y = predict(model, CodeOnlyDataset(codes, labels, tok), a.eval_batch_size, args.device)
    thr_5pct = find_threshold_at_fpr(y, p1, 0.05)
    print(f"  threshold @ in-dist FPR=5%: {thr_5pct:.4f}", flush=True)
    hmcorp_report = report_corpus("hmcorp_python_test", p1, y, thr_5pct)
    print(f"  {hmcorp_report}", flush=True)

    results: dict = {
        "checkpoint": str(ckpt),
        "lang": "python",
        "calibration": {"fpr_5pct_threshold": thr_5pct},
        "per_corpus": {"hmcorp_python_test": hmcorp_report},
    }

    # OOD corpora
    for name, path in [
        ("matrixstudio_humans", ROOT / "data" / "human_code.parquet"),
        ("legendary_humans",    ROOT / "data" / "legendary_code.parquet"),
        ("droidcollection_test", ROOT / "data" / "droid_py_test.parquet"),
    ]:
        print(f"\n[{name}] {path}", flush=True)
        df = pd.read_parquet(path)
        codes = df["code"].tolist()
        labels = df["label"].astype(int).tolist() if "label" in df.columns else [0] * len(codes)
        ds = CodeOnlyDataset(codes, labels, tok)
        p1, y = predict(model, ds, a.eval_batch_size, args.device)
        rep = report_corpus(name, p1, y, thr_5pct)
        print(f"  {rep}", flush=True)
        results["per_corpus"][name] = rep

    out_p = Path(a.output_dir) / "eval_phase_d.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
