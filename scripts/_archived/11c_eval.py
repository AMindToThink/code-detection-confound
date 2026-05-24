"""Evaluate an Experiment-A checkpoint and record the formatting-flip metrics.

For a trained model, score the DroidCollection-Python TEST set presented two ways:
raw and black-normalized. Reports, per presentation: human flag rate (P(machine)>0.5 on
human rows = false-positive rate), machine flag rate (recall), and AUROC.

The headline comparison is the human flag rate raw->black:
  - the ORIGINAL-trained model should reproduce the confound (low raw -> high black);
  - the BLACK-trained model should NOT flip (formatting cue removed at training time).

Writes/updates results/experiment_a.json under the given --tag.

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u scripts/11c_eval.py --model-dir results/train/original/best --tag original_model
  CUDA_VISIBLE_DEVICES=1 python -u scripts/11c_eval.py --model-dir results/train/black/best    --tag black_model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401
from src import config as C

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 512
BATCH = 128


@torch.no_grad()
def p_machine(model, tok, codes: list[str], device="cuda") -> np.ndarray:
    out = []
    for i in range(0, len(codes), BATCH):
        enc = tok(codes[i:i + BATCH], return_tensors="pt", truncation=True,
                  max_length=MAX_LEN, padding=True).to(device)
        logits = model(**enc).logits.float()
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics_for(df: pd.DataFrame, pm: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    lab = df.label.to_numpy()
    flag = pm > 0.5
    return {
        "auroc": float(roc_auc_score(lab, pm)),
        "human_flag_rate": float(flag[lab == 0].mean()),   # false-positive rate
        "machine_flag_rate": float(flag[lab == 1].mean()),  # recall
        "n_human": int((lab == 0).sum()), "n_machine": int((lab == 1).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval().to("cuda").half()

    test_raw = pd.read_parquet(C.DATA / "droid_py_test.parquet")
    test_black = pd.read_parquet(C.DATA / "droid_py_test_black.parquet")
    res = {}
    for name, df in [("test_raw", test_raw), ("test_black", test_black)]:
        pm = p_machine(model, tok, df.code.tolist())
        res[name] = metrics_for(df, pm)
        print(f"[{args.tag}] {name}: human_FPR={res[name]['human_flag_rate']:.3f} "
              f"machine_recall={res[name]['machine_flag_rate']:.3f} auroc={res[name]['auroc']:.3f}",
              flush=True)
    # human flip under black = the headline confound magnitude
    res["human_flip_raw_to_black"] = (res["test_black"]["human_flag_rate"]
                                      - res["test_raw"]["human_flag_rate"])
    print(f"[{args.tag}] human flip raw->black = {res['human_flip_raw_to_black']:+.3f}", flush=True)

    path = C.RESULTS / "experiment_a.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres[args.tag] = res
    path.write_text(json.dumps(allres, indent=2))
    print(f"wrote {path} [{args.tag}]", flush=True)


if __name__ == "__main__":
    main()
