"""Validation: confirm our DroidDetect load reproduces the authors' performance on the
authors' OWN data (DroidCollection). This is the control that proves the high
false-positive rate we see on MatrixStudio is a genuine cross-source generalization
effect, not an implementation bug.

Writes results/droiddetect_validation.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401
from src import config as C

N_PER_CLASS = 120
LANG = "Python"


def main() -> None:
    from datasets import load_dataset
    from src.detectors import DroidDetect, DROID_MAX_TOKENS
    dd = DroidDetect()

    def score(code: str) -> float:
        enc = dd.tok(code, return_tensors="pt", truncation=True,
                     max_length=DROID_MAX_TOKENS).to(dd.model.text_encoder.device if hasattr(dd.model, "text_encoder") else "cuda")
        with torch.no_grad():
            lg = dd.model(enc["input_ids"], enc["attention_mask"])
        return F.softmax(lg.float(), dim=-1)[0, 1].item()

    ds = load_dataset("project-droid/DroidCollection", split="train", streaming=True)
    hum, mac = [], []
    for r in ds:
        if r.get("Language") != LANG:
            continue
        if r["Label"] == "HUMAN_GENERATED" and len(hum) < N_PER_CLASS:
            hum.append(r["Code"])
        elif r["Label"] == "MACHINE_GENERATED" and len(mac) < N_PER_CLASS:
            mac.append(r["Code"])
        if len(hum) >= N_PER_CLASS and len(mac) >= N_PER_CLASS:
            break

    hs = np.array([score(c) for c in hum])
    ms = np.array([score(c) for c in mac])
    y = np.r_[np.zeros(len(hs)), np.ones(len(ms))]
    s = np.r_[hs, ms]
    out = {
        "lang": LANG, "n_human": len(hs), "n_machine": len(ms),
        "human_mean_p_machine": float(hs.mean()),
        "human_flag_rate_argmax": float((hs > 0.5).mean()),
        "machine_flag_rate_argmax": float((ms > 0.5).mean()),
        "auroc_in_distribution": float(roc_auc_score(y, s)),
    }
    (C.RESULTS / "droiddetect_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
