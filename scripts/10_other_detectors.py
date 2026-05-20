"""Replicate the formatting-confound + legendary probe on OTHER trained detectors.

A web audit found NO clean out-of-the-box, standard-architecture AI-*code* detector on HF
besides the DroidDetect family (others are LoRA adapters / raw .pth / sklearn-joblib with
undocumented feature pipelines -> running them would require the reimplementation we forbid).
So this tests the DroidDetect FAMILY siblings (different checkpoints/sizes/granularity, same
authors+data) via the vendored verbatim TLModel loader:
  - DroidDetect-Base-Binary  (2-class, ModernBERT-base)  [baseline]
  - DroidDetect-Base         (4-class, ModernBERT-base)
  - DroidDetect-Large-Binary (2-class, ModernBERT-large)

For each: (a) does black-reformatting flip in-distribution human code to 'machine'?
          (b) flag rate on legendary pre-LLM elite code.
Output: results/other_detectors.json
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401
from src import config as C
from src.vendor.droiddetect_model import load_droiddetect

N = 150
REPOS = ["project-droid/DroidDetect-Base-Binary",
         "project-droid/DroidDetect-Base",
         "project-droid/DroidDetect-Large-Binary"]


def black_format(codes: list[str]) -> list[str | None]:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        for i, c in enumerate(codes):
            (dp / f"{i}.py").write_text(c)
        subprocess.run(["uvx", "black", "-q", str(dp)], capture_output=True, text=True)
        out = []
        for i, c in enumerate(codes):
            ok = True
            try:
                ast.parse(c)
            except Exception:
                ok = False
            out.append((dp / f"{i}.py").read_text() if ok else None)
        return out


def main() -> None:
    from datasets import load_dataset
    # corpora
    dc_h, dc_m = [], []
    for r in load_dataset("project-droid/DroidCollection", split="train", streaming=True):
        if r.get("Language") != "Python":
            continue
        if r["Label"] == "HUMAN_GENERATED" and len(dc_h) < N:
            dc_h.append(r["Code"])
        elif r["Label"] == "MACHINE_GENERATED" and len(dc_m) < N:
            dc_m.append(r["Code"])
        if len(dc_h) >= N and len(dc_m) >= N:
            break
    dc_h_black = black_format(dc_h)
    ms_h = pd.read_parquet(C.HUMAN_PARQUET).code.tolist()[:N]
    leg = pd.read_parquet(C.DATA / "legendary_code.parquet")

    results = {}
    for repo in REPOS:
        model, tok = load_droiddetect(repo, device="cuda")
        n_classes = model.classifier.out_features

        def p_ai(code: str) -> float:
            enc = tok(code, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                probs = F.softmax(model(enc["input_ids"], enc["attention_mask"]).float(), dim=-1)[0]
            # class 0 = HUMAN; P(AI) = 1 - P(human) (covers 2-class and 4-class heads)
            return float(1.0 - probs[0].item())

        def flag(codes):
            s = np.array([p_ai(c) for c in codes if c])
            return float((s > 0.5).mean()), int(len(s))

        fr_h, n_h = flag(dc_h)
        fr_hb, n_hb = flag([c for c in dc_h_black if c])
        fr_m, n_m = flag(dc_m)
        fr_ms, n_ms = flag(ms_h)
        by_author = {}
        for au, g in leg.groupby("author"):
            fa, na = flag(g.code.tolist())
            by_author[au] = {"flag_rate": fa, "n": na}
        results[repo] = {
            "n_classes": n_classes,
            "dc_human_original": fr_h, "dc_human_black": fr_hb,
            "dc_machine": fr_m, "matrixstudio_human": fr_ms,
            "legendary_by_author": by_author,
            "legendary_all": flag(leg.code.tolist())[0],
        }
        print(f"\n=== {repo} ({n_classes}-class) ===")
        print(f"  DC-human  original={fr_h:.3f} -> black={fr_hb:.3f}   (flip = formatting confound)")
        print(f"  DC-machine={fr_m:.3f}  MatrixStudio-human={fr_ms:.3f}  legendary-all={results[repo]['legendary_all']:.3f}")
        for au, v in by_author.items():
            print(f"    legendary {au:24s} {v['flag_rate']:.3f} (n={v['n']})")
        del model
        torch.cuda.empty_cache()

    (C.RESULTS / "other_detectors.json").write_text(json.dumps(results, indent=2))
    print("\nwrote results/other_detectors.json")


if __name__ == "__main__":
    main()
