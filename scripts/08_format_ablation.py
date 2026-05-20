"""Format-artifact causal ablation.

Tests the mechanism behind DroidDetect's cross-source false positives: does it key on
superficial FORMATTING (whitespace, comments) rather than on human-vs-AI content?

Content-preserving transform = `ast.unparse(ast.parse(code))`: re-emits canonical Python
(normalized whitespace/spacing, comments and docstrings removed) WITHOUT changing program
semantics. Stdlib only — no environment changes.

Causal logic:
  * If MatrixStudio-human FPR DROPS after canonicalization -> detector keyed on its formatting.
  * If DroidCollection-human FPR RISES after canonicalization -> its 'human' signal was formatting/comments.
  * If the two human sources' FPRs CONVERGE after canonicalization -> formatting was the
    whole cross-source difference (the decisive cross-over test).
Statistical (zero-shot) detectors are scored too, as the robust contrast.

Outputs results/format_ablation.json and results/figures/fig7_format_ablation.png.
"""
from __future__ import annotations

import ast
import itertools
import json
import sys
import tokenize
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401
from src import config as C

RNG = np.random.default_rng(0)
N = 200


def ast_canon(code: str) -> str | None:
    """Canonical formatting via round-trip; strips comments/docstrings, normalizes ws."""
    try:
        return ast.unparse(ast.parse(code))
    except Exception:
        return None


def strip_comments(code: str) -> str | None:
    """Remove comments only, preserve layout (isolates comments from whitespace)."""
    try:
        toks = []
        for tok in tokenize.generate_tokens(StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                continue
            toks.append(tok)
        return tokenize.untokenize(toks)
    except Exception:
        return None


def flag_ci(scores, thr, n=2000):
    s = np.asarray(scores, float); s = s[~np.isnan(s)]
    if len(s) < 2:
        return float("nan"), float("nan"), float("nan")
    pt = float((s > thr).mean())
    b = [float((s[RNG.integers(0, len(s), len(s))] > thr).mean()) for _ in range(n)]
    return pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    from datasets import load_dataset
    import pandas as pd
    from src.detectors import DroidDetect, StatDetectors, DROID_MAX_TOKENS

    # ---- source corpora
    ms_human = pd.read_parquet(C.HUMAN_PARQUET).code.tolist()[:N]
    dc_human, dc_machine = [], []
    for r in load_dataset("project-droid/DroidCollection", split="train", streaming=True):
        if r.get("Language") != "Python":
            continue
        if r["Label"] == "HUMAN_GENERATED" and len(dc_human) < N:
            dc_human.append(r["Code"])
        elif r["Label"] == "MACHINE_GENERATED" and len(dc_machine) < N:
            dc_machine.append(r["Code"])
        if len(dc_human) >= N and len(dc_machine) >= N:
            break

    corpora = {"MatrixStudio-human": ms_human,
               "DroidCollection-human": dc_human,
               "DroidCollection-machine": dc_machine}
    transforms = {"original": lambda c: c, "ast_canon": ast_canon, "strip_comments": strip_comments}

    dd = DroidDetect()

    def droid_score(code: str) -> float:
        enc = dd.tok(code, return_tensors="pt", truncation=True, max_length=DROID_MAX_TOKENS).to("cuda")
        with torch.no_grad():
            lg = dd.model(enc["input_ids"], enc["attention_mask"])
        return F.softmax(lg.float(), dim=-1)[0, 1].item()

    out = {"n_target": N, "transform_success": {}, "droiddetect_argmax": {}}
    for tname, fn in transforms.items():
        out["droiddetect_argmax"][tname] = {}
        out["transform_success"][tname] = {}
        for cname, codes in corpora.items():
            transformed = [fn(c) for c in codes]
            ok = [t for t in transformed if t and len(t) >= 20]
            out["transform_success"][tname][cname] = len(ok) / max(len(codes), 1)
            scores = [droid_score(t) for t in ok]
            fr, lo, hi = flag_ci(scores, 0.5)
            out["droiddetect_argmax"][tname][cname] = {
                "flag_rate": fr, "ci_lo": lo, "ci_hi": hi, "n": len(ok),
                "mean_p_machine": float(np.mean(scores)) if scores else float("nan"),
            }
            print(f"DroidDetect [{tname:14s}] {cname:26s} flag@0.5={fr:.3f} (n={len(ok)})", flush=True)

    # statistical detectors: robustness contrast (mean score shift under canonicalization)
    sd = StatDetectors()
    out["statistical_mean_score"] = {}
    for tname in ("original", "ast_canon"):
        fn = transforms[tname]
        out["statistical_mean_score"][tname] = {}
        for cname, codes in corpora.items():
            ok = [fn(c) for c in codes]; ok = [t for t in ok if t and len(t) >= 20]
            fd = [sd.score_one(t)["FastDetectGPT"] for t in ok]
            bn = [sd.score_one(t)["Binoculars"] for t in ok]
            out["statistical_mean_score"][tname][cname] = {
                "FastDetectGPT": float(np.nanmean(fd)), "Binoculars": float(np.nanmean(bn))}

    (C.RESULTS / "format_ablation.json").write_text(json.dumps(out, indent=2))
    print("wrote results/format_ablation.json")


if __name__ == "__main__":
    main()
