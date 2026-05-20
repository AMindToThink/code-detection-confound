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
    """Canonical formatting via AST round-trip: normalizes whitespace, drops # comments,
    removes redundant parens, normalizes quotes. Preserves identifiers/literals/docstrings."""
    try:
        return ast.unparse(ast.parse(code))
    except Exception:
        return None


def black_format_corpus(codes: list[str]) -> list[str | None]:
    """Format a list of code strings with `black` via `uvx` (ephemeral, no env changes).
    black PRESERVES comments and docstrings; it only restyles whitespace/indentation/quotes/
    line-wrapping. Returns black output per item, or None if black could not parse it."""
    import subprocess
    import tempfile
    out: list[str | None] = [None] * len(codes)
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        for i, c in enumerate(codes):
            (dp / f"{i}.py").write_text(c)
        # one ephemeral black over the whole dir; unparseable files are left untouched + reported
        subprocess.run(["uvx", "black", "-q", str(dp)], capture_output=True, text=True)
        for i, c in enumerate(codes):
            formatted = (dp / f"{i}.py").read_text()
            # black leaves unparseable files byte-identical; mark those None so "black" only
            # measures files black actually restyled or accepted-as-formatted.
            out[i] = formatted if _py_parses(c) else None
    return out


def _py_parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


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
    callable_transforms = {"original": lambda c: c, "ast_canon": ast_canon,
                           "strip_comments": strip_comments}
    # black is a batch (subprocess) transform — precompute per corpus
    print("formatting corpora with black (uvx, ephemeral)...", flush=True)
    black_versions = {cname: black_format_corpus(codes) for cname, codes in corpora.items()}
    transform_names = ["original", "black", "ast_canon", "strip_comments"]

    dd = DroidDetect()

    def droid_score(code: str) -> float:
        enc = dd.tok(code, return_tensors="pt", truncation=True, max_length=DROID_MAX_TOKENS).to("cuda")
        with torch.no_grad():
            lg = dd.model(enc["input_ids"], enc["attention_mask"])
        return F.softmax(lg.float(), dim=-1)[0, 1].item()

    # --- preflight: confirm ast_canon PRESERVES identifiers/words (only formatting changes),
    # so the ablation does NOT remove the legitimate lexical "humans vs AI use different words"
    # signal. Reports the mean fraction of the identifier-token multiset retained after canon.
    def ident_multiset(code):
        from collections import Counter
        names = Counter()
        try:
            for t in tokenize.generate_tokens(StringIO(code).readline):
                if t.type == tokenize.NAME:
                    names[t.string] += 1
        except Exception:
            return None
        return names

    pres = []
    for c in corpora["DroidCollection-human"]:
        cc = ast_canon(c)
        if cc is None:
            continue
        n0, n1 = ident_multiset(c), ident_multiset(cc)
        if n0 and n1:
            pres.append(sum((n0 & n1).values()) / max(sum(n0.values()), 1))
    identifier_preservation = float(np.mean(pres)) if pres else float("nan")

    out = {"n_target": N, "identifier_preservation_ast_canon": identifier_preservation,
           "transform_success": {}, "droiddetect_argmax": {}}
    print(f"identifier multiset preserved by ast_canon: {identifier_preservation:.3f}", flush=True)
    for tname in transform_names:
        out["droiddetect_argmax"][tname] = {}
        out["transform_success"][tname] = {}
        for cname, codes in corpora.items():
            if tname == "black":
                transformed = black_versions[cname]
            else:
                transformed = [callable_transforms[tname](c) for c in codes]
            ok = [t for t in transformed if t and len(t) >= 20]
            out["transform_success"][tname][cname] = len(ok) / max(len(codes), 1)
            scores = [droid_score(t) for t in ok]
            fr, lo, hi = flag_ci(scores, 0.5)
            out["droiddetect_argmax"][tname][cname] = {
                "flag_rate": fr, "ci_lo": lo, "ci_hi": hi, "n": len(ok),
                "mean_p_machine": float(np.mean(scores)) if scores else float("nan"),
            }
            print(f"DroidDetect [{tname:14s}] {cname:26s} flag@0.5={fr:.3f} (n={len(ok)})", flush=True)

    # save DroidDetect (incl. black) results NOW, before the slow statistical-detector pass,
    # so the headline figure can be built without waiting on it.
    (C.RESULTS / "format_ablation.json").write_text(json.dumps(out, indent=2))
    print("wrote results/format_ablation.json (DroidDetect; pre-statistical)", flush=True)

    # statistical detectors: robustness contrast (mean score shift under canonicalization)
    sd = StatDetectors()
    out["statistical_mean_score"] = {}
    for tname in ("original", "black", "ast_canon"):
        out["statistical_mean_score"][tname] = {}
        for cname, codes in corpora.items():
            if tname == "black":
                transformed = black_versions[cname]
            else:
                transformed = [callable_transforms[tname](c) for c in codes]
            ok = [t for t in transformed if t and len(t) >= 20][:80]   # subset: speed
            fd = [sd.score_one(t)["FastDetectGPT"] for t in ok]
            bn = [sd.score_one(t)["Binoculars"] for t in ok]
            out["statistical_mean_score"][tname][cname] = {
                "FastDetectGPT": float(np.nanmean(fd)), "Binoculars": float(np.nanmean(bn))}

    (C.RESULTS / "format_ablation.json").write_text(json.dumps(out, indent=2))
    print("wrote results/format_ablation.json")


if __name__ == "__main__":
    main()
