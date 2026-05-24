"""A.5 — retroactive per-label parse-fail-rate audit on the format pipeline.

`scripts/15_format_hmcorp.py` filtered rows where EITHER `code` or `contrast` failed
to parse/format. The audit-agent concern: if `contrast` (gpt-3.5 output) was more
likely to fail than `code` (human), we'd keep machine-easy rows preferentially,
biasing the retained pool. This script computes, per (lang, split, label), the rate
at which rows were dropped, broken down by which side (`code` or `contrast`) failed.

Method: match indices between
  external/CodeGPTSensor/dataset/{lang}/{split}.jsonl   (original)
  data/hmcorp/{lang}/{split}.jsonl                       (filtered)
and re-run the parse/format check on dropped rows to attribute the failure stage.

Usage:
  python3 -u scripts/21_audit_parsefilter.py
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import black
import javalang
import javalang.parse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

JAR = str((C.ROOT / "external/google-java-format.jar").resolve())
ORIG_DIR = C.ROOT / "external/CodeGPTSensor/dataset"
FILT_DIR = C.DATA / "hmcorp"


def py_check(code: str) -> str:
    try:
        ast.parse(code)
    except Exception:
        return "parse_fail"
    try:
        black.format_str(code, mode=black.Mode())
    except Exception:
        return "format_fail"
    return "ok"


def java_check_single(code: str) -> str:
    # Same wrap-and-format approach as scripts/15_format_hmcorp.py
    try:
        javalang.parse.parse("class _W {\n" + code + "\n}")
    except Exception:
        return "parse_fail"
    with tempfile.NamedTemporaryFile("w", suffix=".java", delete=False) as f:
        f.write(f"class A_x {{\n{code}\n}}\n")
        p = f.name
    try:
        r = subprocess.run(["java", "-jar", JAR, "-i", p],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return "format_fail"
        return "ok"
    except Exception:
        return "format_fail"
    finally:
        Path(p).unlink(missing_ok=True)


def audit_split(lang: str, split: str) -> dict:
    print(f"\n=== {lang}/{split} ===", flush=True)
    orig_p = ORIG_DIR / lang / f"{split}.jsonl"
    filt_p = FILT_DIR / lang / f"{split}.jsonl"

    orig_rows = [json.loads(l) for l in orig_p.open()]
    filt_indexes = {json.loads(l)["index"] for l in filt_p.open()}
    print(f"  original n={len(orig_rows)}  kept n={len(filt_indexes)}  "
          f"dropped n={len(orig_rows) - len(filt_indexes)}", flush=True)

    # Count per-label drop rates from index match alone (cheap, exact)
    per_label = {0: {"orig": 0, "kept": 0}, 1: {"orig": 0, "kept": 0}}
    for r in orig_rows:
        lbl = int(r["label"])
        per_label[lbl]["orig"] += 1
        if r["index"] in filt_indexes:
            per_label[lbl]["kept"] += 1
    for lbl, name in [(0, "human"), (1, "machine")]:
        s = per_label[lbl]
        drop_rate = 1.0 - s["kept"] / max(s["orig"], 1)
        print(f"  {name:7s}: orig {s['orig']:6d}  kept {s['kept']:6d}  "
              f"drop_rate {drop_rate:.4f}", flush=True)

    # On a sample of dropped rows, attribute fate to code-vs-contrast and parse-vs-format
    # (don't run all dropped rows — Java is slow due to JVM startup per row)
    dropped = [r for r in orig_rows if r["index"] not in filt_indexes]
    SAMPLE = min(200, len(dropped))
    if SAMPLE == 0:
        return {"lang": lang, "split": split, "per_label": per_label, "attribution": {}}

    import random
    random.seed(0)
    sample = random.sample(dropped, SAMPLE)
    print(f"  attributing fate on {SAMPLE} sampled dropped rows …", flush=True)
    checker = py_check if lang == "python" else java_check_single

    fates = {"code_parse": 0, "code_format": 0,
             "contrast_parse": 0, "contrast_format": 0,
             "both_ok_should_not_have_dropped": 0}
    per_label_attr = {0: dict(fates), 1: dict(fates)}
    for r in sample:
        cf = checker(r["code"])
        co = checker(r["contrast"])
        lbl = int(r["label"])
        # First non-ok wins (mirrors short-circuit logic in 15_format_hmcorp.py)
        if cf == "parse_fail":
            per_label_attr[lbl]["code_parse"] += 1
        elif cf == "format_fail":
            per_label_attr[lbl]["code_format"] += 1
        elif co == "parse_fail":
            per_label_attr[lbl]["contrast_parse"] += 1
        elif co == "format_fail":
            per_label_attr[lbl]["contrast_format"] += 1
        else:
            per_label_attr[lbl]["both_ok_should_not_have_dropped"] += 1

    for lbl, name in [(0, "human"), (1, "machine")]:
        a = per_label_attr[lbl]
        total = sum(a.values())
        if total:
            print(f"  {name:7s} fate (n={total}): "
                  + ", ".join(f"{k}={v} ({v/total:.0%})" for k, v in a.items() if v),
                  flush=True)

    return {"lang": lang, "split": split, "per_label": per_label,
            "attribution_sample_n": SAMPLE, "attribution": per_label_attr}


def main() -> None:
    out = []
    for lang in ("python", "java"):
        for split in ("train", "valid", "test"):
            out.append(audit_split(lang, split))
    out_p = C.ROOT / "results" / "phase_a" / "parsefilter_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
