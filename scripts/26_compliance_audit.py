"""WHY investigation — per-label formatter compliance audit.

Tests the hypothesis: DroidDetect's catastrophic formatting confound exists because
its training data has a stark per-label compliance gap (machine 44% black-compliant
vs human 15%); CodeGPTSensor's HMCorp does NOT exhibit this confound because its
per-label compliance gap is similar across classes (no shortcut).

For each (dataset, language, label), sample N rows from the ORIGINAL (unfiltered)
data and compute the fraction where the canonical formatter is a no-op
(`format(code) == code`). The unfiltered source matters: scripts/15_format_hmcorp.py
already drops rows where black/google-java-format failed, which would artificially
inflate compliance in the kept set.

Inputs (all expected to exist already):
  external/CodeGPTSensor/dataset/python/train.jsonl   (HMCorp Python original)
  external/CodeGPTSensor/dataset/java/train.jsonl     (HMCorp Java original)
  data/droid_py_train.parquet                          (DroidCollection Python)

Outputs:
  results/phase_e/compliance_audit.json
"""
from __future__ import annotations

import difflib
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import black
import pandas as pd

# Metric notes:
# - 'strict':     black.format_str(code) == code  (literal equality)
# - 'near_p98':   SequenceMatcher(code, black(code)).ratio() > 0.98  (matches the
#                 historic scripts/09_formatting_confound.py metric — "already
#                 near-black-canonical" — so we can compare against memory numbers)
# - 'similarity': mean SequenceMatcher ratio (continuous, for stratification later)
NEAR_THRESHOLD = 0.98

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

SAMPLE_PER_LABEL = 3000
SEED = 0
JAR = str((C.ROOT / "external/google-java-format.jar").resolve())


def py_compliance_metrics(code: str) -> tuple[bool, bool, float] | None:
    """Return (strict_equal, near_p98, similarity_ratio). None if black fails."""
    try:
        formatted = black.format_str(code, mode=black.Mode())
    except Exception:
        return None
    sim = difflib.SequenceMatcher(None, code, formatted).ratio()
    return (formatted == code, sim > NEAR_THRESHOLD, sim)


def py_is_compliant(code: str) -> bool:
    """Legacy entry point — strict equality only. Kept for backwards-compat callers."""
    m = py_compliance_metrics(code)
    return bool(m[0]) if m is not None else False


def java_compliance_batch(codes: list[str]) -> list[bool]:
    """For each method-level snippet: wrap in class A_{i} { ... }, format with
    google-java-format, strip the wrapper, compare to input. Batches into one JVM
    invocation to amortize ~500 ms startup."""
    if not codes:
        return []
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, c in enumerate(codes):
            p = Path(d) / f"A_{i:08d}.java"
            p.write_text(f"class A_{i:08d} {{\n{c}\n}}\n")
            files.append(p)
        try:
            subprocess.run(
                ["java", "-jar", JAR, "-i", *map(str, files)],
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            return [False] * len(codes)
        out: list[bool] = []
        for i, c in enumerate(codes):
            try:
                txt = (Path(d) / f"A_{i:08d}.java").read_text()
            except Exception:
                out.append(False); continue
            # strip wrapper symmetric with 15_format_hmcorp.strip_wrapper
            lines = txt.splitlines()
            start = 0
            while start < len(lines) and not lines[start].rstrip().endswith("{"):
                start += 1
            if start >= len(lines):
                out.append(False); continue
            body = lines[start + 1:]
            while body and body[-1].strip() == "":
                body.pop()
            if not body or body[-1].strip() != "}":
                out.append(False); continue
            body.pop()
            stripped = "\n".join((line[2:] if line.startswith("  ") else line) for line in body)
            stripped = stripped.rstrip() + "\n"
            out.append(stripped == c)
    return out


def _summarize_py(sample: list[str]) -> dict:
    with ProcessPoolExecutor(max_workers=14) as P:
        results = list(P.map(py_compliance_metrics, sample, chunksize=50))
    strict = [bool(r[0]) for r in results if r is not None]
    near = [bool(r[1]) for r in results if r is not None]
    sims = [float(r[2]) for r in results if r is not None]
    n_ok = len(strict)
    return {
        "n_sampled": len(sample),
        "n_black_ok": n_ok,
        "compliance_rate_strict": (sum(strict) / n_ok) if n_ok else 0.0,
        "compliance_rate_near_p98": (sum(near) / n_ok) if n_ok else 0.0,
        "similarity_mean": (sum(sims) / n_ok) if n_ok else 0.0,
        "similarity_median": (sorted(sims)[n_ok // 2] if n_ok else 0.0),
    }


def audit_python_jsonl(path: Path, n_per_label: int, label_field: str = "code") -> dict:
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in path.open()]
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[int(r["label"])].append(r[label_field])
    out: dict[int, dict] = {}
    for lbl, codes in by_label.items():
        sample = rng.sample(codes, min(n_per_label, len(codes)))
        out[lbl] = _summarize_py(sample)
        out[lbl]["n_total"] = len(codes)
    return out


def audit_java_jsonl(path: Path, n_per_label: int) -> dict:
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in path.open()]
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[int(r["label"])].append(r["code"])
    out: dict[int, dict] = {}
    for lbl, codes in by_label.items():
        sample = rng.sample(codes, min(n_per_label, len(codes)))
        # Chunk to keep JVM under-time + each shell argv-list short
        flags: list[bool] = []
        CHUNK = 200
        for s in range(0, len(sample), CHUNK):
            flags.extend(java_compliance_batch(sample[s:s + CHUNK]))
        out[lbl] = {"n_sampled": len(sample), "n_total": len(codes),
                    "n_compliant": int(sum(flags)),
                    "compliance_rate": sum(flags) / len(sample) if sample else 0.0}
    return out


def audit_droid_parquet(path: Path, n_per_label: int) -> dict:
    rng = random.Random(SEED)
    df = pd.read_parquet(path)
    by_label = {0: [], 1: []}
    for _, r in df.iterrows():
        by_label[int(r["label"])].append(r["code"])
    out: dict[int, dict] = {}
    for lbl, codes in by_label.items():
        sample = rng.sample(codes, min(n_per_label, len(codes)))
        out[lbl] = _summarize_py(sample)
        out[lbl]["n_total"] = len(codes)
    return out


def _print_py(name: str, d_by_label: dict) -> None:
    print(f"\n=== {name} ===", flush=True)
    for lbl, who in [(0, "human"), (1, "machine")]:
        d = d_by_label[lbl]
        print(f"  {who:7s}: n={d['n_sampled']}  strict={d['compliance_rate_strict']:.4f}  "
              f"near(>{NEAR_THRESHOLD})={d['compliance_rate_near_p98']:.4f}  "
              f"sim_mean={d['similarity_mean']:.4f}  sim_median={d['similarity_median']:.4f}",
              flush=True)


def main() -> None:
    out: dict = {}

    out["hmcorp_python"] = audit_python_jsonl(
        C.ROOT / "external/CodeGPTSensor/dataset/python/train.jsonl",
        SAMPLE_PER_LABEL, label_field="code",
    )
    _print_py("HMCorp Python (CodeGPTSensor training data)", out["hmcorp_python"])

    out["hmcorp_java"] = audit_java_jsonl(
        C.ROOT / "external/CodeGPTSensor/dataset/java/train.jsonl",
        SAMPLE_PER_LABEL,
    )
    # Java reports strict-equality only (google-java-format is a JVM batch job; no
    # similarity metric needed since the JVM cost dominates anyway)
    print(f"\n=== HMCorp Java (strict equality only) ===", flush=True)
    for lbl, who in [(0, "human"), (1, "machine")]:
        d = out["hmcorp_java"][lbl]
        n_sampled = d.get("n_sampled", 0)
        n_compliant = d.get("n_compliant", 0)
        rate = (n_compliant / n_sampled) if n_sampled else 0.0
        print(f"  {who:7s}: n={n_sampled}  strict={rate:.4f}", flush=True)

    out["droid_python"] = audit_droid_parquet(
        C.ROOT / "data/droid_py_train.parquet", SAMPLE_PER_LABEL,
    )
    _print_py("DroidCollection Python (DroidDetect training data)", out["droid_python"])

    # Headline gaps under both metrics
    print("\n=== COMPLIANCE GAP per dataset (machine - human) ===", flush=True)
    for ds in ("hmcorp_python", "droid_python"):
        for metric in ("compliance_rate_strict", "compliance_rate_near_p98"):
            h = out[ds][0][metric]
            m = out[ds][1][metric]
            gap = m - h
            short = "strict" if "strict" in metric else "near"
            print(f"  {ds:16s} [{short:>6s}]:  human={h:.4f}  machine={m:.4f}  gap={gap:+.4f}",
                  flush=True)
            out[ds][f"gap_{short}"] = gap

    out_p = C.ROOT / "results" / "phase_e" / "compliance_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
