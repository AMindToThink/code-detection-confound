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

import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import black
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

SAMPLE_PER_LABEL = 3000
SEED = 0
JAR = str((C.ROOT / "external/google-java-format.jar").resolve())


def py_is_compliant(code: str) -> bool:
    """True iff black is a no-op on `code` — same metric used for DroidDetect audit."""
    try:
        return black.format_str(code, mode=black.Mode()) == code
    except Exception:
        return False


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


def audit_python_jsonl(path: Path, n_per_label: int, label_field: str = "code") -> dict:
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in path.open()]
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[int(r["label"])].append(r[label_field])
    out: dict[int, dict] = {}
    for lbl, codes in by_label.items():
        sample = rng.sample(codes, min(n_per_label, len(codes)))
        with ProcessPoolExecutor(max_workers=14) as P:
            flags = list(P.map(py_is_compliant, sample, chunksize=50))
        out[lbl] = {"n_sampled": len(sample), "n_total": len(codes),
                    "n_compliant": int(sum(flags)),
                    "compliance_rate": sum(flags) / len(sample) if sample else 0.0}
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
        with ProcessPoolExecutor(max_workers=14) as P:
            flags = list(P.map(py_is_compliant, sample, chunksize=50))
        out[lbl] = {"n_sampled": len(sample), "n_total": len(codes),
                    "n_compliant": int(sum(flags)),
                    "compliance_rate": sum(flags) / len(sample) if sample else 0.0}
    return out


def main() -> None:
    out: dict = {}
    print("=== HMCorp Python (CodeGPTSensor training data) ===", flush=True)
    out["hmcorp_python"] = audit_python_jsonl(
        C.ROOT / "external/CodeGPTSensor/dataset/python/train.jsonl",
        SAMPLE_PER_LABEL, label_field="code",
    )
    for lbl, name in [(0, "human"), (1, "machine")]:
        d = out["hmcorp_python"][lbl]
        print(f"  {name:7s}: {d['n_compliant']}/{d['n_sampled']} = {d['compliance_rate']:.4f}",
              flush=True)

    print("\n=== HMCorp Java (CodeGPTSensor training data) ===", flush=True)
    out["hmcorp_java"] = audit_java_jsonl(
        C.ROOT / "external/CodeGPTSensor/dataset/java/train.jsonl",
        SAMPLE_PER_LABEL,
    )
    for lbl, name in [(0, "human"), (1, "machine")]:
        d = out["hmcorp_java"][lbl]
        print(f"  {name:7s}: {d['n_compliant']}/{d['n_sampled']} = {d['compliance_rate']:.4f}",
              flush=True)

    print("\n=== DroidCollection Python (DroidDetect training data) ===", flush=True)
    out["droid_python"] = audit_droid_parquet(
        C.ROOT / "data/droid_py_train.parquet", SAMPLE_PER_LABEL,
    )
    for lbl, name in [(0, "human"), (1, "machine")]:
        d = out["droid_python"][lbl]
        print(f"  {name:7s}: {d['n_compliant']}/{d['n_sampled']} = {d['compliance_rate']:.4f}",
              flush=True)

    # Headline: gap = machine_rate - human_rate
    print("\n=== COMPLIANCE GAP per dataset ===", flush=True)
    for ds in ("hmcorp_python", "hmcorp_java", "droid_python"):
        h = out[ds][0]["compliance_rate"]
        m = out[ds][1]["compliance_rate"]
        gap = m - h
        print(f"  {ds:18s}: human={h:.4f}  machine={m:.4f}  gap={gap:+.4f}", flush=True)
        out[ds]["gap_machine_minus_human"] = gap

    out_p = C.ROOT / "results" / "phase_e" / "compliance_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
