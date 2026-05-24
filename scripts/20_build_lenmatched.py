"""A.2 — build length-matched HMCorp test slices.

For each language, produce `test_lenmatched.jsonl` and `test_lenmatched_formatted.jsonl`
in `data/hmcorp/{python,java}/`, where per-label BPE-token-length distributions match
within 5% on median+IQR. Used as a secondary diagnostic in Phase B.4(c) — does AUC
collapse when length signal is held constant?

Method: quantile-stratify by BPE-token-length (computed on the unformatted `code`
field using UniXcoder tokenizer; matches what the model sees), bin into NBINS equal-
size bins on the union distribution, then within each bin subsample so #human ==
#machine. The `_formatted` counterpart uses the same row indices.

Inputs:
  data/hmcorp/{python,java}/test.jsonl
  data/hmcorp/{python,java}/test_formatted.jsonl
Outputs:
  data/hmcorp/{python,java}/test_lenmatched.jsonl
  data/hmcorp/{python,java}/test_lenmatched_formatted.jsonl
  results/phase_a/lenmatched_stats.json

Usage:
  python3 -u scripts/20_build_lenmatched.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

from transformers import RobertaTokenizer  # noqa: E402

BASE = "microsoft/unixcoder-base-nine"
NBINS = 10
SEED = 0


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f]


def tokens_per_row(rows: list[dict], tok) -> np.ndarray:
    """BPE-token-length of the whitespace-collapsed `code` field, matching the
    tokenization the training script uses (see convert_examples_to_features)."""
    out = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i] = len(tok.tokenize(" ".join(r["code"].split())))
    return out


def median_iqr(x: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))


def build_one(lang: str, tok) -> dict:
    print(f"\n=== {lang} ===", flush=True)
    raw_p = ROOT / "data" / "hmcorp" / lang / "test.jsonl"
    fmt_p = ROOT / "data" / "hmcorp" / lang / "test_formatted.jsonl"
    out_raw = ROOT / "data" / "hmcorp" / lang / "test_lenmatched.jsonl"
    out_fmt = ROOT / "data" / "hmcorp" / lang / "test_lenmatched_formatted.jsonl"

    raw_rows = load_jsonl(raw_p)
    fmt_rows = load_jsonl(fmt_p)
    assert len(raw_rows) == len(fmt_rows), (len(raw_rows), len(fmt_rows))
    # 15_format_hmcorp wrote raw and formatted with the same row set in identical order
    # — sanity-check by index matching.
    for r, f in zip(raw_rows, fmt_rows):
        assert r["index"] == f["index"] and r["label"] == f["label"], (r["index"], f["index"])

    labels = np.asarray([r["label"] for r in raw_rows], dtype=np.int64)
    print(f"  loaded n={len(raw_rows)}  (h={int((labels==0).sum())}/m={int((labels==1).sum())})", flush=True)
    print(f"  tokenizing …", flush=True)
    lens = tokens_per_row(raw_rows, tok)

    # Per-label BEFORE matching
    for lbl, name in [(0, "human"), (1, "machine")]:
        m = labels == lbl
        med, q25, q75 = median_iqr(lens[m])
        print(f"  BEFORE {name:7s}: median={med:.1f}  IQR=[{q25:.1f}, {q75:.1f}]  n={int(m.sum())}",
              flush=True)

    # Quantile bins on the union (label-blind) distribution
    bin_edges = np.percentile(lens, np.linspace(0, 100, NBINS + 1))
    bin_edges[0] -= 1  # ensure min row lands in bin 0
    bin_edges[-1] += 1
    bin_ids = np.digitize(lens, bin_edges[1:-1])  # 0..NBINS-1

    rng = np.random.default_rng(SEED)
    keep_idx: list[int] = []
    bin_stats: list[dict] = []
    for b in range(NBINS):
        in_bin = np.where(bin_ids == b)[0]
        h_idx = in_bin[labels[in_bin] == 0]
        m_idx = in_bin[labels[in_bin] == 1]
        k = min(len(h_idx), len(m_idx))
        if k == 0:
            bin_stats.append({"bin": b, "kept_per_label": 0, "skipped": True})
            continue
        h_keep = rng.choice(h_idx, size=k, replace=False)
        m_keep = rng.choice(m_idx, size=k, replace=False)
        keep_idx.extend(h_keep.tolist())
        keep_idx.extend(m_keep.tolist())
        bin_stats.append({"bin": b, "kept_per_label": int(k),
                          "edge_lo": float(bin_edges[b]), "edge_hi": float(bin_edges[b + 1])})

    keep_idx.sort()
    print(f"  AFTER matching: kept {len(keep_idx)} rows ({len(keep_idx)//2} per label)", flush=True)

    # AFTER stats
    kept_lens = lens[keep_idx]
    kept_labels = labels[keep_idx]
    after_stats = {}
    for lbl, name in [(0, "human"), (1, "machine")]:
        m = kept_labels == lbl
        med, q25, q75 = median_iqr(kept_lens[m])
        print(f"  AFTER  {name:7s}: median={med:.1f}  IQR=[{q25:.1f}, {q75:.1f}]  n={int(m.sum())}",
              flush=True)
        after_stats[name] = {"median": med, "q25": q25, "q75": q75, "n": int(m.sum())}

    # Verify within-5% match on median+IQR
    h_med = after_stats["human"]["median"]
    m_med = after_stats["machine"]["median"]
    rel = abs(h_med - m_med) / max(h_med, m_med)
    print(f"  median match: rel-gap = {rel:.4f} (target ≤ 0.05)", flush=True)
    if rel > 0.05:
        print(f"  WARNING: median gap > 5% target — bin count {NBINS} may be too coarse "
              f"for this distribution. Consider tightening.", flush=True)

    # Write outputs
    with out_raw.open("w") as fr, out_fmt.open("w") as ff:
        for i in keep_idx:
            fr.write(json.dumps(raw_rows[i]) + "\n")
            ff.write(json.dumps(fmt_rows[i]) + "\n")
    print(f"  wrote {out_raw.name} + {out_fmt.name}", flush=True)

    return {
        "lang": lang,
        "n_in": len(raw_rows),
        "n_out": len(keep_idx),
        "median_human_after": h_med,
        "median_machine_after": m_med,
        "median_rel_gap": rel,
        "bins": bin_stats,
        "before_stats": {
            "human":   dict(zip(["median", "q25", "q75", "n"],
                                [*median_iqr(lens[labels == 0]), int((labels == 0).sum())])),
            "machine": dict(zip(["median", "q25", "q75", "n"],
                                [*median_iqr(lens[labels == 1]), int((labels == 1).sum())])),
        },
        "after_stats": after_stats,
    }


def main() -> None:
    tok = RobertaTokenizer.from_pretrained(BASE)
    out = {lang: build_one(lang, tok) for lang in ("python", "java")}
    out_p = ROOT / "results" / "phase_a" / "lenmatched_stats.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
