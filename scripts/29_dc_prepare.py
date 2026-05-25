"""Convert DroidCollection-Python parquet (data/droid_py_train_bal.parquet +
droid_py_test.parquet + droid_py_test_black.parquet) into the HMCorp-format JSONL
the CodeGPTSensor training script expects: {index, code, contrast, label}.

`contrast` is unused under CE-only training; we set it to `code` (the model never
reads contrast when args.contrast=False and args.do_train=False/True).

Also carves out a small valid split (5% stratified) from the balanced train set so
the EarlyStopping logic + per-epoch eval has a real held-out signal.

Outputs:
  data/droid_py/train.jsonl
  data/droid_py/valid.jsonl
  data/droid_py/test.jsonl
  data/droid_py/test_formatted.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

OUT = C.DATA / "droid_py"
SEED = 0
VALID_FRAC = 0.05


def to_rows(df: pd.DataFrame, prefix: str) -> list[dict]:
    out = []
    for i, r in df.iterrows():
        out.append({
            "index": f"{prefix}{i:06d}",   # last-6-chars-as-int parsing works
            "code": r["code"],
            "contrast": r["code"],          # unused under CE-only
            "label": int(r["label"]),
        })
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("[train+valid] reading droid_py_train_bal.parquet …", flush=True)
    train_df = pd.read_parquet(C.DATA / "droid_py_train_bal.parquet")
    print(f"  n={len(train_df)} labels={train_df['label'].value_counts().to_dict()}", flush=True)

    # Stratified 95/5 split
    idx_by_label = {0: [], 1: []}
    for i, lbl in enumerate(train_df["label"]):
        idx_by_label[int(lbl)].append(i)
    valid_idx_set = set()
    for lbl, idxs in idx_by_label.items():
        idxs_a = np.asarray(idxs)
        rng.shuffle(idxs_a)
        n_v = int(len(idxs_a) * VALID_FRAC)
        valid_idx_set.update(idxs_a[:n_v].tolist())
    train_mask = np.asarray([i not in valid_idx_set for i in range(len(train_df))])
    valid_mask = ~train_mask
    train_split = train_df[train_mask].reset_index(drop=True)
    valid_split = train_df[valid_mask].reset_index(drop=True)
    print(f"  train: {len(train_split)}  valid: {len(valid_split)}", flush=True)

    write_jsonl(to_rows(train_split, "dc"), OUT / "train.jsonl")
    write_jsonl(to_rows(valid_split, "dv"), OUT / "valid.jsonl")
    print(f"  wrote {OUT / 'train.jsonl'} + {OUT / 'valid.jsonl'}", flush=True)

    print("\n[test] reading droid_py_test.parquet …", flush=True)
    test_df = pd.read_parquet(C.DATA / "droid_py_test.parquet")
    write_jsonl(to_rows(test_df, "dt"), OUT / "test.jsonl")
    print(f"  wrote {OUT / 'test.jsonl'}", flush=True)

    print("\n[test_formatted] reading droid_py_test_black.parquet …", flush=True)
    test_fmt_df = pd.read_parquet(C.DATA / "droid_py_test_black.parquet")
    # Paired with test.jsonl by row order
    assert len(test_fmt_df) == len(test_df), "test/test_formatted row counts differ"
    write_jsonl(to_rows(test_fmt_df, "dt"), OUT / "test_formatted.jsonl")
    print(f"  wrote {OUT / 'test_formatted.jsonl'}", flush=True)

    # Sanity: labels match between test and test_formatted (same rows by construction)
    for r_raw, r_fmt in zip(test_df["label"], test_fmt_df["label"]):
        assert int(r_raw) == int(r_fmt)
    print("\nlabel parity check passed", flush=True)


if __name__ == "__main__":
    main()
