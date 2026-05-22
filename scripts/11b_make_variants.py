"""Filter -> balance -> blackify, in one parallel pass.

Why: ~6% of DroidCollection-Python rows are NOT valid Python (ast.parse fails). Training
on them is noise -- the model would learn surface signatures of broken syntax. We keep
ONLY rows that are simultaneously ast.parse-OK and black.format_str-OK, so the black
variant is a true normalization (no verbatim fallback rows masquerading as 'formatted').

Input:  data/droid_py_{train,test}.parquet
Output: data/droid_py_train_bal.parquet              (filtered + balanced 60k/class)
        data/droid_py_train_bal_black.parquet        (same rows, code -> black.format_str)
        data/droid_py_test.parquet                   (overwritten: filtered)
        data/droid_py_test_black.parquet             (filtered, black-formatted)

Run: uv run --no-project --with black --with pandas --with pyarrow python -u scripts/11b_make_variants.py
"""
from __future__ import annotations

import ast
import sys
from multiprocessing import Pool
from pathlib import Path

import black
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

N_PER_CLASS = 60_000
SEED = 0
N_PROC = 14
MODE = black.Mode()


def _check_and_fmt(code: str) -> tuple[bool, str | None]:
    """Returns (ast_ok, black_formatted_or_None). ast_ok=False => broken Python (drop).
    black_formatted=None => valid Python but black can't format it (also drop, so the
    black variant is a real normalization, not a verbatim fallback)."""
    try:
        ast.parse(code)
    except Exception:
        return False, None
    try:
        return True, black.format_str(code, mode=MODE)
    except Exception:
        return True, None


def filter_and_format(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (raw_kept, black_kept) DataFrames with identical row order."""
    n = len(df)
    with Pool(N_PROC) as pool:
        res = pool.map(_check_and_fmt, df["code"].tolist(), chunksize=200)
    ast_ok = [r[0] for r in res]
    keep = [(r[0] and r[1] is not None) for r in res]
    n_ast = sum(ast_ok); n_keep = sum(keep)
    print(f"[{name}] n={n}  ast_ok={n_ast} ({n_ast/n:.3f})  "
          f"ast_ok AND black_ok={n_keep} ({n_keep/n:.3f})", flush=True)
    raw = df.loc[keep].reset_index(drop=True)
    black_codes = [res[i][1] for i in range(n) if keep[i]]
    blk = raw.copy(); blk["code"] = black_codes
    return raw, blk


def main() -> None:
    full_train = pd.read_parquet(C.DATA / "droid_py_train.parquet")
    test = pd.read_parquet(C.DATA / "droid_py_test.parquet")
    print(f"input: train={len(full_train)} test={len(test)}", flush=True)

    train_raw, train_blk = filter_and_format(full_train, "train_full")
    test_raw, test_blk = filter_and_format(test, "test")

    # balanced subset (seed-fixed) from FILTERED train -> equal classes of valid Python
    per = min(N_PER_CLASS, int(train_raw.label.value_counts().min()))
    bal_idx = (pd.concat([g.sample(per, random_state=SEED)
                          for _, g in train_raw.groupby("label")])
                 .sample(frac=1, random_state=SEED).index)
    bal_raw = train_raw.loc[bal_idx].reset_index(drop=True)
    bal_blk = train_blk.loc[bal_idx].reset_index(drop=True)
    assert set(bal_raw.columns) == set(bal_blk.columns) >= {"code", "label"}
    print(f"balanced train: {len(bal_raw)} ({per}/class)", flush=True)

    bal_raw.to_parquet(C.DATA / "droid_py_train_bal.parquet")
    bal_blk.to_parquet(C.DATA / "droid_py_train_bal_black.parquet")
    test_raw.to_parquet(C.DATA / "droid_py_test.parquet")
    test_blk.to_parquet(C.DATA / "droid_py_test_black.parquet")
    print("wrote: train_bal, train_bal_black, test, test_black (all filtered)", flush=True)


if __name__ == "__main__":
    main()
