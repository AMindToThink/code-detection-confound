"""Build the balanced training subset and the black-formatted variants for Experiment A.

From data/droid_py_{train,test}.parquet:
  - balance train to N per class (seed-fixed)            -> droid_py_train_bal.parquet
  - black-format the balanced train and the test set     -> *_black.parquet

black is applied IN-PROCESS (black.format_str) for speed over ~150k files. Files black
cannot parse are kept verbatim and flagged (parse_ok=False); we report the rate. The black
variants replace `code` with the formatted source so downstream tokenization is identical.

Run: uv run --no-project --with black --with pandas --with pyarrow python scripts/11b_make_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import black
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

N_PER_CLASS = 60_000
SEED = 0
MODE = black.Mode()  # default 88-col


def blackify_series(codes: pd.Series) -> tuple[pd.Series, float]:
    out, ok = [], 0
    for c in codes:
        try:
            out.append(black.format_str(c, mode=MODE)); ok += 1
        except Exception:
            out.append(c)  # keep verbatim if black can't parse it
    return pd.Series(out, index=codes.index), ok / max(len(codes), 1)


def main() -> None:
    train = pd.read_parquet(C.DATA / "droid_py_train.parquet")
    test = pd.read_parquet(C.DATA / "droid_py_test.parquet")

    # balanced train subset (seed-fixed, equal classes). NB: groupby().apply() drops the
    # grouping column in pandas 2.x, so sample per class explicitly to keep `label`.
    per = min(N_PER_CLASS, int(train.label.value_counts().min()))
    parts = [g.sample(per, random_state=SEED) for _, g in train.groupby("label")]
    bal = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    assert set(bal.columns) >= {"code", "label"}, bal.columns
    bal.to_parquet(C.DATA / "droid_py_train_bal.parquet")
    print(f"balanced train: {len(bal)} ({per}/class)", flush=True)

    for name, df in [("train_bal", bal), ("test", test)]:
        coded, rate = blackify_series(df["code"])
        bdf = df.copy(); bdf["code"] = coded
        bdf.to_parquet(C.DATA / f"droid_py_{name}_black.parquet")
        print(f"black {name}: {len(bdf)} rows, black-parseable {rate:.3f} "
              f"-> droid_py_{name}_black.parquet", flush=True)


if __name__ == "__main__":
    main()
