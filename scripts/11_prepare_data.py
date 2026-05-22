"""Prepare the DroidCollection Python subset for our own training (Experiment A).

Downloads the released parquet shards (small, ~0.65 GB total), filters to Python,
keeps the clean binary task HUMAN_GENERATED vs MACHINE_GENERATED (the same two classes
our ablation measured), and writes train/test parquets using DroidCollection's OWN
train/test split (no leakage, comparable to the published detector).

Output:
  data/droid_py_train.parquet  (columns: code, label)   label 0=human, 1=machine
  data/droid_py_test.parquet
Fail-loud: asserts both classes present, non-empty code, known label strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

REPO = "project-droid/DroidCollection"
TRAIN_SHARDS = [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]
TEST_SHARD = "data/test-00000-of-00001.parquet"
HUMAN, MACHINE = "HUMAN_GENERATED", "MACHINE_GENERATED"  # the clean binary task


def load_python(files: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        local = hf_hub_download(REPO, f, repo_type="dataset")
        t = pq.read_table(local, columns=["Code", "Label", "Language"])
        df = t.to_pandas()
        df = df[df["Language"] == "Python"]
        parts.append(df[["Code", "Label"]])
        print(f"  {f}: {len(df)} Python rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def to_binary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"[{name}] Python label distribution:\n{df['Label'].value_counts()}", flush=True)
    df = df[df["Label"].isin([HUMAN, MACHINE])].copy()
    df["label"] = (df["Label"] == MACHINE).astype(int)  # 0=human, 1=machine
    df["code"] = df["Code"].astype(str)
    df = df[df["code"].str.len() >= 20].reset_index(drop=True)
    out = df[["code", "label"]]
    assert out["label"].nunique() == 2, f"{name}: need both classes, got {out['label'].unique()}"
    print(f"[{name}] kept {len(out)} rows | human={int((out.label==0).sum())} "
          f"machine={int((out.label==1).sum())}", flush=True)
    return out


def main() -> None:
    C.DATA.mkdir(parents=True, exist_ok=True)
    print("downloading + filtering TRAIN shards...", flush=True)
    train = to_binary(load_python(TRAIN_SHARDS), "train")
    print("downloading + filtering TEST shard...", flush=True)
    test = to_binary(load_python([TEST_SHARD]), "test")
    train.to_parquet(C.DATA / "droid_py_train.parquet")
    test.to_parquet(C.DATA / "droid_py_test.parquet")
    print(f"\nwrote data/droid_py_train.parquet ({len(train)}) and "
          f"data/droid_py_test.parquet ({len(test)})", flush=True)


if __name__ == "__main__":
    main()
