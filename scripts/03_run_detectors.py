"""Assemble all samples (human + LLM) and score them with every detector + style features.

Inputs : data/human_code.parquet, data/gen/*.parquet
Output : data/scored_samples.parquet  (one row per sample, + detector scores + features)

Detector scores are oriented HIGHER == MORE AI for all detectors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401
from src import config as C
from src.features import style_features

import pandas as pd
from tqdm import tqdm


def assemble() -> pd.DataFrame:
    human = pd.read_parquet(C.HUMAN_PARQUET)
    gen_files = sorted((C.DATA / "gen").glob("*.parquet"))
    if not gen_files:
        raise RuntimeError("no LLM generations found in data/gen/ — run 02 first")
    llm = pd.concat([pd.read_parquet(f) for f in gen_files], ignore_index=True)
    df = pd.concat([human, llm], ignore_index=True)
    df = df.drop_duplicates("sample_id").reset_index(drop=True)
    return df


def main() -> None:
    df = assemble()
    print(f"assembled {len(df)} samples "
          f"({(df.species=='human').sum()} human, {(df.species=='llm').sum()} llm)")

    # style features
    feats = pd.DataFrame([style_features(c) for c in df["code"]])
    df = pd.concat([df.reset_index(drop=True), feats.add_prefix("feat_")], axis=1)

    from src.detectors import DroidDetect, StatDetectors

    codes = df["code"].tolist()
    print("scoring DroidDetect ...")
    dd = DroidDetect()
    df["DroidDetect"] = dd.score_batch(codes, bs=32)
    del dd
    import torch
    torch.cuda.empty_cache()

    print("scoring statistical detectors ...")
    sd = StatDetectors()
    fd, bino = [], []
    for code in tqdm(codes):
        r = sd.score_one(code)
        fd.append(r["FastDetectGPT"])
        bino.append(r["Binoculars"])
    df["FastDetectGPT"] = fd
    df["Binoculars"] = bino

    df.to_parquet(C.SCORED_PARQUET, index=False)
    print(f"wrote {C.SCORED_PARQUET}")
    # quick global sanity: pooled AUROC should be > 0.5 for each detector (orientation)
    from sklearn.metrics import roc_auc_score
    y = (df.species == "llm").astype(int)
    for det in C.DETECTOR_FAMILY:
        sub = df[df[det].notna()]
        auc = roc_auc_score((sub.species == "llm").astype(int), sub[det])
        print(f"  pooled AUROC {det:14s} = {auc:.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
