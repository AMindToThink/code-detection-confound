"""Is the formatting shortcut a TRAINING-DATA artifact? Measures how black-clean the human
vs machine classes already are in DroidCollection (the training distribution). If the machine
class is systematically closer to black-canonical than the human class, a classifier can
learn 'clean formatting = machine' — a spurious correlation, since real humans also run black.

Outputs results/formatting_confound.json. CPU only (black via uvx, difflib).
"""
from __future__ import annotations

import ast
import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

N = 150
NEAR = 0.98


def parses(c: str) -> bool:
    try:
        ast.parse(c); return True
    except Exception:
        return False


def black_all(codes: list[str]) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        for i, c in enumerate(codes):
            (dp / f"{i}.py").write_text(c)
        subprocess.run(["uvx", "black", "-q", str(dp)], capture_output=True, text=True)
        return [(dp / f"{i}.py").read_text() for i in range(len(codes))]


def measure(codes: list[str]) -> dict:
    P = [c for c in codes if parses(c)]
    bf = black_all(P)
    sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(P, bf)]
    return {"n": len(P), "black_similarity_mean": float(np.mean(sims)),
            "already_black_frac": float(np.mean([s > NEAR for s in sims]))}


def main() -> None:
    from datasets import load_dataset
    hum, mac = [], []
    for r in load_dataset("project-droid/DroidCollection", split="train", streaming=True):
        if r.get("Language") != "Python":
            continue
        if r["Label"] == "HUMAN_GENERATED" and len(hum) < N:
            hum.append(r["Code"])
        elif r["Label"] == "MACHINE_GENERATED" and len(mac) < N:
            mac.append(r["Code"])
        if len(hum) >= N and len(mac) >= N:
            break
    ms = pd.read_parquet(C.HUMAN_PARQUET).code.tolist()[:N]

    out = {
        "near_threshold": NEAR,
        "DroidCollection_human": measure(hum),
        "DroidCollection_machine": measure(mac),
        "MatrixStudio_human": measure(ms),
    }
    (C.RESULTS / "formatting_confound.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
