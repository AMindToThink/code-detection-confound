"""Build the human-code table from MatrixStudio/Codeforces-Python-Submissions.

Streams the dataset (no full download), keeps accepted (verdict == OK) Python-3
solutions, buckets problems into EASY / HARD difficulty bands, and samples up to
MAX_HUMAN_PER_PROBLEM distinct solutions for N_PROBLEMS_PER_BAND problems per band.

Outputs:
  data/human_code.parquet   one row per human solution
  data/problems.parquet     one row per sampled problem (statement + test cases)

Fails loudly if a band cannot be filled (no silent under-sampling).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src import config as C

DATASET = "MatrixStudio/Codeforces-Python-Submissions"
MIN_SOLUTIONS_PER_PROBLEM = 3      # need a few humans per problem for stable per-cell stats
MAX_STREAM_ROWS = 600_000          # safety bound on the stream
MIN_CODE_CHARS = 60                # drop trivial / truncated snippets


def normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def main() -> None:
    ds = load_dataset(DATASET, split="train", streaming=True)

    # problem_id -> dict(meta + list of (code))
    problems: dict[str, dict] = {}
    seen_code: dict[str, set[str]] = defaultdict(set)
    n_seen = 0

    def band_full(band: str) -> bool:
        ready = [p for p in problems.values()
                 if p["band"] == band and len(p["codes"]) >= MIN_SOLUTIONS_PER_PROBLEM]
        return len(ready) >= C.N_PROBLEMS_PER_BAND

    for row in ds:
        n_seen += 1
        if n_seen > MAX_STREAM_ROWS:
            break
        if row.get("verdict") != "OK":
            continue
        lang = (row.get("programmingLanguage") or "")
        if "Python" not in lang:
            continue
        band = C.band_of(row.get("rating"))
        if band is None:
            continue
        code = row.get("code") or ""
        if len(code) < MIN_CODE_CHARS:
            continue
        code = normalize(code)
        pid = f"{row.get('contestId')}{row.get('index')}"

        if pid not in problems:
            # only open new problems for a band that still needs them
            if band_full(band):
                continue
            problems[pid] = {
                "problem_id": pid,
                "contestId": row.get("contestId"),
                "index": row.get("index"),
                "band": band,
                "rating": row.get("rating"),
                "tags": json.dumps(row.get("tags")),
                "statement": row.get("prompt") or "",
                "test_cases": json.dumps(row.get("test_cases") or []),
                "codes": [],
            }
        p = problems[pid]
        if len(p["codes"]) >= C.MAX_HUMAN_PER_PROBLEM:
            continue
        if code in seen_code[pid]:
            continue
        seen_code[pid].add(code)
        p["codes"].append(code)

        if band_full("easy") and band_full("hard"):
            break

    # keep only problems meeting the minimum, capped per band
    rows_h, rows_p = [], []
    for band in C.BANDS:
        ready = [p for p in problems.values()
                 if p["band"] == band and len(p["codes"]) >= MIN_SOLUTIONS_PER_PROBLEM]
        ready.sort(key=lambda p: p["problem_id"])
        if len(ready) < C.N_PROBLEMS_PER_BAND:
            raise RuntimeError(
                f"band '{band}': only {len(ready)} problems with >= "
                f"{MIN_SOLUTIONS_PER_PROBLEM} solutions after {n_seen} rows; "
                f"need {C.N_PROBLEMS_PER_BAND}. Lower N_PROBLEMS_PER_BAND or "
                f"MIN_SOLUTIONS_PER_PROBLEM, or raise MAX_STREAM_ROWS."
            )
        for p in ready[: C.N_PROBLEMS_PER_BAND]:
            rows_p.append({k: p[k] for k in
                           ("problem_id", "contestId", "index", "band", "rating",
                            "tags", "statement", "test_cases")})
            for j, code in enumerate(p["codes"]):
                rows_h.append({
                    "sample_id": f"H::{p['problem_id']}::{j}",
                    "problem_id": p["problem_id"],
                    "band": band,
                    "problem_rating": p["rating"],
                    "species": "human",
                    "model": "human",
                    "condition": "human",
                    "code": code,
                    "skill_source": C.SKILL_SOURCE,
                })

    df_h = pd.DataFrame(rows_h)
    df_p = pd.DataFrame(rows_p)
    df_h.to_parquet(C.HUMAN_PARQUET, index=False)
    df_p.to_parquet(C.DATA / "problems.parquet", index=False)

    print(f"streamed {n_seen} rows")
    print(f"human solutions: {len(df_h)} across {df_p.shape[0]} problems")
    print(df_h.groupby('band').agg(problems=('problem_id', 'nunique'),
                                   solutions=('sample_id', 'count')))
    print(f"wrote {C.HUMAN_PARQUET}")
    print(f"wrote {C.DATA / 'problems.parquet'}")


if __name__ == "__main__":
    main()
