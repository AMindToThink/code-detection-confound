"""Generate LLM solutions with an OpenAI frontier model (default gpt-5.4-nano) via the
Responses API. Adds a true post-RLHF frontier model to the panel — the most direct test
of the 'senior-engineer attractor' hypothesis. API-only (no GPU).

Reads OPENAI_API_KEY from .env (never printed). Threaded with retries.
Output: data/gen/openai__<model>.parquet
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import config as C

GEN_DIR = C.DATA / "gen"
GEN_DIR.mkdir(exist_ok=True)
MAX_WORKERS = 8
MAX_OUTPUT_TOKENS = 2200      # room for reasoning + code


def load_key() -> str:
    env = pathlib.Path(C.ROOT / ".env")
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r'\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                return m.group(1)
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found in .env or environment")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|python3|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(import|from|def|class|n\s*=|#!|for |while |if __name__)", ln):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--n", type=int, default=C.N_GEN_PER_MODEL_COND)
    args = ap.parse_args()
    model = args.model

    os.environ["OPENAI_API_KEY"] = load_key()
    from openai import OpenAI
    client = OpenAI()

    problems = pd.read_parquet(C.DATA / "problems.parquet")
    jobs = []   # (pid, band, rating, cond, sample_idx, prompt)
    for _, p in problems.iterrows():
        for cond in C.CONDITIONS:
            user = C.PROMPT_TEMPLATES[cond].format(problem=p.statement)
            for j in range(args.n):
                jobs.append((p.problem_id, p.band, p.rating, cond, j, user))

    def call(job):
        pid, band, rating, cond, j, user = job
        for attempt in range(4):
            try:
                r = client.responses.create(model=model, input=user,
                                            max_output_tokens=MAX_OUTPUT_TOKENS,
                                            reasoning={"effort": "low"})
                return job, (r.output_text or "")
            except Exception as e:
                if attempt == 3:
                    return job, f"__ERR__{e}"
                time.sleep(2 * (attempt + 1))
        return job, "__ERR__"

    safe = "openai__" + model.replace("/", "__").replace(".", "_")
    rows, errs = [], 0
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(call, j) for j in jobs]
        for fut in as_completed(futs):
            (pid, band, rating, cond, j, _), text = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} calls done", flush=True)
            if text.startswith("__ERR__"):
                errs += 1
                continue
            code = extract_code(text)
            if len(code) < 30:
                continue
            rows.append({
                "sample_id": f"L::{safe}::{pid}::{cond}::{j}",
                "problem_id": pid, "band": band, "problem_rating": rating,
                "species": "llm", "model": model, "condition": cond,
                "code": code, "skill_source": C.SKILL_SOURCE,
            })
    df = pd.DataFrame(rows)
    out = GEN_DIR / f"{safe}.parquet"
    df.to_parquet(out, index=False)
    print(f"{model}: {len(df)} generations ({errs} errors) -> {out}")
    print(df.groupby(['band', 'condition']).size())


if __name__ == "__main__":
    main()
