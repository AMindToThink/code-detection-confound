"""Generate LLM solutions for the sampled problems under natural/novice/expert prompts.

Runs ONE model per process (subprocess isolation frees GPU memory cleanly between
models). For each problem x condition, draws N_GEN_PER_MODEL_COND samples. Extracts
code from the model's markdown, writes one row per generation to data/gen/<model>.parquet.

Uses HuggingFace transformers batched generation (the env's vLLM wheel has an ABI
mismatch with torch 2.7.1: vllm._C undefined symbol _ZN3c104cuda9SetDeviceEab).

Usage: python scripts/02_generate_llm.py --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import _env  # noqa: F401
from src import config as C

import pandas as pd

GEN_DIR = C.DATA / "gen"
GEN_DIR.mkdir(exist_ok=True)
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 420
BATCH = 16


def extract_code(text: str) -> str:
    """Pull the code out of a chat reply. Prefer fenced blocks; else strip prose-ish lines."""
    blocks = re.findall(r"```(?:python|python3|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()      # the substantive block
    # no fences: drop obvious prose lines, keep from first code-looking line
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(import|from|def|class|n\s*=|#!|for |while |if __name__)", ln):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    model = args.model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(C.GEN_SEED)
    problems = pd.read_parquet(C.DATA / "problems.parquet")
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float16, trust_remote_code=True, device_map={"": 0})
    lm.eval()

    # flat prompt list (problem x condition); we expand n samples per prompt manually
    meta, prompts = [], []
    for _, p in problems.iterrows():
        for cond in C.CONDITIONS:
            user = C.PROMPT_TEMPLATES[cond].format(problem=p.statement)
            text = tok.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            meta.append((p.problem_id, p.band, p.rating, cond))

    safe = model.replace("/", "__")
    rows = []

    @torch.no_grad()
    def gen_batch(batch_prompts: list[str]) -> list[str]:
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_PROMPT_TOKENS).to(lm.device)
        out = lm.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95,
                          max_new_tokens=MAX_NEW_TOKENS,
                          num_return_sequences=C.N_GEN_PER_MODEL_COND,
                          pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]                 # strip the prompt
        return tok.batch_decode(gen, skip_special_tokens=True)

    for i in range(0, len(prompts), BATCH):
        chunk = prompts[i:i + BATCH]
        decoded = gen_batch(chunk)                               # len = BATCH * n
        for bi, (pid, band, rating, cond) in enumerate(meta[i:i + BATCH]):
            for j in range(C.N_GEN_PER_MODEL_COND):
                text = decoded[bi * C.N_GEN_PER_MODEL_COND + j]
                code = extract_code(text)
                if len(code) < 30:
                    continue
                rows.append({
                    "sample_id": f"L::{safe}::{pid}::{cond}::{j}",
                    "problem_id": pid, "band": band, "problem_rating": rating,
                    "species": "llm", "model": model, "condition": cond,
                    "code": code, "skill_source": C.SKILL_SOURCE,
                })
        print(f"  {min(i + BATCH, len(prompts))}/{len(prompts)} prompts done", flush=True)

    df = pd.DataFrame(rows)
    out_path = GEN_DIR / f"{safe}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{model}: {len(df)} generations -> {out_path}")
    print(df.groupby(['band', 'condition']).size())


if __name__ == "__main__":
    main()
