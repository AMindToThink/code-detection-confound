"""Two empirical measurements to inform the speedup decision:

  (a) Batch-size scaling sweep at the published seq_len=400 with --contrast (3 fwd).
      Reports steps/s, samples/s, and peak mem at batch 8/16/32/64.
  (b) Tokenized-length distribution of HMCorp Python + Java train rows under the
      UniXcoder tokenizer (1000 sampled rows per language).

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/16b_bench_and_lengths.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer  # noqa: E402
from model import Model  # noqa: E402


SEQ = 400
WARMUP = 5
TIMED = 30
LR = 2e-5


class _Args:
    contrast = True
    do_train = True
    device = "cuda"


def bench(batch: int, model: Model, vocab: int) -> tuple[float, float, float]:
    g = torch.Generator(device="cpu").manual_seed(0)
    inp = torch.randint(0, vocab, (batch, SEQ), generator=g).cuda()
    con = torch.randint(0, vocab, (batch, SEQ), generator=g).cuda()
    lab = torch.ones(batch, dtype=torch.long).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        loss, _ = model(inp, con, lab); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(TIMED):
        loss, _ = model(inp, con, lab); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = TIMED / dt
    return sps, sps * batch, torch.cuda.max_memory_allocated() / 1024**3


def lengths(tok, path: Path, n: int = 1000) -> np.ndarray:
    rng = random.Random(0)
    all_lines = path.read_text().splitlines()
    sample = rng.sample(all_lines, min(n, len(all_lines)))
    lens = []
    for line in sample:
        r = json.loads(line)
        for fld in ("code", "contrast"):
            # Mirror their convert_examples_to_features whitespace-collapse on line 69-76
            txt = " ".join(r[fld].split())
            lens.append(len(tok.tokenize(txt)))
    return np.array(lens)


def main() -> None:
    assert torch.cuda.is_available()
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

    base = "microsoft/unixcoder-base-nine"
    tok = RobertaTokenizer.from_pretrained(base)
    cfg = RobertaConfig.from_pretrained(base)
    enc = RobertaModel.from_pretrained(base)
    model = Model(enc, cfg, tok, _Args()).cuda()

    # --- (a) batch-size sweep ---
    for b in (8, 16, 32, 64):
        try:
            sps, spr, gb = bench(b, model, cfg.vocab_size)
            print(f"[bench batch={b}] {sps:.2f} steps/s ({spr:.1f} samples/s), peak {gb:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[bench batch={b}] OOM", flush=True)
            torch.cuda.empty_cache()
            break

    # --- (b) length distribution ---
    for lang in ("python", "java"):
        p = ROOT / "data" / "hmcorp" / lang / "train.jsonl"
        lens = lengths(tok, p, n=1000)
        pcts = np.percentile(lens, [50, 75, 90, 95, 99])
        frac_le = {n: float((lens <= n).mean()) for n in (128, 200, 256, 320, 400)}
        print(
            f"[lengths {lang}] n={len(lens)}, median={int(pcts[0])}, p75={int(pcts[1])}, "
            f"p90={int(pcts[2])}, p95={int(pcts[3])}, p99={int(pcts[4])}",
            flush=True,
        )
        print(f"[lengths {lang}] frac<=N: {frac_le}", flush=True)


if __name__ == "__main__":
    main()
