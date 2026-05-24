"""Bench CodeGPTSensor's training step on the target GPU.

Loads UniXcoder-base-nine + their `Model` class, runs N warmup + M timed train steps
at the real config (batch 8, seq 400, --contrast = 3 forward passes per step), reports
steps/sec, and projects full-run wall time for Python (225,693 rows) and Java (151,756
rows) at 20 epochs (cap) and 5 epochs (likely early-stop point).

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run --no-project python -u scripts/16_bench_cgs.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401  torchvision shim

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer  # noqa: E402
from model import Model  # noqa: E402  their model module

BATCH = 8
SEQ = 400
WARMUP = 10
TIMED = 50
LR = 2e-5


class _Args:
    contrast = True
    do_train = True
    device = "cuda"


def main() -> None:
    assert torch.cuda.is_available(), "need CUDA"
    name = torch.cuda.get_device_name(0)
    print(f"[bench] device: {name}", flush=True)

    base = "microsoft/unixcoder-base-nine"
    tok = RobertaTokenizer.from_pretrained(base)
    cfg = RobertaConfig.from_pretrained(base)
    enc = RobertaModel.from_pretrained(base)
    model = Model(enc, cfg, tok, _Args()).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # Fake batch: random token ids inside vocab, all-ones labels (real values don't
    # matter for throughput — only shape & dtype).
    vocab = cfg.vocab_size
    g = torch.Generator(device="cpu").manual_seed(0)
    inp = torch.randint(0, vocab, (BATCH, SEQ), generator=g).cuda()
    con = torch.randint(0, vocab, (BATCH, SEQ), generator=g).cuda()
    lab = torch.ones(BATCH, dtype=torch.long).cuda()

    # Warmup
    for _ in range(WARMUP):
        loss, _ = model(inp, con, lab)
        loss.backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(TIMED):
        loss, _ = model(inp, con, lab)
        loss.backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = TIMED / dt
    spr = sps * BATCH  # samples/sec
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[bench] {TIMED} steps in {dt:.2f}s -> {sps:.2f} steps/s ({spr:.1f} samples/s), peak mem {peak_gb:.1f} GB", flush=True)

    for lang, n_rows in [("python", 225_693), ("java", 151_756)]:
        steps_per_epoch = (n_rows + BATCH - 1) // BATCH
        for epochs in (5, 10, 20):
            tot_steps = steps_per_epoch * epochs
            hrs = tot_steps / sps / 3600
            print(f"[proj] {lang} {epochs}ep: {tot_steps:,} steps -> {hrs:.1f} h on {name}", flush=True)


if __name__ == "__main__":
    main()
