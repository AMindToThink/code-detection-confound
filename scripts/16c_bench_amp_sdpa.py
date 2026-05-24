"""Bench CodeGPTSensor train step under four configurations:

  (1) FP32 + eager attention   (baseline)
  (2) FP32 + SDPA attention    (mem-eff kernel on Turing)
  (3) AMP-FP16 + eager attention
  (4) AMP-FP16 + SDPA attention   (the stack we're considering)

Reports steps/s, samples/s, peak mem for each. We need to know the actual multipliers on
THIS hardware (Quadro RTX 8000, sm_75) before locking the training plan — Agent 1's
literature numbers (FP16 ~2x, SDPA ~1.15x) are guesses.

Numerical-equivalent AMP wrapper: autocast around encoder forward; cast KL/cosine inputs
to FP32 (as recommended for R-drop FP16 stability). Uses GradScaler.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/16c_bench_amp_sdpa.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401

CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer  # noqa: E402

BATCH = 8
SEQ = 400
WARMUP = 8
TIMED = 40
LR = 2e-5


class _Args:
    contrast = True
    do_train = True
    device = "cuda"


class ModelAMPSafe(nn.Module):
    """Same math as their model.py Model class, with KL+cosine forced to FP32 under AMP."""

    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)

    def get_xcode_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.encoder(
            source_ids,
            attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2),
            output_hidden_states=True,
        )
        tok_emb = out[0]
        sent_emb = (tok_emb * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return sent_emb

    def forward(self, input_ids, contrast_ids, labels):
        vec = self.get_xcode_vec(input_ids)
        logits = self.classifier(vec)
        loss = F.cross_entropy(logits, labels)
        if self.args.contrast and self.args.do_train:
            vec2 = self.get_xcode_vec(input_ids)
            logits2 = self.classifier(vec2)
            # KL: cast logits to FP32 to avoid FP16 instability under R-drop
            kl_loss = _kl_loss(logits.float(), logits2.float())
            vec3 = self.get_xcode_vec(contrast_ids)
            # cosine: cast to FP32 (small embeddings can underflow in FP16)
            cos = F.cosine_embedding_loss(
                vec.float(), vec3.float(),
                torch.full(labels.size(), -1, device=labels.device),
            )
            loss = loss + 0.1 * kl_loss + 0.2 * cos
        return loss


def _kl_loss(p, q):
    pl = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none").sum()
    ql = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none").sum()
    return (pl + ql) / 2


def make_model(attn_impl: str):
    base = "microsoft/unixcoder-base-nine"
    cfg = RobertaConfig.from_pretrained(base)
    kw = {"attn_implementation": attn_impl} if attn_impl != "eager" else {}
    enc = RobertaModel.from_pretrained(base, **kw)
    return ModelAMPSafe(enc, cfg, _Args()).cuda(), cfg.vocab_size


def bench(model, vocab, amp: bool):
    g = torch.Generator(device="cpu").manual_seed(0)
    inp = torch.randint(0, vocab, (BATCH, SEQ), generator=g).cuda()
    con = torch.randint(0, vocab, (BATCH, SEQ), generator=g).cuda()
    lab = torch.ones(BATCH, dtype=torch.long).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def step():
        if amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(inp, con, lab)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss = model(inp, con, lab)
            loss.backward()
            opt.step()
        opt.zero_grad()
        return loss

    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    last_loss = 0.0
    for _ in range(TIMED):
        last_loss = float(step().detach().item())
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = TIMED / dt
    return sps, sps * BATCH, torch.cuda.max_memory_allocated() / 1024**3, last_loss


def main() -> None:
    assert torch.cuda.is_available()
    print(f"[device] {torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    configs = [
        ("FP32 + eager", "eager", False),
        ("FP32 + sdpa",  "sdpa",  False),
        ("FP16 + eager", "eager", True),
        ("FP16 + sdpa",  "sdpa",  True),
    ]
    baseline_sps = None
    for name, attn, amp in configs:
        try:
            model, vocab = make_model(attn)
        except Exception as e:
            print(f"[{name}] model build failed: {type(e).__name__}: {e}", flush=True)
            continue
        try:
            sps, spr, gb, loss = bench(model, vocab, amp)
        except Exception as e:
            print(f"[{name}] bench failed: {type(e).__name__}: {e}", flush=True)
            del model; torch.cuda.empty_cache()
            continue
        if baseline_sps is None:
            baseline_sps = sps
        mult = sps / baseline_sps
        print(f"[{name}] {sps:.2f} steps/s ({spr:.1f} samples/s), peak {gb:.1f} GB, "
              f"x{mult:.2f}, last_loss={loss:.4f}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
