"""Render metrics.jsonl -> a single PNG summary plot (loss / val AUROC / grad-norm / sps).

Zero-auth dashboard snapshot. Run anytime during a training run; use `SendUserFile` to
proactively push it to the human's phone.

Usage:
  python scripts/11e_plot.py --dir results/train/original
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    out = Path(a.out) if a.out else d / "dashboard.png"

    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "eval_auroc" in r]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    if train:
        axes[0, 0].plot([r["step"] for r in train], [r["loss"] for r in train])
        axes[0, 0].set_title(f"train loss (last={train[-1]['loss']:.3f})")
        axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("loss"); axes[0, 0].grid(alpha=.3)
        axes[0, 1].plot([r["step"] for r in train], [r.get("grad_norm", float("nan")) for r in train])
        axes[0, 1].set_title("pre-clip grad-norm")
        axes[0, 1].set_xlabel("step"); axes[0, 1].set_yscale("log"); axes[0, 1].grid(alpha=.3)
        axes[1, 1].plot([r["step"] for r in train], [r.get("sps", float("nan")) for r in train])
        axes[1, 1].set_title(f"throughput samples/s (last={train[-1].get('sps','?')})")
        axes[1, 1].set_xlabel("step"); axes[1, 1].grid(alpha=.3)
    if evals:
        axes[1, 0].plot([r["step"] for r in evals], [r["eval_auroc"] for r in evals], "o-")
        axes[1, 0].axhline(0.5, color="gray", lw=.7, ls="--", label="chance")
        last = evals[-1]
        axes[1, 0].set_title(f"val AUROC (last={last['eval_auroc']:.3f} @ step {last['step']})")
        axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylim(0.4, 1.02); axes[1, 0].grid(alpha=.3)
    fig.suptitle(f"{d.name} — {len(train)} train logs, {len(evals)} evals")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
