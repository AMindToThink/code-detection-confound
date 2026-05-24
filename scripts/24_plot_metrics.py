"""Render a PNG snapshot of one or more training runs' metrics.jsonl files.

Reads the JSONL telemetry produced by scripts/18_train_cgs_amp.py and produces a
3-row × N-col matplotlib figure: per-step loss curve, per-epoch val_auroc/f1, per-
epoch val_loss with ES-counter annotation. Phone-friendly proportions.

Usage:
  python3 -u scripts/24_plot_metrics.py results/cgs/python_raw_ce results/cgs/java_raw_ce \\
      --out results/cgs/snapshot.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir: Path) -> tuple[list[dict], list[dict]]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return [], []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    step_rows = [r for r in rows if "loss" in r and "eval_f1" not in r]
    eval_rows = [r for r in rows if "eval_f1" in r]
    return step_rows, eval_rows


def es_counter(eval_rows: list[dict]) -> list[int]:
    """Replay their EarlyStopping(patience=5) on val_loss to compute counter at each
    epoch. counter resets on new best."""
    best = float("inf")
    out = []
    cnt = 0
    for r in eval_rows:
        if r["eval_loss"] < best:
            best = r["eval_loss"]
            cnt = 0
        else:
            cnt += 1
        out.append(cnt)
    return out


def plot_run(axes, run_dir: Path, col: int, n_cols: int):
    name = run_dir.name
    step_rows, eval_rows = load(run_dir)
    if not step_rows and not eval_rows:
        for r in range(3):
            axes[r][col].set_title(f"{name}\n(no metrics yet)")
            axes[r][col].axis("off")
        return
    # Top: loss curve (training)
    ax = axes[0][col]
    if step_rows:
        s = np.asarray([r["step"] for r in step_rows])
        ll = np.asarray([r["loss"] for r in step_rows])
        ax.plot(s, ll, lw=0.7, color="C0", alpha=0.5, label="raw")
        # Rolling mean for clarity
        if len(ll) > 50:
            w = min(200, len(ll) // 10)
            kernel = np.ones(w) / w
            smooth = np.convolve(ll, kernel, mode="valid")
            ax.plot(s[w - 1:], smooth, lw=1.5, color="C0", label=f"mean({w} steps)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("train loss")
    ax.set_title(f"{name}\nsteps={step_rows[-1]['step'] if step_rows else 0}, "
                 f"elapsed={int(step_rows[-1]['t']) if step_rows else 0}s")

    # Middle: per-epoch val metrics
    ax = axes[1][col]
    if eval_rows:
        ep = [r["epoch"] for r in eval_rows]
        ax.plot(ep, [r["eval_auroc"] for r in eval_rows], "o-", color="C2",
                label="AUROC", markersize=6)
        ax.plot(ep, [r["eval_f1"] for r in eval_rows], "s-", color="C3",
                label="F1", markersize=6)
        ax.set_ylim(0.9, 1.0)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_ylabel("val metric")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)

    # Bottom: per-epoch val_loss + ES counter
    ax = axes[2][col]
    if eval_rows:
        ep = [r["epoch"] for r in eval_rows]
        vl = [r["eval_loss"] for r in eval_rows]
        ax.plot(ep, vl, "o-", color="C1", markersize=6, label="val_loss")
        cnts = es_counter(eval_rows)
        for x, y, c in zip(ep, vl, cnts):
            ax.annotate(f"ES:{c}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8,
                        color="red" if c >= 4 else "black")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("val_loss")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="run directories containing metrics.jsonl")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = len(a.dirs)
    fig, axes = plt.subplots(3, n, figsize=(5.5 * n, 9), squeeze=False)
    for i, d in enumerate(a.dirs):
        plot_run(axes, Path(d), i, n)
    fig.suptitle(f"CodeGPTSensor CE-only training (Phase B) — snapshot",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
