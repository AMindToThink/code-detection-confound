"""Watchdog for a training run: notify EARLY on anomaly, otherwise exit when done.

Polls a metrics.jsonl + a completion sentinel and asserts health thresholds (from the
training-failure-signals review). Exits 0 = healthy completion, 7 = ANOMALY (with reason),
so the launching session is re-invoked promptly either way instead of polling blindly.

Usage:
  python -u scripts/11d_watchdog.py --dir results/train/original
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

POLL = 30          # seconds between checks
MAX_STALL = 360    # alarm if metrics file hasn't advanced this long (stall / dead process)
MIN_AUROC = 0.70   # after MIN_AUROC_STEP, val AUROC below this = stuck (easy task)
MIN_AUROC_STEP = 500
HARD_TIMEOUT = 3 * 3600


def last_lines(p: Path, k: int = 50) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines()[-k:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    metrics, done = d / "metrics.jsonl", d / "final_val_metrics.json"
    t0 = time.time()

    def alarm(reason: str) -> None:
        print(f"ANOMALY: {reason}", flush=True)
        raise SystemExit(7)

    while True:
        time.sleep(POLL)
        if done.exists():
            print(f"OK: training completed, sentinel present. final={done.read_text()[:300]}", flush=True)
            return
        if time.time() - t0 > HARD_TIMEOUT:
            alarm(f"hard timeout {HARD_TIMEOUT}s with no completion sentinel")
        if not metrics.exists():
            if time.time() - t0 > MAX_STALL:
                alarm("no metrics file after MAX_STALL (process never started?)")
            continue
        # Stall = nothing has been written since EITHER watchdog start OR the last write,
        # whichever is later — prevents tripping on a stale file from a previous run.
        last_write = max(t0, metrics.stat().st_mtime)
        if time.time() - last_write > MAX_STALL:
            alarm(f"metrics stalled >{MAX_STALL}s (process likely died/hung)")
        rows = last_lines(metrics)
        losses = [r for r in rows if "loss" in r]
        if losses:
            l = losses[-1]["loss"]
            if not isinstance(l, (int, float)) or l != l or abs(l) == float("inf"):
                alarm(f"non-finite loss at step {losses[-1].get('step')}")
        evals = [r for r in rows if "eval_auroc" in r]
        for e in evals:
            if e.get("step", 0) >= MIN_AUROC_STEP and e["eval_auroc"] < MIN_AUROC:
                alarm(f"val AUROC {e['eval_auroc']} < {MIN_AUROC} at step {e['step']} (stuck)")
        last = rows[-1] if rows else {}
        print(f"[watchdog t={int(time.time()-t0)}s] last={last}", flush=True)


if __name__ == "__main__":
    main()
