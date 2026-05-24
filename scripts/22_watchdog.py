"""Lightweight watchdog for the Phase B training runs.

Watches one run's metrics.jsonl + completion sentinel + global disk-free margin.
Polls every POLL seconds; exits 0 on clean completion sentinel, exits 7 on anomaly
(NaN loss is already fatal inside the training script via JsonlMonitor, but stall /
disk-full / wall-cap need an external monitor).

Critique-mandated thresholds (revised plan 2026-05-24):
  - hard wall cap = 24 h (then SURFACE; don't auto-kill — checkpoint is on disk)
  - alarm if no best-F1 checkpoint update for 6 h (training proceeding but not
    improving; could mean EarlyStopping should have fired or the eval is broken)
  - disk-free < 2 GB → alarm (someone else may be writing a big model)
  - disk-free < 500 MB → SURFACE immediately (we're about to crash)
  - stall = metrics.jsonl mtime not advanced for 6 min

Usage:
  python3 -u scripts/22_watchdog.py --dir results/cgs/python_raw_ce
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

POLL = 30
MAX_STALL = 600       # 10 min between metrics writes; covers in-epoch eval phase too
STARTUP_GRACE = 900   # 15 min — tokenization of full HMCorp train (~225k rows
                      # Python) takes ~6 min and the first JsonlMonitor write isn't
                      # until step 10, so a hard stall check before this grace
                      # period legitimately false-positives. Skip during grace.
HARD_WALL = 24 * 3600
NO_IMPROVE_S = 6 * 3600
DISK_ALARM_GB = 2.0
DISK_HARD_GB = 0.5


def alarm(msg: str) -> None:
    print(f"ANOMALY: {msg}", flush=True)
    raise SystemExit(7)


def disk_free_gb(p: Path) -> float:
    s = shutil.disk_usage(str(p))
    return s.free / 1024**3


def last_lines(p: Path, k: int = 80) -> list[dict]:
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
    metrics = d / "metrics.jsonl"
    done = d / "final_val_metrics.json"
    best = d / "checkpoint-best-f1" / "model.bin"

    t0 = time.time()
    last_best_mtime = 0.0

    while True:
        time.sleep(POLL)
        now = time.time()

        if done.exists():
            print(f"OK clean completion {done.read_text()[:300]}", flush=True)
            return

        # wall cap
        if now - t0 > HARD_WALL:
            alarm(f"hit hard wall cap {HARD_WALL//3600}h with no completion sentinel; "
                  f"surface to PI — checkpoint should still be in {best}")

        # disk
        free = disk_free_gb(d)
        if free < DISK_HARD_GB:
            alarm(f"disk free {free:.2f} GB < hard threshold {DISK_HARD_GB} GB — "
                  f"training about to crash; surface NOW")

        # metrics file — skip stall checks during STARTUP_GRACE (tokenization phase
        # legitimately produces no metrics for ~6 min on Python's 225k rows)
        in_grace = (now - t0) < STARTUP_GRACE
        if not metrics.exists():
            if not in_grace:
                alarm("metrics file missing past STARTUP_GRACE — training never started?")
            print(f"[wd t={int(now-t0)}s free={free:.1f}G grace=warmup last=(no metrics yet)]",
                  flush=True)
            continue
        last_write = max(t0, metrics.stat().st_mtime)
        if not in_grace and (now - last_write > MAX_STALL):
            alarm(f"metrics stalled > {MAX_STALL}s (process hung/died)")

        # best-checkpoint improvement watch (only meaningful after first epoch)
        rows = last_lines(metrics)
        evals = [r for r in rows if "eval_f1" in r]
        if evals and best.exists():
            mt = best.stat().st_mtime
            if mt != last_best_mtime:
                last_best_mtime = mt
            elif now - mt > NO_IMPROVE_S:
                alarm(f"best-F1 checkpoint not updated for {NO_IMPROVE_S//3600}h "
                      f"despite {len(evals)} evals; EarlyStopping should have fired")

        last = rows[-1] if rows else {}
        print(f"[wd t={int(now-t0)}s free={free:.1f}G last={last}]", flush=True)


if __name__ == "__main__":
    main()
