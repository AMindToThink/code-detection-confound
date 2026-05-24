"""Wandb sidecar: tail one metrics.jsonl file and log to a wandb run.

Reads all existing lines on startup (backfill) then tails for new lines, logging
each to wandb with the row's own "step". Per-step rows (loss/lr/grad_norm/sps) and
per-epoch rows (eval_*) are logged with distinct prefixes so the dashboard groups
sensibly. Exits cleanly when the training process is gone AND no new lines have
appeared for `--idle-exit-s` seconds.

Bridges WANDB_KEY → WANDB_API_KEY (the .env on this machine uses the former; wandb
expects the latter). Uses `resume="allow"` + a deterministic `id` so re-running this
script attaches to the same wandb run instead of starting a new one.

Usage:
  python3 -u scripts/25_wandb_tailer.py \\
      --run-name python_raw_ce \\
      --jsonl results/cgs/python_raw_ce/metrics.jsonl \\
      --project code-detection-confound \\
      --entity matthewkhoriaty-northwestern-university
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import wandb


def deterministic_id(run_name: str) -> str:
    """Stable run-id so we can resume the same wandb run across restarts."""
    return hashlib.sha1(run_name.encode()).hexdigest()[:8]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--entity", default=None)
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--idle-exit-s", type=float, default=600.0,
                   help="exit if no new line and metrics file mtime not advanced for this long")
    a = p.parse_args()

    # Bridge env var name (.env uses WANDB_KEY; wandb wants WANDB_API_KEY)
    if "WANDB_API_KEY" not in os.environ and "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    if "WANDB_API_KEY" not in os.environ:
        raise SystemExit("WANDB_API_KEY / WANDB_KEY missing from env")

    jsonl = Path(a.jsonl)
    print(f"[tailer] run={a.run_name} jsonl={jsonl}", flush=True)

    run = wandb.init(
        project=a.project, entity=a.entity, name=a.run_name,
        id=deterministic_id(a.run_name), resume="allow",
        config={"jsonl_path": str(jsonl)},
    )
    print(f"[tailer] wandb url: {run.url}", flush=True)

    pos = 0  # byte offset into jsonl
    last_change = time.time()
    last_step_logged = -1

    def consume_new_lines() -> int:
        nonlocal pos, last_step_logged
        if not jsonl.exists():
            return 0
        size = jsonl.stat().st_size
        if size < pos:  # file truncated (rare; training restart)
            pos = 0
        if size == pos:
            return 0
        with jsonl.open("rb") as f:
            f.seek(pos)
            chunk = f.read(size - pos).decode("utf-8", errors="replace")
            pos = size
        n = 0
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # Wandb requires monotonic step. The JsonlMonitor writes step rows in
            # monotonic order and eval rows ALSO carry a "step" (cum_step at eval
            # time). Skip out-of-order rows defensively.
            step = int(r.get("step", 0))
            if step <= last_step_logged:
                # Allow equal step for eval-vs-final-step-log collisions: shift by 1
                step = last_step_logged + 1
            d: dict[str, float] = {}
            if "eval_f1" in r:
                # per-epoch eval row
                d["val/epoch"] = float(r.get("epoch", 0))
                d["val/loss"] = float(r["eval_loss"])
                d["val/acc"] = float(r["eval_acc"])
                d["val/f1"] = float(r["eval_f1"])
                d["val/auroc"] = float(r["eval_auroc"])
                d["val/elapsed_s"] = float(r["t"])
            else:
                # per-10-step train row
                d["train/loss"] = float(r["loss"])
                if r.get("lr") is not None:
                    d["train/lr"] = float(r["lr"])
                if r.get("grad_norm") is not None:
                    d["train/grad_norm"] = float(r["grad_norm"])
                if r.get("sps") is not None:
                    d["train/samples_per_sec"] = float(r["sps"])
                d["train/elapsed_s"] = float(r["t"])
            wandb.log(d, step=step)
            last_step_logged = step
            n += 1
        return n

    # Main loop
    while True:
        added = consume_new_lines()
        now = time.time()
        if added:
            last_change = now
            print(f"[tailer] {a.run_name}: +{added} lines (last_step={last_step_logged})",
                  flush=True)
        else:
            # Check idle exit condition
            mtime = jsonl.stat().st_mtime if jsonl.exists() else 0
            since_mtime = now - max(mtime, last_change)
            if since_mtime > a.idle_exit_s:
                print(f"[tailer] {a.run_name}: idle for {int(since_mtime)}s "
                      f"(>{int(a.idle_exit_s)}s); finishing wandb run", flush=True)
                wandb.finish()
                return
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
