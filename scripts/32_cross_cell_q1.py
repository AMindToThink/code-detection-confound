"""Compute the CORRECTED Q1 metric (human-FPR shift) across all available cells
and write a consolidated table.

Background: the Q1 metric in the original plan doc was pre-registered on the
machine class (TPR drop on machine rows after black-formatting). The original
DroidDetect formatting-confound symptom is on the human class (FPR increase on
human rows after black-formatting). The metrics agree directionally for severe
confounds but diverge for moderate ones — Cell 3 (UniXcoder × DC) is the
diverging case: machine TPR drops by +0.2pp (says no confound) while human FPR
rises +21pp (says yes confound).

This script reads every available eval JSON and emits a uniform per-cell table
with BOTH metrics, plus the AUC drop on test_formatted. Numbers go to
results/phase_e/cross_cell_q1.json (durable).

Usage:
  python3 -u scripts/32_cross_cell_q1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


def per_cell(label: str, arch: str, data: str, eval_json: Path) -> dict:
    """eval_json is either eval_phase_b.json (HMCorp Phase B format) or
    eval_q1.json (v2 format). Both share the same per_corpus->{test,test_formatted}
    structure."""
    if not eval_json.exists():
        return {"label": label, "arch": arch, "data": data,
                "available": False, "path": str(eval_json)}
    j = json.loads(eval_json.read_text())
    t = j["per_corpus"]["test"]
    f = j["per_corpus"]["test_formatted"]
    # Human-side metric (FPR shift) — symmetric with the original DroidDetect framing
    fpr_t = t["flag_rate_human"]
    fpr_f = f["flag_rate_human"]
    fpr_shift = fpr_f - fpr_t
    # Machine-side metric (TPR drop) — what we pre-registered (in retrospect, the
    # wrong side for catching the DroidDetect symptom)
    tpr_t = t["flag_rate_machine"]
    tpr_f = f["flag_rate_machine"]
    tpr_drop = tpr_t - tpr_f
    p1_t = t["machine_prob_mean_on_machine_rows"]
    p1_f = f["machine_prob_mean_on_machine_rows"]
    machine_prob_drop = p1_t - p1_f
    return {
        "label": label, "arch": arch, "data": data,
        "available": True,
        "n_test": t["n"], "n_formatted": f["n"],
        "auroc_test": t["auroc"], "auroc_formatted": f["auroc"],
        "auroc_drop": t["auroc"] - f["auroc"],
        "fpr_human_test": fpr_t, "fpr_human_formatted": fpr_f,
        "fpr_human_shift": fpr_shift,
        "tpr_machine_test": tpr_t, "tpr_machine_formatted": tpr_f,
        "tpr_machine_drop": tpr_drop,
        "machine_prob_drop": machine_prob_drop,
        # Pre-registered Q1 (machine-side) AND corrected Q1 (human-side)
        "q1_machine_side": (machine_prob_drop >= 0.10) or (tpr_drop >= 0.10),
        "q1_human_side":   (fpr_shift >= 0.10),
    }


def main() -> None:
    cells = [
        per_cell("Cell1_Py",  "UniXcoder", "HMCorp Python",
                 C.RESULTS / "cgs" / "python_raw_ce" / "eval_phase_b.json"),
        per_cell("Cell1_Ja",  "UniXcoder", "HMCorp Java",
                 C.RESULTS / "cgs" / "java_raw_ce"   / "eval_phase_b.json"),
        per_cell("Cell3",     "UniXcoder", "DroidCollection",
                 C.RESULTS / "cgs" / "unixcoder_dc_ce" / "eval_q1.json"),
    ]

    print(f"{'cell':12s} {'arch':10s} {'data':16s} "
          f"{'AUC_t':>6s} {'AUC_f':>6s} {'ΔAUC':>6s} "
          f"{'FPR_t':>6s} {'FPR_f':>6s} {'ΔFPR':>6s} "
          f"{'TPR_t':>6s} {'TPR_f':>6s} {'q1_M':>5s} {'q1_H':>5s}", flush=True)
    print("-" * 130, flush=True)
    for c in cells:
        if not c.get("available"):
            print(f"  {c['label']:10s} (file not present: {c['path']})", flush=True)
            continue
        print(
            f"{c['label']:12s} {c['arch']:10s} {c['data']:16s} "
            f"{c['auroc_test']:6.3f} {c['auroc_formatted']:6.3f} {c['auroc_drop']:+6.3f} "
            f"{c['fpr_human_test']:6.3f} {c['fpr_human_formatted']:6.3f} {c['fpr_human_shift']:+6.3f} "
            f"{c['tpr_machine_test']:6.3f} {c['tpr_machine_formatted']:6.3f} "
            f"{'YES' if c['q1_machine_side'] else 'NO':>5s} "
            f"{'YES' if c['q1_human_side']  else 'NO':>5s}",
            flush=True,
        )

    out_p = C.RESULTS / "phase_e" / "cross_cell_q1.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"cells": cells}, indent=2))
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
