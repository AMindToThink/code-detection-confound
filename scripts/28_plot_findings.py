"""Render a summary PNG of Phase B + D + WHY-audit findings for the paper appendix
and for at-a-glance status messages.

Reads:
  results/cgs/python_raw_ce/eval_phase_b.json
  results/cgs/java_raw_ce/eval_phase_b.json
  results/cgs/python_raw_ce/eval_phase_d.json
  results/phase_e/compliance_audit.json

Writes:
  results/cgs/findings_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    pb_py = json.loads((ROOT / "results/cgs/python_raw_ce/eval_phase_b.json").read_text())
    pb_ja = json.loads((ROOT / "results/cgs/java_raw_ce/eval_phase_b.json").read_text())
    pd = json.loads((ROOT / "results/cgs/python_raw_ce/eval_phase_d.json").read_text())
    comp = json.loads((ROOT / "results/phase_e/compliance_audit.json").read_text())
    cell3_path = ROOT / "results/cgs/unixcoder_dc_ce/eval_q1.json"
    cell3 = json.loads(cell3_path.read_text()) if cell3_path.exists() else None
    ccq_path = ROOT / "results/phase_e/cross_cell_q1.json"
    ccq = json.loads(ccq_path.read_text()) if ccq_path.exists() else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (0,0) Headline: human-FPR shift after black-formatting across all cells
    # This is the CORRECTED metric (matches DroidDetect's original framing).
    ax = axes[0][0]
    cell_labels = [
        ("UniXcoder ×\nHMCorp Py",  0.011, 0.010),   # Cell 1 Py
        ("UniXcoder ×\nHMCorp Ja",  0.026, 0.031),   # Cell 1 Ja
        ("UniXcoder ×\nDroidColl",  0.069, 0.279),   # Cell 3 (NEW)
        ("ModernBERT ×\nDroidColl\n(DroidDetect)", 0.01, 0.83),  # Cell 4 (ref)
    ]
    pos = np.arange(len(cell_labels))
    raw = [c[1] for c in cell_labels]
    fmt = [c[2] for c in cell_labels]
    ax.bar(pos - 0.2, raw, 0.4, label="test (raw)",         color="C0")
    ax.bar(pos + 0.2, fmt, 0.4, label="test_formatted (black)", color="C1")
    for i, (_, r, f) in enumerate(cell_labels):
        ax.annotate(f"Δ={f-r:+.2f}", (i, max(f, r) + 0.04),
                    ha="center", fontsize=9, fontweight="bold",
                    color="red" if (f - r) >= 0.10 else "black")
    ax.set_xticks(pos)
    ax.set_xticklabels([c[0] for c in cell_labels], fontsize=8.5)
    ax.set_ylabel("human-flag-rate (FPR on humans)")
    ax.set_ylim(0, 1.0)
    ax.set_title("2×2 arch×data: human-FPR shift under black\n"
                 "(formatting-confound = data + arch interaction)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Phase D flag rates across corpora
    ax = axes[0][1]
    corpora = [
        ("HMCorp test\n(in-dist)",        pd["per_corpus"]["hmcorp_python_test"]["flag_rate_human_at_0.5"]),
        ("MatrixStudio\nhumans",          pd["per_corpus"]["matrixstudio_humans"]["flag_rate_human_at_0.5"]),
        ("Legendary\nhumans",             pd["per_corpus"]["legendary_humans"]["flag_rate_human_at_0.5"]),
        ("DroidCollection\ntest (mixed)", pd["per_corpus"]["droidcollection_test"]["flag_rate_human_at_0.5"]),
    ]
    names = [c[0] for c in corpora]
    cgs_vals = [c[1] for c in corpora]
    # Reference DroidDetect numbers (from project memory)
    dd_ref = {"MatrixStudio\nhumans": 0.620, "Legendary\nhumans": 0.785}
    pos = np.arange(len(names))
    ax.bar(pos - 0.2, cgs_vals, 0.4, label="CodeGPTSensor (this work)", color="C2")
    dd_vals = [dd_ref.get(n, np.nan) for n in names]
    ax.bar(pos + 0.2, dd_vals, 0.4, label="DroidDetect-Base-Binary (memory ref)",
           color="C3", alpha=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("human-flag-rate (FPR-on-humans)")
    ax.set_ylim(0, 1)
    ax.set_title("Phase D OOD: both detectors fail OOD, in different directions")
    ax.axhline(0.05, ls="--", color="gray", lw=0.7, alpha=0.7)
    ax.text(0.0, 0.06, "calibrated 5% FPR target", fontsize=8, color="gray")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0) WHY audit: training-data compliance gap
    ax = axes[1][0]
    rows = [
        ("HMCorp\nPython", comp["hmcorp_python"]),
        ("DroidCollection\nPython", comp["droid_python"]),
    ]
    pos = np.arange(len(rows))
    strict_h = [r[1]["0"]["compliance_rate_strict"] for r in rows]
    strict_m = [r[1]["1"]["compliance_rate_strict"] for r in rows]
    near_h = [r[1]["0"]["compliance_rate_near_p98"] for r in rows]
    near_m = [r[1]["1"]["compliance_rate_near_p98"] for r in rows]
    width = 0.18
    ax.bar(pos - 1.5 * width, strict_h, width, label="human (strict ==)",      color="C0", alpha=0.6)
    ax.bar(pos - 0.5 * width, strict_m, width, label="machine (strict ==)",    color="C1", alpha=0.6)
    ax.bar(pos + 0.5 * width, near_h,   width, label="human (near, sim>0.98)", color="C0")
    ax.bar(pos + 1.5 * width, near_m,   width, label="machine (near, sim>0.98)",color="C1")
    ax.set_xticks(pos)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("black-format compliance rate")
    ax.set_title("WHY audit: HMCorp's gap is LARGER than DC's, yet CGS robust →\n"
                 "training-data gap does NOT drive the formatting confound")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate gaps
    for i, r in enumerate(rows):
        gap_near = r[1]["gap_near"]
        gap_strict = r[1]["gap_strict"]
        ax.annotate(f"near gap: +{gap_near:.3f}\nstrict gap: +{gap_strict:.3f}",
                    (i, 0.72), ha="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # (1,1) In-dist AUC vs OOD AUC summary table-as-plot
    ax = axes[1][1]
    ax.axis("off")
    cell3_q1_h = (cell3["q1_decision"] if cell3 else "?")
    cell3_fpr_shift = (cell3["per_corpus"]["test_formatted"]["flag_rate_human"]
                       - cell3["per_corpus"]["test"]["flag_rate_human"]) if cell3 else float("nan")
    txt = (
        "Headline summary (2x2 architecture × data)\n"
        "==========================================\n"
        "\n"
        "In-distribution (each model on its own test):\n"
        f"  UniX × HMCorp Py:  AUC {pb_py['per_corpus']['test']['auroc']:.3f}  "
        f"FPR(h) {pb_py['per_corpus']['test']['flag_rate_human']:.3f}\n"
        f"  UniX × HMCorp Ja:  AUC {pb_ja['per_corpus']['test']['auroc']:.3f}  "
        f"FPR(h) {pb_ja['per_corpus']['test']['flag_rate_human']:.3f}\n"
        + (f"  UniX × DroidColl:  AUC {cell3['per_corpus']['test']['auroc']:.3f}  "
           f"FPR(h) {cell3['per_corpus']['test']['flag_rate_human']:.3f}\n" if cell3 else
           "  UniX × DroidColl:  (not yet eval'd)\n")
        + f"  MBERT × DroidColl: AUC 0.999 FPR(h) ~0.01  (DroidDetect, vendored)\n"
        "\n"
        "Q1 corrected (human-FPR shift after black; >=0.10 = confound):\n"
        f"  UniX × HMCorp Py:  Δ FPR = {-0.001:+.3f}  → NO\n"
        f"  UniX × HMCorp Ja:  Δ FPR = {+0.005:+.3f}  → NO\n"
        + (f"  UniX × DroidColl:  Δ FPR = {cell3_fpr_shift:+.3f}  → "
           f"{'YES (corrected)' if cell3_fpr_shift >= 0.10 else 'NO'}\n" if cell3 else
           "  UniX × DroidColl:  (pending)\n")
        + f"  MBERT × DroidColl: Δ FPR = +0.82    → YES (memory)\n"
        "\n"
        "WHY (per-label compliance gap, near metric):\n"
        f"  HMCorp Python:      +{comp['hmcorp_python']['gap_near']:.3f}\n"
        f"  DroidCollection Py: +{comp['droid_python']['gap_near']:.3f}\n"
        "(HMCorp has BIGGER gap yet doesn't induce confound → gap-size\n"
        " doesn't predict severity.)\n"
        "\n"
        "Python OOD (CGS reproduction):\n"
        f"  DC test:        AUC {pd['per_corpus']['droidcollection_test']['auroc']:.3f}\n"
        f"  MS humans FPR:  {pd['per_corpus']['matrixstudio_humans']['flag_rate_human_at_0.5']:.3f}\n"
        f"  Leg humans FPR: {pd['per_corpus']['legendary_humans']['flag_rate_human_at_0.5']:.3f}\n"
        "\n"
        "Story:\n"
        "  * DroidCollection induces a formatting shortcut in BOTH archs.\n"
        "  * HMCorp does not, at least in UniXcoder (Cell 2 untested).\n"
        "  * Arch modulates severity ~4×: UniX/DC = +21pp, MBERT/DC = +82pp.\n"
        "  * Data composition is necessary; arch modulates magnitude.\n"
        "  * Compliance gap size doesn't predict the confound.\n"
    )
    ax.text(0.0, 1.0, txt, fontsize=8.5, family="monospace",
            verticalalignment="top", transform=ax.transAxes)

    fig.suptitle("CodeGPTSensor reproduction — Phase B + D + WHY-audit findings",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    out = ROOT / "results" / "cgs" / "findings_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
