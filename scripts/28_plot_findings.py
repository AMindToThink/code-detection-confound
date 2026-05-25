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

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (0,0) Phase B formatting flip: per-row machine-prob distributions, paired test→formatted
    ax = axes[0][0]
    width = 0.35
    langs = ["python", "java"]
    rep_for = {"python": pb_py, "java": pb_ja}
    labels = ["test", "test_formatted"]
    colors = ["C0", "C1"]
    bar_pos = np.arange(len(langs))
    for i, lab in enumerate(labels):
        vals = [rep_for[L]["per_corpus"][lab]["machine_prob_mean_on_machine_rows"]
                for L in langs]
        ax.bar(bar_pos + (i - 0.5) * width, vals, width,
               label=lab, color=colors[i], alpha=0.85)
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(langs)
    ax.set_ylabel("mean P(machine) on machine rows")
    ax.set_title("Phase B Q1: formatting flip on raw checkpoint's own test\n"
                 "(Q1=YES would need Δ ≥ 0.10; observed ≤ 0.008)")
    ax.set_ylim(0.5, 0.8)
    ax.legend()
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
    txt = (
        "Headline summary\n"
        "================\n"
        "\n"
        "In-distribution (reproduces paper Table 6):\n"
        f"  Python: AUC {pb_py['per_corpus']['test']['auroc']:.4f}   F1 {pb_py['per_corpus']['test']['f1']:.4f}\n"
        f"  Java:   AUC {pb_ja['per_corpus']['test']['auroc']:.4f}   F1 {pb_ja['per_corpus']['test']['f1']:.4f}\n"
        "\n"
        "Q1 formatting-flip on own test set (Q1=YES needs Δ ≥ 0.10):\n"
        f"  Python: Δp1={pb_py['q1_metrics']['machine_prob_drop_test_vs_formatted']:+.4f}  "
        f"ΔTPR={pb_py['q1_metrics']['flag_rate_drop_test_vs_formatted']:+.4f}  → Q1=NO\n"
        f"  Java:   Δp1={pb_ja['q1_metrics']['machine_prob_drop_test_vs_formatted']:+.4f}  "
        f"ΔTPR={pb_ja['q1_metrics']['flag_rate_drop_test_vs_formatted']:+.4f}  → Q1=NO\n"
        "\n"
        "OOD generalization (Python only):\n"
        f"  HMCorp test (in-dist):  AUC {pd['per_corpus']['hmcorp_python_test']['auroc']:.4f}\n"
        f"  DroidCollection mixed:  AUC {pd['per_corpus']['droidcollection_test']['auroc']:.4f}\n"
        f"  MatrixStudio humans FPR: {pd['per_corpus']['matrixstudio_humans']['flag_rate_human_at_0.5']:.4f}\n"
        f"  Legendary humans FPR:    {pd['per_corpus']['legendary_humans']['flag_rate_human_at_0.5']:.4f}\n"
        "\n"
        "Training-data per-label compliance:\n"
        f"  HMCorp gap (near):     +{comp['hmcorp_python']['gap_near']:.3f}\n"
        f"  DroidCollection (near): +{comp['droid_python']['gap_near']:.3f}\n"
        "\n"
        "Implication:\n"
        "  * Q1=NO on both languages → no formatting shortcut here.\n"
        "  * Yet CGS catastrophic OOD on Codeforces + DC → it has SOME\n"
        "    shortcut, just not formatting.\n"
        "  * Training-data formatting gap is LARGER in HMCorp than DC,\n"
        "    yet CGS has no formatting confound → data composition is\n"
        "    NOT the driver. Architecture / pretraining drives it."
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
