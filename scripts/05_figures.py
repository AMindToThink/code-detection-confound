"""Figures from results/analysis.json. Saves PNGs to results/figures/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

A = json.loads((C.RESULTS / "analysis.json").read_text())
DETECTORS = list(C.DETECTOR_FAMILY)
FAM_COLOR = {"trained": "#a23a1f", "statistical": "#2a6f8e"}
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})


def fig_auroc_heatmaps():
    cells = A["cell_aurocs"]["cells"]
    fig, axes = plt.subplots(1, len(DETECTORS), figsize=(5 * len(DETECTORS), 3.4))
    if len(DETECTORS) == 1:
        axes = [axes]
    for ax, det in zip(axes, DETECTORS):
        M = np.full((len(C.BANDS), len(C.CONDITIONS)), np.nan)
        for i, b in enumerate(C.BANDS):
            for j, c in enumerate(C.CONDITIONS):
                M[i, j] = cells[det][f"{b}|{c}"]["auroc"]
        im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(C.CONDITIONS))); ax.set_xticklabels(C.CONDITIONS)
        ax.set_yticks(range(len(C.BANDS))); ax.set_yticklabels([b + "\nhumans" for b in C.BANDS])
        for i, b in enumerate(C.BANDS):
            for j, c in enumerate(C.CONDITIONS):
                kind = cells[det][f"{b}|{c}"]["kind"]
                tag = "M" if kind == "matched" else ("X" if kind == "mismatched" else "·")
                ax.text(j, i, f"{M[i,j]:.2f}\n[{tag}]", ha="center", va="center",
                        fontsize=10, fontweight="bold")
        ax.set_title(f"{det}\n({C.DETECTOR_FAMILY[det]})", fontsize=11)
        ax.set_xlabel("LLM prompt condition")
    fig.colorbar(im, ax=axes, label="AUROC (human vs LLM)", fraction=0.025)
    fig.suptitle("Detector AUROC per skill cell — M=skill-matched, X=mismatched", y=1.05)
    fig.savefig(C.FIGURES / "fig1_auroc_heatmaps.png")
    plt.close(fig)


def fig_delta_confound():
    dc = A["cell_aurocs"]["delta_confound"]
    ci = A["delta_confound_ci"]
    dets = DETECTORS
    deltas = [dc[d]["delta"] for d in dets]
    los = [max(dc[d]["delta"] - ci[d]["ci_lo"], 0) if ci.get(d) else 0 for d in dets]
    his = [max(ci[d]["ci_hi"] - dc[d]["delta"], 0) if ci.get(d) else 0 for d in dets]
    colors = [FAM_COLOR[C.DETECTOR_FAMILY[d]] for d in dets]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(dets)), deltas, yerr=[los, his], color=colors, capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(dets))); ax.set_xticklabels(dets, rotation=15)
    ax.set_ylabel(r"$\Delta_{confound}$ = AUROC$_{mismatched}$ − AUROC$_{matched}$")
    ax.set_title("Skill-confound size per detector (higher = more confounded)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in FAM_COLOR.items()])
    fig.savefig(C.FIGURES / "fig2_delta_confound.png")
    plt.close(fig)


def fig_within_species_slopes():
    sl = A["within_species_slopes"]
    dets = DETECTORS
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dets)); w = 0.35
    hs = [sl[d]["human_slope"] for d in dets]
    ls = [sl[d]["llm_slope"] for d in dets]
    h_err = [[sl[d]["human_slope"] - sl[d]["human_ci"][0] for d in dets],
             [sl[d]["human_ci"][1] - sl[d]["human_slope"] for d in dets]]
    l_err = [[sl[d]["llm_slope"] - sl[d]["llm_ci"][0] for d in dets],
             [sl[d]["llm_ci"][1] - sl[d]["llm_slope"] for d in dets]]
    ax.bar(x - w/2, hs, w, yerr=h_err, capsize=3, color="#b48a3a", label="humans: hard − easy")
    ax.bar(x + w/2, ls, w, yerr=l_err, capsize=3, color="#4a544f", label="LLMs: novice→natural→expert slope")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(dets, rotation=15)
    ax.set_ylabel("within-species score slope (SD units)")
    ax.set_title("Mechanism: does detector score track skill WITHIN a species?\n"
                 "(species classifier ⇒ ≈0; skill classifier ⇒ large positive)")
    ax.legend()
    fig.savefig(C.FIGURES / "fig3_within_species_slopes.png")
    plt.close(fig)


def fig_legendary():
    leg = A.get("legendary")
    if not leg:
        return
    authors = list(leg["by_author"])
    _generic = {"core", "team", "contributors", "devs"}
    short = [a.split()[0] if a.split()[-1].lower() in _generic else a.split()[-1] for a in authors]
    base = leg.get("cf_human_baseline", {})
    fig, axes = plt.subplots(1, len(DETECTORS), figsize=(5.2 * len(DETECTORS), 4.2), sharey=True)
    if len(DETECTORS) == 1:
        axes = [axes]
    na = leg.get("native_argmax_DroidDetect", {})
    x = np.arange(len(authors)); w = 0.38
    for ax, det in zip(axes, DETECTORS):
        col = FAM_COLOR[C.DETECTOR_FAMILY[det]]
        if det == "DroidDetect" and na:
            # NATIVE argmax operating point (the model's actual predictions); baseline is
            # DroidDetect's true FPR on ordinary human code, not a forced 5%.
            src = na["by_author"]
            leg_v = [src[au]["flag_rate"] for au in authors]
            leg_err = [[max(src[au]["flag_rate"] - src[au]["ci_lo"], 0) for au in authors],
                       [max(src[au]["ci_hi"] - src[au]["flag_rate"], 0) for au in authors]]
            bv = na["cf_human"]["flag_rate"]
            blo, bhi = na["cf_human"]["ci_lo"], na["cf_human"]["ci_hi"]
            opname = "native argmax (P(machine)>0.5)"
            llm_ref = na["cf_llm"]["flag_rate"]
        else:
            src = {au: leg["by_author"][au][det] for au in authors}
            leg_v = [src[au]["flag_rate_5pctFPR"] for au in authors]
            leg_err = [[max(src[au]["flag_rate_5pctFPR"] - src[au].get("ci_lo", src[au]["flag_rate_5pctFPR"]), 0) for au in authors],
                       [max(src[au].get("ci_hi", src[au]["flag_rate_5pctFPR"]) - src[au]["flag_rate_5pctFPR"], 0) for au in authors]]
            b = base.get(det, {})
            bv = b.get("flag_rate_5pctFPR", float("nan"))
            blo, bhi = b.get("ci_lo", bv), b.get("ci_hi", bv)
            opname = "threshold @ 5% FPR on humans"
            llm_ref = None
        ax.bar(x - w / 2, leg_v, w, yerr=leg_err, capsize=3, color=col,
               label="legendary pre-LLM (human)")
        ax.bar(x + w / 2, [bv] * len(authors), w,
               yerr=[[max(bv - blo, 0)] * len(authors), [max(bhi - bv, 0)] * len(authors)],
               capsize=3, color=col, alpha=0.30, label="in-distribution Codeforces humans")
        if llm_ref is not None:
            ax.axhline(llm_ref, color="0.25", ls="--", lw=1, label="Codeforces LLM code")
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{det} ({C.DETECTOR_FAMILY[det]})\n{opname}", fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction flagged AI")
    axes[0].legend(fontsize=7.5, loc="upper right")
    for ax in axes[1:]:
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Provably-human pre-LLM elite code flagged as AI vs. in-distribution human "
                 "baseline (bars = bootstrap 95% CI; each detector at its own operating point)",
                 y=1.03, fontsize=11)
    fig.savefig(C.FIGURES / "fig4_legendary_flag_rates.png")
    plt.close(fig)


def fig_difficulty_style():
    d2s = A["difficulty_to_style"]
    keys = ["feat_comment_density", "feat_has_type_hints", "feat_mean_identifier_len",
            "feat_single_char_ident_frac", "feat_has_docstring", "feat_mean_line_len"]
    keys = [k for k in keys if k in d2s]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(keys)); w = 0.38
    easy = [d2s[k]["easy"] for k in keys]
    hard = [d2s[k]["hard"] for k in keys]
    # normalize each feature to its max for visual comparability
    norm = [max(e, h, 1e-9) for e, h in zip(easy, hard)]
    ax.bar(x - w/2, [e/n for e, n in zip(easy, norm)], w, label="easy-problem humans", color="#7fb069")
    ax.bar(x + w/2, [h/n for h, n in zip(hard, norm)], w, label="hard-problem humans", color="#a23a1f")
    ax.set_xticks(x); ax.set_xticklabels([k.replace("feat_", "") for k in keys], rotation=25, ha="right")
    ax.set_ylabel("mean (normalized per feature)")
    ax.set_title("Do harder-problem human solutions look more 'expert'?")
    ax.legend()
    fig.savefig(C.FIGURES / "fig5_difficulty_to_style.png")
    plt.close(fig)


def fig_atcoder():
    at = A.get("atcoder_author_skill")
    if not at:
        return
    dets = DETECTORS
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(dets))
    hi = [at["by_detector"][d]["mean_z_high"] for d in dets]
    lo = [at["by_detector"][d]["mean_z_low"] for d in dets]
    w = 0.38
    ax.bar(x - w/2, lo, w, label="low-skill authors (rating ≤ 800)", color="#7fb069")
    ax.bar(x + w/2, hi, w, label="high-skill authors (rating ≥ 2000)", color="#a23a1f")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(dets, rotation=15)
    ax.set_ylabel("mean detector score (z, within AtCoder)")
    ax.set_title("CONFIRMATORY: real author Elo — do detectors score\n"
                 "higher-skill HUMANS as more 'AI'? (n=%d)" % at["n"])
    ax.legend()
    fig.savefig(C.FIGURES / "fig6_atcoder_author_skill.png")
    plt.close(fig)


def main():
    fig_auroc_heatmaps()
    fig_delta_confound()
    fig_within_species_slopes()
    fig_legendary()
    fig_difficulty_style()
    fig_atcoder()
    print("wrote figures to", C.FIGURES)


if __name__ == "__main__":
    main()
