"""Analysis: AUROC cells, Δ_confound, within-species skill slopes, style mediation,
and the legendary-coder probe. Emits results/analysis.json (for figures) and
results/tables/paper_macros.json (inline scalars for the report).

Methodology (per the two statistics critiques):
  * AUROC is the PRIMARY metric (threshold-free, prevalence-robust), not F1.
  * All CIs use a CLUSTER BOOTSTRAP over problems (samples are clustered within problem).
  * Δ_confound is reported PER DETECTOR (never averaged across detectors).
  * Within-species score slopes on the skill axis are the mechanism test
    (pure species classifier => slope ~ 0; skill classifier => large signed slope).
  * Detector scores are z-scored per detector so slopes are comparable across detectors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

DETECTORS = list(C.DETECTOR_FAMILY)
RNG = np.random.default_rng(0)
N_BOOT = 2000

MATCHED = {("hard", "natural"), ("easy", "novice")}
MISMATCHED = {("easy", "natural"), ("hard", "novice")}
COND_RANK = {"novice": 0, "natural": 1, "expert": 2}   # skill ordering of LLM prompts


# ----------------------------------------------------------------- helpers
def auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    if len(neg_scores) < 3 or len(pos_scores) < 3:
        return float("nan")
    y = np.r_[np.zeros(len(neg_scores)), np.ones(len(pos_scores))]
    s = np.r_[neg_scores, pos_scores]
    return roc_auc_score(y, s)


def cluster_bootstrap_auroc(df_neg: pd.DataFrame, df_pos: pd.DataFrame, det: str):
    """AUROC point estimate + 95% CI, resampling PROBLEMS (clusters) with replacement.
    Both classes are resampled on the shared problem id so the pairing is preserved."""
    point = auroc(df_neg[det].dropna().values, df_pos[det].dropna().values)
    probs = np.union1d(df_neg["problem_id"].unique(), df_pos["problem_id"].unique())
    neg_by = {p: g[det].dropna().values for p, g in df_neg.groupby("problem_id")}
    pos_by = {p: g[det].dropna().values for p, g in df_pos.groupby("problem_id")}
    boots = []
    for _ in range(N_BOOT):
        samp = RNG.choice(probs, size=len(probs), replace=True)
        n = np.concatenate([neg_by[p] for p in samp if p in neg_by]) if len(probs) else np.array([])
        p = np.concatenate([pos_by[q] for q in samp if q in pos_by]) if len(probs) else np.array([])
        a = auroc(n, p)
        if not np.isnan(a):
            boots.append(a)
    if not boots:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def zscore_by_detector(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for det in DETECTORS:
        v = df[det].astype(float)
        mu, sd = v.mean(), v.std(ddof=0)
        df[det + "_z"] = (v - mu) / (sd if sd > 0 else 1.0)
    return df


# ----------------------------------------------------------------- main analyses
def cell_aurocs(cf: pd.DataFrame) -> dict:
    """AUROC for every (detector, band, condition) cell + matched/mismatched + Δ_confound."""
    humans = cf[cf.species == "human"]
    llms = cf[cf.species == "llm"]
    out = {"cells": {}, "delta_confound": {}}
    for det in DETECTORS:
        out["cells"][det] = {}
        matched_vals, mismatched_vals = [], []
        for band in C.BANDS:
            hb = humans[humans.band == band]
            for cond in C.CONDITIONS:
                lc = llms[llms.condition == cond]
                pt, lo, hi = cluster_bootstrap_auroc(hb, lc, det)
                kind = ("matched" if (band, cond) in MATCHED
                        else "mismatched" if (band, cond) in MISMATCHED else "expert")
                out["cells"][det][f"{band}|{cond}"] = {
                    "auroc": pt, "ci_lo": lo, "ci_hi": hi, "kind": kind,
                    "n_human": int(len(hb)), "n_llm": int(len(lc)),
                }
                if kind == "matched":
                    matched_vals.append(pt)
                elif kind == "mismatched":
                    mismatched_vals.append(pt)
        out["delta_confound"][det] = {
            "auroc_matched": float(np.nanmean(matched_vals)),
            "auroc_mismatched": float(np.nanmean(mismatched_vals)),
            "delta": float(np.nanmean(mismatched_vals) - np.nanmean(matched_vals)),
            "family": C.DETECTOR_FAMILY[det],
        }
    return out


def delta_confound_ci(cf: pd.DataFrame) -> dict:
    """Cluster-bootstrap CI on Δ_confound per detector. Δ = mean(per-cell AUROC over
    mismatched cells) − mean(per-cell AUROC over matched cells), consistent with the
    point estimate in cell_aurocs(). Resample problems once per draw, recompute every
    cell's AUROC, then average within kind and difference."""
    humans = cf[cf.species == "human"]
    llms = cf[cf.species == "llm"]
    probs = cf["problem_id"].unique()
    # precompute per-problem score arrays per (group, det) to avoid pandas in the loop
    res = {}
    for det in DETECTORS:
        hb = {(b): {p: g[det].dropna().values for p, g in humans[humans.band == b].groupby("problem_id")}
              for b in C.BANDS}
        lc = {(c): {p: g[det].dropna().values for p, g in llms[llms.condition == c].groupby("problem_id")}
              for c in C.CONDITIONS}

        def kind_mean(kind_set, samp):
            cell_aucs = []
            for (band, cond) in kind_set:
                neg = np.concatenate([hb[band][p] for p in samp if p in hb[band]]) if hb[band] else np.array([])
                pos = np.concatenate([lc[cond][p] for p in samp if p in lc[cond]]) if lc[cond] else np.array([])
                a = auroc(neg, pos)
                if not np.isnan(a):
                    cell_aucs.append(a)
            return np.mean(cell_aucs) if cell_aucs else np.nan

        boots = []
        for _ in range(500):
            samp = RNG.choice(probs, size=len(probs), replace=True)
            m = kind_mean(MATCHED, samp); mm = kind_mean(MISMATCHED, samp)
            if not (np.isnan(m) or np.isnan(mm)):
                boots.append(mm - m)
        res[det] = ({"ci_lo": float(np.percentile(boots, 2.5)),
                     "ci_hi": float(np.percentile(boots, 97.5))} if boots else {})
    return res


def within_species_slopes(cf: pd.DataFrame) -> dict:
    """Mechanism test. Human slope = mean(z|hard) - mean(z|easy). LLM slope = OLS slope
    of z on COND_RANK (novice<natural<expert). Problem-clustered bootstrap CIs.
    Skill-classifier prediction: human slope > 0 (harder=more 'AI'); LLM slope > 0
    (more 'expert' prompt = more 'AI', novice = less)."""
    cfz = zscore_by_detector(cf)
    humans = cfz[cfz.species == "human"]
    llms = cfz[cfz.species == "llm"]
    out = {}
    for det in DETECTORS:
        z = det + "_z"

        def human_slope(d):
            hi = d[d.band == "hard"][z]; lo = d[d.band == "easy"][z]
            return hi.mean() - lo.mean()

        def llm_slope(d):
            x = d["condition"].map(COND_RANK).values.astype(float)
            y = d[z].values
            if np.std(x) == 0:
                return float("nan")
            return float(np.polyfit(x, y, 1)[0])

        hp = human_slope(humans)
        lp = llm_slope(llms)
        # cluster bootstrap over problems
        hprobs = humans["problem_id"].unique(); lprobs = llms["problem_id"].unique()
        hb, lb = [], []
        hg = {p: g for p, g in humans.groupby("problem_id")}
        lg = {p: g for p, g in llms.groupby("problem_id")}
        for _ in range(N_BOOT):
            hs = RNG.choice(hprobs, len(hprobs), replace=True)
            ls = RNG.choice(lprobs, len(lprobs), replace=True)
            hb.append(human_slope(pd.concat([hg[p] for p in hs])))
            lb.append(llm_slope(pd.concat([lg[p] for p in ls])))
        out[det] = {
            "human_slope": float(hp), "human_ci": [float(np.nanpercentile(hb, 2.5)), float(np.nanpercentile(hb, 97.5))],
            "llm_slope": float(lp), "llm_ci": [float(np.nanpercentile(lb, 2.5)), float(np.nanpercentile(lb, 97.5))],
            "family": C.DETECTOR_FAMILY[det],
        }
    return out


def difficulty_to_style(cf: pd.DataFrame) -> dict:
    """Do harder-problem HUMAN solutions actually look more 'expert' in surface style?
    Mean style features by band (humans only). Tests the proxy's premise."""
    h = cf[cf.species == "human"]
    feat_cols = [c for c in cf.columns if c.startswith("feat_")]
    res = {}
    for c in feat_cols:
        res[c] = {"easy": float(h[h.band == "easy"][c].mean()),
                  "hard": float(h[h.band == "hard"][c].mean())}
    return res


def style_mediation(cf: pd.DataFrame) -> dict:
    """How much of each detector's score is explained by surface style features alone?
    R^2 of OLS(detector_score ~ style features) on the cf set. High R^2 => the detector
    largely reduces to a style detector."""
    from sklearn.linear_model import LinearRegression
    feat_cols = [c for c in cf.columns if c.startswith("feat_")]
    X = cf[feat_cols].fillna(0).values
    res = {}
    for det in DETECTORS:
        y = cf[det].astype(float).values
        ok = ~np.isnan(y)
        m = LinearRegression().fit(X[ok], y[ok])
        res[det] = {"r2_style": float(m.score(X[ok], y[ok]))}
    return res


def legendary_probe(df: pd.DataFrame) -> dict:
    """AI-flag rate on ironclad-human pre-LLM elite code, per detector and author.
    Threshold per statistical detector = 95th percentile of cf-human scores (=> 5% FPR
    on ordinary humans). DroidDetect uses its native 0.5 and also the 5%-FPR threshold."""
    cf_h = df[(df.dataset == "cf") & (df.species == "human")]
    leg = df[df.dataset == "legendary"]
    if leg.empty:
        return {}
    out = {"thresholds": {}, "by_detector": {}, "by_author": {}, "cf_human_baseline": {}}
    thr = {}
    for det in DETECTORS:
        t = float(np.nanpercentile(cf_h[det].astype(float), 95))
        thr[det] = t
        out["thresholds"][det] = t
        out["by_detector"][det] = {
            "flag_rate_5pctFPR": float((leg[det] > t).mean()),
            "mean_score": float(leg[det].mean()),
            "family": C.DETECTOR_FAMILY[det],
        }
        out["cf_human_baseline"][det] = {"mean_score": float(cf_h[det].mean())}
    out["by_detector"]["DroidDetect"]["flag_rate_native_0.5"] = float((leg["DroidDetect"] > 0.5).mean())
    for author, g in leg.groupby("author"):
        out["by_author"][author] = {
            det: {"flag_rate_5pctFPR": float((g[det] > thr[det]).mean()),
                  "mean_score": float(g[det].mean())} for det in DETECTORS}
        out["by_author"][author]["n"] = int(len(g))
    return out


def main() -> None:
    df = pd.read_parquet(C.SCORED_PARQUET)
    cf = df[df.dataset == "cf"].copy()
    print(f"cf samples: {len(cf)} ({(cf.species=='human').sum()} human, {(cf.species=='llm').sum()} llm)")

    analysis = {
        "n_samples": {"cf": int(len(cf)), "legendary": int((df.dataset == "legendary").sum())},
        "cell_aurocs": cell_aurocs(cf),
        "delta_confound_ci": delta_confound_ci(cf),
        "within_species_slopes": within_species_slopes(cf),
        "difficulty_to_style": difficulty_to_style(cf),
        "style_mediation": style_mediation(cf),
        "legendary": legendary_probe(df),
        "detector_family": C.DETECTOR_FAMILY,
        "skill_source": C.SKILL_SOURCE,
    }
    (C.RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print("wrote results/analysis.json")
    emit_macros(analysis)


def emit_macros(a: dict) -> None:
    """Inline scalars for the report (one source of truth — no hand-typed numbers)."""
    m = {}
    for det in DETECTORS:
        dc = a["cell_aurocs"]["delta_confound"][det]
        ci = a["delta_confound_ci"].get(det, {})
        m[f"delta_{det}"] = round(dc["delta"], 3)
        m[f"aurocMatched_{det}"] = round(dc["auroc_matched"], 3)
        m[f"aurocMismatched_{det}"] = round(dc["auroc_mismatched"], 3)
        if ci:
            m[f"deltaCIlo_{det}"] = round(ci["ci_lo"], 3)
            m[f"deltaCIhi_{det}"] = round(ci["ci_hi"], 3)
        sl = a["within_species_slopes"][det]
        m[f"humanSlope_{det}"] = round(sl["human_slope"], 3)
        m[f"llmSlope_{det}"] = round(sl["llm_slope"], 3)
        m[f"styleR2_{det}"] = round(a["style_mediation"][det]["r2_style"], 3)
        if a["legendary"]:
            m[f"legFlag_{det}"] = round(a["legendary"]["by_detector"][det]["flag_rate_5pctFPR"], 3)
    if a["legendary"]:
        m["legFlag_DroidDetect_native"] = round(a["legendary"]["by_detector"]["DroidDetect"]["flag_rate_native_0.5"], 3)
    (C.TABLES / "paper_macros.json").write_text(json.dumps(m, indent=2))
    print(f"wrote results/tables/paper_macros.json ({len(m)} macros)")


if __name__ == "__main__":
    main()
