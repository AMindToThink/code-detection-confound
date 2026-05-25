"""Emit LaTeX macros (VARIABLES ONLY — no prose) from results/analysis.json.

This is the single source of truth for every number in paper/paper.tex. The paper
\\input{}s paper/macros.tex and references each value by name (e.g. \\resDeltaDroidDetect),
so no number is ever hand-typed into the prose. Edit prose in paper/paper.tex; edit
numbers nowhere (re-run this script).

Macro names are letters-only (LaTeX requirement): \\res<CamelCaseName>.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

PAPER = C.ROOT / "paper"
PAPER.mkdir(exist_ok=True)
DET = list(C.DETECTOR_FAMILY)


def camel(s: str) -> str:
    """letters-only CamelCase token for a LaTeX command name."""
    parts = re.split(r"[^A-Za-z]+", s)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def main() -> None:
    a = json.loads((C.RESULTS / "analysis.json").read_text())
    M: dict[str, str] = {}

    def put(name: str, val, nd: int = 3) -> None:
        if isinstance(val, float):
            val = f"{val:.{nd}f}"
        M[name] = str(val)

    # sample counts
    put("NCf", a["n_samples"]["cf"], 0)
    put("NLegendary", a["n_samples"]["legendary"], 0)
    put("NAtcoder", a["n_samples"].get("atcoder", 0), 0)
    put("NModels", len(a.get("llm_models", [])), 0)
    mean_delta = sum(a["cell_aurocs"]["delta_confound"][d]["delta"] for d in DET) / len(DET)
    put("MeanDelta", mean_delta)
    mean_hslope = sum(a["within_species_slopes"][d]["human_slope"] for d in DET) / len(DET)
    put("MeanHumanSlope", mean_hslope, 2)

    for det in DET:
        D = camel(det)  # e.g. DroidDetect / FastDetectGpt / Binoculars
        dc = a["cell_aurocs"]["delta_confound"][det]
        put(f"Delta{D}", dc["delta"])
        put(f"AurocMatched{D}", dc["auroc_matched"])
        put(f"AurocMismatched{D}", dc["auroc_mismatched"])
        ci = a["delta_confound_ci"].get(det, {})
        if ci:
            put(f"DeltaCIlo{D}", ci["ci_lo"]); put(f"DeltaCIhi{D}", ci["ci_hi"])
        sl = a["within_species_slopes"][det]
        put(f"HumanSlope{D}", sl["human_slope"], 2)
        put(f"HumanSlopeLo{D}", sl["human_ci"][0], 2); put(f"HumanSlopeHi{D}", sl["human_ci"][1], 2)
        put(f"LlmSlope{D}", sl["llm_slope"], 2)
        put(f"LlmSlopeLo{D}", sl["llm_ci"][0], 2); put(f"LlmSlopeHi{D}", sl["llm_ci"][1], 2)
        put(f"StyleRsq{D}", a["style_mediation"][det]["r2_style"])
        if a.get("legendary"):
            lg = a["legendary"]["by_detector"][det]
            put(f"LegFlag{D}", lg["flag_rate_5pctFPR"])
        if a.get("atcoder_author_skill"):
            at = a["atcoder_author_skill"]["by_detector"][det]
            put(f"AtSlope{D}", at["slope_per_400elo"]); put(f"AtHiLo{D}", at["high_minus_low"], 2)

    # legendary specifics
    if a.get("legendary"):
        put("LegFlagDroidNative", a["legendary"]["by_detector"]["DroidDetect"]["flag_rate_native_0.5"])
        for au, v in a["legendary"]["by_author"].items():
            put(f"LegAuthor{camel(au)}", v["DroidDetect"]["flag_rate_5pctFPR"])
            put(f"LegAuthor{camel(au)}N", v["n"], 0)
        # DroidDetect native-argmax operating point (its actual predictions)
        na = a["legendary"].get("native_argmax_DroidDetect", {})
        if na:
            put("DroidArgmaxCfHumanFPR", na["cf_human"]["flag_rate"])
            put("DroidArgmaxCfLlm", na["cf_llm"]["flag_rate"])
            put("DroidArgmaxLegendary", na["legendary"]["flag_rate"])
            for au, v in na["by_author"].items():
                put(f"DroidArgmax{camel(au)}", v["flag_rate"])

    # atcoder bands
    if a.get("atcoder_author_skill"):
        at = a["atcoder_author_skill"]
        for b, n in at["band_n"].items():
            put(f"AtN{camel(b)}", n, 0)
        for b, r in at["mean_rating"].items():
            put(f"AtRating{camel(b)}", round(r), 0)

    # difficulty -> style (humans), a few key features
    d2s = a["difficulty_to_style"]
    for feat in ["feat_has_type_hints", "feat_comment_density", "feat_mean_identifier_len"]:
        if feat in d2s:
            nm = camel(feat.replace("feat_", ""))
            put(f"Style{nm}Easy", d2s[feat]["easy"]); put(f"Style{nm}Hard", d2s[feat]["hard"])

    # Format-artifact ablation (headline causal result)
    abl_path = C.RESULTS / "format_ablation.json"
    if abl_path.exists():
        abl_full = json.loads(abl_path.read_text())
        ab = abl_full["droiddetect_argmax"]
        if "identifier_preservation_ast_canon" in abl_full:
            put("AblIdentPreserved", abl_full["identifier_preservation_ast_canon"])
        def fr(t, c): return ab[t][c]["flag_rate"]
        put("AblDcHumanOrig", fr("original", "DroidCollection-human"))
        put("AblDcHumanCanon", fr("ast_canon", "DroidCollection-human"))
        put("AblDcHumanStrip", fr("strip_comments", "DroidCollection-human"))
        if "black" in ab:
            put("AblDcHumanBlack", fr("black", "DroidCollection-human"))
            put("AblMsHumanBlack", fr("black", "MatrixStudio-human"))
            put("AblDcMachineBlack", fr("black", "DroidCollection-machine"))
            put("AblGapBlack", fr("black", "DroidCollection-machine") - fr("black", "DroidCollection-human"))
        put("AblMsHumanOrig", fr("original", "MatrixStudio-human"))
        put("AblMsHumanCanon", fr("ast_canon", "MatrixStudio-human"))
        put("AblDcMachineOrig", fr("original", "DroidCollection-machine"))
        put("AblDcMachineCanon", fr("ast_canon", "DroidCollection-machine"))
        put("AblGapOrig", fr("original", "DroidCollection-machine") - fr("original", "DroidCollection-human"))
        put("AblGapCanon", fr("ast_canon", "DroidCollection-machine") - fr("ast_canon", "DroidCollection-human"))

    # Training-data formatting confound (why black ruins predictions)
    cf_path = C.RESULTS / "formatting_confound.json"
    if cf_path.exists():
        cfd = json.loads(cf_path.read_text())
        put("FmtHumanAlreadyBlack", cfd["DroidCollection_human"]["already_black_frac"])
        put("FmtMachineAlreadyBlack", cfd["DroidCollection_machine"]["already_black_frac"])
        put("FmtHumanBlackSim", cfd["DroidCollection_human"]["black_similarity_mean"])
        put("FmtMachineBlackSim", cfd["DroidCollection_machine"]["black_similarity_mean"])

    # Finding 7: family-sibling detectors (different size/granularity, same authors+data).
    # Numbers come from results/other_detectors.json (scripts/10_other_detectors.py).
    oth_path = C.RESULTS / "other_detectors.json"
    if oth_path.exists():
        oth = json.loads(oth_path.read_text())
        sib_token = {  # repo id -> short letters-only macro token
            "project-droid/DroidDetect-Base-Binary": "BaseBinary",
            "project-droid/DroidDetect-Base": "BaseFour",
            "project-droid/DroidDetect-Large-Binary": "LargeBinary",
        }
        for repo, tok in sib_token.items():
            if repo not in oth:
                continue
            r = oth[repo]
            put(f"Sib{tok}NClasses", r["n_classes"], 0)
            put(f"Sib{tok}HumanOrig", r["dc_human_original"])
            put(f"Sib{tok}HumanBlack", r["dc_human_black"])
            put(f"Sib{tok}Machine", r["dc_machine"])
            put(f"Sib{tok}MsHuman", r["matrixstudio_human"])
            put(f"Sib{tok}LegAll", r["legendary_all"])
        # smallest and largest human-black flip across the whole family (for prose bounds)
        blacks = [r["dc_human_black"] for r in oth.values()]
        origs = [r["dc_human_original"] for r in oth.values()]
        put("SibFamilyHumanBlackMin", min(blacks))
        put("SibFamilyHumanBlackMax", max(blacks))
        put("SibFamilyHumanOrigMax", max(origs))
        put("SibFamilyN", len(oth), 0)

    # DroidDetect in-distribution validation (proof the implementation is faithful)
    val_path = C.RESULTS / "droiddetect_validation.json"
    if val_path.exists():
        v = json.loads(val_path.read_text())
        put("DroidInDistAUROC", v["auroc_in_distribution"])
        put("DroidInDistHumanFPR", v["human_flag_rate_argmax"])
        put("DroidInDistMachineFlag", v["machine_flag_rate_argmax"])

    # --------------------------------------------------------------------------
    # CodeGPTSensor Phase B — Q1 formatting flip on its own test data
    # --------------------------------------------------------------------------
    for lang_id, name_tok in [("python", "Py"), ("java", "Ja")]:
        pb = C.RESULTS / "cgs" / f"{lang_id}_raw_ce" / "eval_phase_b.json"
        if not pb.exists():
            continue
        r = json.loads(pb.read_text())
        t = r["per_corpus"]["test"]
        f = r["per_corpus"]["test_formatted"]
        lm = r["per_corpus"]["test_lenmatched"]
        put(f"Cgs{name_tok}TestAUROC", t["auroc"])
        put(f"Cgs{name_tok}TestF1", t["f1"])
        put(f"Cgs{name_tok}TestAcc", t["acc"])
        put(f"Cgs{name_tok}TestFprHuman", t["flag_rate_human"])
        put(f"Cgs{name_tok}TestTprMachine", t["flag_rate_machine"])
        put(f"Cgs{name_tok}FmtAUROC", f["auroc"])
        put(f"Cgs{name_tok}FmtFprHuman", f["flag_rate_human"])
        put(f"Cgs{name_tok}FmtTprMachine", f["flag_rate_machine"])
        put(f"Cgs{name_tok}LenmAUROC", lm["auroc"])
        put(f"Cgs{name_tok}QoneMachineProbDrop", r["q1_metrics"]["machine_prob_drop_test_vs_formatted"])
        put(f"Cgs{name_tok}QoneFlagDrop", r["q1_metrics"]["flag_rate_drop_test_vs_formatted"])
        put(f"Cgs{name_tok}Qone", r["q1_decision"].upper())

    # --------------------------------------------------------------------------
    # CodeGPTSensor Phase D — OOD generalization (Python only)
    # --------------------------------------------------------------------------
    pd_path = C.RESULTS / "cgs" / "python_raw_ce" / "eval_phase_d.json"
    if pd_path.exists():
        d = json.loads(pd_path.read_text())["per_corpus"]
        put("CgsPyDcAUROC", d["droidcollection_test"]["auroc"])
        put("CgsPyDcHumanFPR", d["droidcollection_test"]["flag_rate_human_at_0.5"])
        put("CgsPyDcMachineFlag", d["droidcollection_test"]["flag_rate_machine_at_0.5"])
        put("CgsPyMsHumanFPR", d["matrixstudio_humans"]["flag_rate_human_at_0.5"])
        put("CgsPyLegHumanFPR", d["legendary_humans"]["flag_rate_human_at_0.5"])
        put("CgsPyHmcorpTestECE", d["hmcorp_python_test"]["ece"])

    # --------------------------------------------------------------------------
    # WHY audit — per-label formatter compliance
    # --------------------------------------------------------------------------
    cau = C.RESULTS / "phase_e" / "compliance_audit.json"
    if cau.exists():
        a = json.loads(cau.read_text())
        for ds, tok in [("hmcorp_python", "Hmcorp"), ("droid_python", "Droid")]:
            if ds in a:
                put(f"Comp{tok}HumanNear",   a[ds]["0"]["compliance_rate_near_p98"])
                put(f"Comp{tok}MachineNear", a[ds]["1"]["compliance_rate_near_p98"])
                put(f"Comp{tok}GapNear",     a[ds]["gap_near"])
                put(f"Comp{tok}HumanStrict",   a[ds]["0"]["compliance_rate_strict"])
                put(f"Comp{tok}MachineStrict", a[ds]["1"]["compliance_rate_strict"])
                put(f"Comp{tok}GapStrict",     a[ds]["gap_strict"])

    # --------------------------------------------------------------------------
    # Cell 3 — UniXcoder × DroidCollection (2x2 architecture x data experiment)
    # --------------------------------------------------------------------------
    cell3 = C.RESULTS / "cgs" / "unixcoder_dc_ce" / "eval_q1.json"
    if cell3.exists():
        r = json.loads(cell3.read_text())
        t = r["per_corpus"]["test"]
        f = r["per_corpus"]["test_formatted"]
        put("CellThreeTestAUROC", t["auroc"])
        put("CellThreeTestF1", t["f1"])
        put("CellThreeTestFprHuman", t["flag_rate_human"])
        put("CellThreeTestTprMachine", t["flag_rate_machine"])
        put("CellThreeFmtAUROC", f["auroc"])
        put("CellThreeFmtFprHuman", f["flag_rate_human"])
        put("CellThreeFmtTprMachine", f["flag_rate_machine"])
        put("CellThreeMachineProbDrop", r["q1_metrics"]["machine_prob_drop_test_vs_formatted"])
        put("CellThreeFlagDrop", r["q1_metrics"]["flag_rate_drop_test_vs_formatted"])
        put("CellThreeQone", r["q1_decision"].upper())

    # --------------------------------------------------------------------------
    # Cross-cell consolidated Q1 — both metrics per cell (corrected vs preregistered)
    # See scripts/32_cross_cell_q1.py for the pre-registration error story.
    # --------------------------------------------------------------------------
    ccq = C.RESULTS / "phase_e" / "cross_cell_q1.json"
    if ccq.exists():
        cells = json.loads(ccq.read_text())["cells"]
        # Map label → CamelCase macro suffix
        label_to_tok = {"Cell1_Py": "CellOnePy", "Cell1_Ja": "CellOneJa",
                        "Cell3": "CellThree"}
        for c in cells:
            if not c.get("available"):
                continue
            tok = label_to_tok.get(c["label"])
            if tok is None:
                continue
            put(f"{tok}HumanFprShift",    c["fpr_human_shift"])
            put(f"{tok}MachineTprDrop",   c["tpr_machine_drop"])
            put(f"{tok}AucDropFmt",       c["auroc_drop"])
            put(f"{tok}QoneMachineSide", "YES" if c["q1_machine_side"] else "NO")
            put(f"{tok}QoneHumanSide",   "YES" if c["q1_human_side"]   else "NO")

    lines = [f"\\newcommand{{\\res{k}}}{{{v}}}" for k, v in sorted(M.items())]
    header = ("% AUTO-GENERATED by scripts/06_build_macros.py from results/analysis.json.\n"
              "% Variables only. Do NOT edit by hand and do NOT type numbers into paper.tex —\n"
              "% reference these macros (\\res...) instead. Re-run the pipeline to refresh.\n")
    (PAPER / "macros.tex").write_text(header + "\n".join(lines) + "\n")
    print(f"wrote paper/macros.tex ({len(M)} macros)")


if __name__ == "__main__":
    main()
