"""A.1 — length-only logistic-regression baseline on HMCorp.

If a model with only `len(code)` features (chars + BPE tokens via UniXcoder
tokenizer) achieves high AUROC on HMCorp valid, then HMCorp is fundamentally
length-confounded and the published 0.999 AUC is uninformative about whether the
detector learned anything semantic.

Pre-registered thresholds (from the revised plan doc, 2026-05-24):
  AUROC ≥ 0.95 → length-confounded; length-controlled eval becomes primary metric.
  0.80 ≤ AUROC < 0.95 → ambiguous; flag in paper caveats.
  AUROC < 0.80 → length not sufficient classifier; main concern resolved.

Fit on TRAIN, evaluate on VALID (TEST is held out). Per language.
Reports AUROC + 95% bootstrap CI + per-feature univariate AUC as drill-down.

Usage:
  python3 -u scripts/19_length_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401  torchvision shim for tokenizer load

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from transformers import RobertaTokenizer  # noqa: E402

BASE = "microsoft/unixcoder-base-nine"
N_BOOT = 1000
SEED = 0


def load_jsonl(path: Path) -> tuple[list[str], np.ndarray]:
    codes, labels = [], []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            codes.append(r["code"])
            labels.append(int(r["label"]))
    return codes, np.asarray(labels, dtype=np.int64)


def feats(codes: list[str], tok) -> np.ndarray:
    """Two features per row: char-length, BPE-token-length (matching how the model
    sees the input). Whitespace-collapse first (mirroring their
    convert_examples_to_features) so the char count reflects what the model ingests."""
    out = np.zeros((len(codes), 2), dtype=np.float64)
    for i, c in enumerate(codes):
        collapsed = " ".join(c.split())
        out[i, 0] = float(len(collapsed))
        out[i, 1] = float(len(tok.tokenize(collapsed)))
    return out


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n: int = N_BOOT,
                  seed: int = SEED) -> tuple[float, float, float]:
    """Mean AUC + 95% pivot bootstrap CI."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.asarray(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUROC using a single feature as the score. Direction-corrected:
    if increasing x predicts y=0 (human, label 0) the raw AUC is < 0.5, so flip."""
    auc = roc_auc_score(y, x)
    return float(auc if auc >= 0.5 else 1.0 - auc)


def verdict(auc: float) -> str:
    if auc >= 0.95:
        return "LENGTH-CONFOUNDED  (length-controlled eval becomes PRIMARY metric)"
    if auc >= 0.80:
        return "AMBIGUOUS          (flag in paper caveats; proceed but watch)"
    return "OK                 (length not a sufficient classifier)"


def run_lang(lang: str, tok) -> dict:
    train_p = ROOT / "data" / "hmcorp" / lang / "train.jsonl"
    valid_p = ROOT / "data" / "hmcorp" / lang / "valid.jsonl"
    print(f"\n=== {lang} ===", flush=True)
    print(f"  loading train: {train_p}", flush=True)
    train_codes, train_y = load_jsonl(train_p)
    print(f"  loading valid: {valid_p}", flush=True)
    valid_codes, valid_y = load_jsonl(valid_p)
    print(f"  train n={len(train_y)} (h={int((train_y==0).sum())}/m={int((train_y==1).sum())})  "
          f"valid n={len(valid_y)} (h={int((valid_y==0).sum())}/m={int((valid_y==1).sum())})",
          flush=True)

    # Featurize (BPE-token-length is the slow step; ~1ms/row)
    print("  featurizing train …", flush=True)
    X_train = feats(train_codes, tok)
    print("  featurizing valid …", flush=True)
    X_valid = feats(valid_codes, tok)

    # Per-label descriptive stats — verify the Table-5 length gap is present
    for lbl, name in [(0, "human"), (1, "machine")]:
        m = train_y == lbl
        print(f"  {name:7s}: chars mean={X_train[m,0].mean():7.1f} median={np.median(X_train[m,0]):7.1f}  "
              f"bpe mean={X_train[m,1].mean():6.1f} median={np.median(X_train[m,1]):6.1f}",
              flush=True)

    # ---- two-feature logistic regression ----
    pipe = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, solver="lbfgs",
                                               class_weight="balanced", random_state=SEED))])
    pipe.fit(X_train, train_y)
    prob_valid = pipe.predict_proba(X_valid)[:, 1]
    auc, lo, hi = bootstrap_auc(valid_y, prob_valid)
    print(f"  2-feature LR AUROC (valid) = {auc:.4f}  [95% CI {lo:.4f}, {hi:.4f}]", flush=True)

    # ---- per-feature univariate ----
    auc_chars = univariate_auc(X_valid[:, 0], valid_y)
    auc_bpe = univariate_auc(X_valid[:, 1], valid_y)
    print(f"  univariate chars-only AUROC = {auc_chars:.4f}", flush=True)
    print(f"  univariate bpe-only   AUROC = {auc_bpe:.4f}", flush=True)

    # ---- coefficients ----
    coef = pipe.named_steps["lr"].coef_[0]
    intercept = pipe.named_steps["lr"].intercept_[0]
    print(f"  LR coef (scaled): chars={coef[0]:+.3f}  bpe={coef[1]:+.3f}  intercept={intercept:+.3f}",
          flush=True)

    v = verdict(auc)
    print(f"  VERDICT: {v}", flush=True)
    return {
        "lang": lang,
        "n_train": int(len(train_y)),
        "n_valid": int(len(valid_y)),
        "auc_2feat": auc, "auc_2feat_ci": [lo, hi],
        "auc_chars_only": auc_chars,
        "auc_bpe_only": auc_bpe,
        "coef": [float(coef[0]), float(coef[1])],
        "intercept": float(intercept),
        "verdict": v,
        "chars_mean_human":   float(X_train[train_y == 0, 0].mean()),
        "chars_mean_machine": float(X_train[train_y == 1, 0].mean()),
        "bpe_mean_human":     float(X_train[train_y == 0, 1].mean()),
        "bpe_mean_machine":   float(X_train[train_y == 1, 1].mean()),
    }


def main() -> None:
    tok = RobertaTokenizer.from_pretrained(BASE)
    results = {lang: run_lang(lang, tok) for lang in ("python", "java")}
    out_p = ROOT / "results" / "phase_a" / "length_baseline.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_p}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    for lang, r in results.items():
        print(f"  {lang}: AUROC {r['auc_2feat']:.4f}  -> {r['verdict']}", flush=True)


if __name__ == "__main__":
    main()
