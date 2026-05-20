"""AI-code detectors. Dataset-agnostic: each detector maps a code string -> a scalar
score oriented so that HIGHER == MORE LIKELY AI (machine-generated).

Three detectors spanning the two families the experiment contrasts:
  * DroidDetect  (trained family)      -- project-droid/DroidDetect-Base-Binary
  * FastDetectGPT (statistical family) -- analytic conditional-prob discrepancy
  * Binoculars   (statistical family)  -- cross-perplexity ratio

The two statistical detectors share one forward pass of the performer/scoring model
(Qwen2.5-1.5B) and one of the observer (Qwen2.5-0.5B), both of which share a tokenizer.
"""
from __future__ import annotations

from src import _env  # noqa: F401  (must precede transformers import; torchvision shim)

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from src import config as C
from src.vendor.binoculars_metrics import entropy as canon_entropy
from src.vendor.binoculars_metrics import perplexity as canon_perplexity
from src.vendor.fast_detect_gpt import get_sampling_discrepancy_analytic as canon_fast_detect

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 1024            # truncation for speed; code rarely needs more for this signal


# ====================================================================== DroidDetect
# Uses the authors' VERBATIM TLModel class (src/vendor/droiddetect_model.py), loaded with
# strict=True. Verified: AUROC 1.000 / 1.3% human-FPR on the authors' own DroidCollection
# Python test samples. DroidDetect's training/eval truncation length is 512 (paper Table 10).
DROID_MAX_TOKENS = 512


class DroidDetect:
    name = "DroidDetect"

    def __init__(self) -> None:
        from src.vendor.droiddetect_model import load_droiddetect
        self.model, self.tok = load_droiddetect(C.DROIDDETECT_REPO, device=DEVICE)

    @torch.no_grad()
    def score_batch(self, codes: list[str], bs: int = 1) -> list[float]:
        # Score per-sample (no padding): the model's mean-pool is UNMASKED, so padding
        # tokens would contaminate the pooled vector. One sequence at a time keeps the
        # pool over real tokens only — matching single-example inference.
        out: list[float] = []
        for code in codes:
            enc = self.tok(code, return_tensors="pt", truncation=True,
                           max_length=DROID_MAX_TOKENS).to(DEVICE)
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            out.append(F.softmax(logits.float(), dim=-1)[0, 1].item())   # P(MACHINE)
        return out


# ============================================================ statistical detectors
# The scoring math lives entirely in the canonical vendored modules
# (src/vendor/fast_detect_gpt.py, src/vendor/binoculars_metrics.py). This class only
# loads the base models, runs forward passes, and orients the scores.
class StatDetectors:
    """Loads observer (0.5B) + performer/scorer (1.5B), shared Qwen2.5 tokenizer."""

    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(C.BINOCULARS_OBSERVER)
        self.observer = AutoModelForCausalLM.from_pretrained(
            C.BINOCULARS_OBSERVER, torch_dtype=torch.float16).eval().to(DEVICE)
        self.performer = AutoModelForCausalLM.from_pretrained(
            C.BINOCULARS_PERFORMER, torch_dtype=torch.float16).eval().to(DEVICE)

    @torch.no_grad()
    def score_one(self, code: str) -> dict[str, float]:
        """Score one code string with the CANONICAL vendored estimators.

        FastDetectGPT: get_sampling_discrepancy_analytic(ref=performer, score=performer)
                       on next-token-shifted logits/labels (single-model analytic mode).
        Binoculars   : perplexity(performer) / entropy(observer, performer); we return
                       the negation so HIGHER == MORE AI.
        """
        enc = self.tok(code, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        ids = enc["input_ids"]
        if ids.shape[1] < 3:
            return {"FastDetectGPT": float("nan"), "Binoculars": float("nan")}
        perf_logits = self.performer(ids, attention_mask=enc["attention_mask"]).logits.float()
        obs_logits = self.observer(ids, attention_mask=enc["attention_mask"]).logits.float()

        # --- FastDetectGPT (canonical): shift exactly as upstream caller does
        fd = canon_fast_detect(perf_logits[:, :-1], perf_logits[:, :-1], ids[:, 1:])

        # --- Binoculars (canonical metrics)
        from transformers import BatchEncoding
        be = BatchEncoding({"input_ids": ids, "attention_mask": enc["attention_mask"]})
        ppl = canon_perplexity(be, perf_logits)
        x_ppl = canon_entropy(obs_logits, perf_logits, be, self.tok.pad_token_id
                              if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        bino = float(ppl[0] / x_ppl[0])      # lower => AI
        return {"FastDetectGPT": fd, "Binoculars": -bino}   # negate => higher == more AI
