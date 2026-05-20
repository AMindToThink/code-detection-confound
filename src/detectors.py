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
class _DroidModel(torch.nn.Module):
    """Faithful reconstruction of project-droid's TLModel inference path.

    forward: emb = mean_pool(encoder(x).last_hidden_state)      [B, 768]
             proj = relu(text_projection(emb))                  [B, 256]
             logits = classifier(proj)                          [B, 2]
    Label map (from the model card): 0 = HUMAN_GENERATED, 1 = MACHINE_GENERATED.
    NOTE: projection_dim is 256 in the *checkpoint weights* (config.json's 128 is stale).
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        self.text_encoder = AutoModel.from_config(cfg)
        self.text_projection = torch.nn.Linear(768, 256)
        self.classifier = torch.nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        h = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = h.mean(dim=1)
        proj = F.relu(self.text_projection(emb))
        return self.classifier(proj)


class DroidDetect:
    name = "DroidDetect"

    def __init__(self) -> None:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(C.DROIDDETECT_REPO)
        st = [f for f in files if f.endswith(".safetensors")]
        if st:
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(C.DROIDDETECT_REPO, st[0]))
        else:
            wf = next(f for f in files if f.endswith(".bin"))
            sd = torch.load(hf_hub_download(C.DROIDDETECT_REPO, wf), map_location="cpu")
        keep = {k: v for k, v in sd.items()
                if k.startswith(("text_encoder.", "text_projection.", "classifier."))}
        self.model = _DroidModel()
        missing, unexpected = self.model.load_state_dict(keep, strict=False)
        # the only allowed "missing" are buffers ModernBERT recreates; assert head loaded
        assert not any("classifier" in m or "projection" in m for m in missing), missing
        self.model.eval().to(DEVICE).half()
        self.tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    @torch.no_grad()
    def score_batch(self, codes: list[str], bs: int = 32) -> list[float]:
        out: list[float] = []
        for i in range(0, len(codes), bs):
            chunk = codes[i:i + bs]
            enc = self.tok(chunk, return_tensors="pt", truncation=True,
                           max_length=MAX_TOKENS, padding=True).to(DEVICE)
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            p_machine = F.softmax(logits.float(), dim=-1)[:, 1]   # P(class 1 = MACHINE)
            out.extend(p_machine.cpu().tolist())
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
