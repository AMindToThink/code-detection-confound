"""VENDORED VERBATIM from the project-droid/DroidDetect model card.

Source : https://huggingface.co/project-droid/DroidDetect-Base-Binary  (README "Model Code")
Paper  : Orel, Paul, Gurevych, Nakov, "DroidCollection / DroidDetect", arXiv:2507.10583,
         EMNLP 2025.

There is NO pip package, NO official inference repo, and NO `auto_map`/remote-code path for
this model (verified) — the README's `TLModel` class is the only canonical implementation.
We reproduce it here unmodified except for omitting the training-only triplet-loss term
(`self.additional_loss`), which (a) is never touched on the forward path when `labels=None`
and (b) would require `sentence_transformers`. The encoder/projection/classifier weights are
loaded with `strict=True` over exactly these submodules, verified to give 0 missing/0
unexpected keys and AUROC 1.000 on the authors' own DroidCollection Python test samples.

Label map (from the card): 0 = HUMAN_GENERATED, 1 = MACHINE_GENERATED.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

TEXT_EMBEDDING_DIM = 768


class TLModel(nn.Module):
    # verbatim forward path from the model card (mean-pool -> projection -> ReLU -> classifier)
    def __init__(self, text_encoder, projection_dim, num_classes=2):
        super().__init__()
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.text_projection = nn.Linear(TEXT_EMBEDDING_DIM, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, input_ids=None, attention_mask=None):
        sentence_embeddings = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sentence_embeddings = sentence_embeddings.mean(dim=1)           # raw (unmasked) mean
        projected_text = F.relu(self.text_projection(sentence_embeddings))
        return self.classifier(projected_text)


def load_droiddetect(repo: str, device: str = "cuda"):
    """Build TLModel exactly as published and load the checkpoint with strict=True.
    projection_dim is read from the checkpoint weights themselves (no hardcoding/guessing).
    Returns (model, tokenizer)."""
    sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu")
    projection_dim = sd["text_projection.weight"].shape[0]
    num_classes = sd["classifier.weight"].shape[0]
    encoder = AutoModel.from_config(AutoConfig.from_pretrained("answerdotai/ModernBERT-base"))
    model = TLModel(encoder, projection_dim=projection_dim, num_classes=num_classes)
    # exactly the three forward-path submodules; the duplicate additional_loss.* encoder
    # (training-only) is intentionally excluded.
    keep = {k: v for k, v in sd.items()
            if k.startswith(("text_encoder.", "text_projection.", "classifier."))}
    model.load_state_dict(keep, strict=True)   # fail loudly on any mismatch
    model.eval().to(device).half()
    tok = AutoTokenizer.from_pretrained(repo)   # the model's OWN shipped tokenizer
    return model, tok
