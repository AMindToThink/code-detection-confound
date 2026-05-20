"""VENDORED VERBATIM from the official Fast-DetectGPT implementation.

Source : https://github.com/baoguangsheng/fast-detect-gpt
File   : scripts/fast_detect_gpt.py  (function get_sampling_discrepancy_analytic)
Paper  : Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-
         Generated Text via Conditional Probability Curvature", ICLR 2024.
License: MIT (Guangsheng Bao).

Reproduced unmodified so the detector uses the authors' canonical estimator rather
than a re-derivation. Caller must pass logits/labels already shifted for next-token
prediction (logits[:, :-1], labels = input_ids[:, 1:]), exactly as the upstream
experiment script does.
"""
import torch


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()
