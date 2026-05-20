"""Property tests for the CANONICAL vendored detector estimators.

We validate the upstream (vendored) functions directly: (1) Fast-DetectGPT uses
sum-over-token aggregation (numerically distinct from a mean variant), (2) both
detectors are oriented so that "more predictable / machine-like" scores higher AI.
"""
import math

import torch
from transformers import BatchEncoding

from src.vendor.binoculars_metrics import entropy as canon_entropy
from src.vendor.binoculars_metrics import perplexity as canon_perplexity
from src.vendor.fast_detect_gpt import get_sampling_discrepancy_analytic as canon_fd


def _ref_fast_detect_sum(logits, ids):
    """Independent reference: explicit per-position loops, SUM aggregation (official)."""
    lg = logits[0, :-1, :].double()
    labels = ids[0, 1:]
    lp = torch.log_softmax(lg, dim=-1)
    p = torch.softmax(lg, dim=-1)
    L = lg.shape[0]
    ll = sum(lp[t, labels[t]].item() for t in range(L))
    mu = sum((p[t] * lp[t]).sum().item() for t in range(L))
    var = sum(((p[t] * lp[t] ** 2).sum() - (p[t] * lp[t]).sum() ** 2).item() for t in range(L))
    return (ll - mu) / math.sqrt(var)


def _wrong_fast_detect_mean(logits, ids):
    """Plausible-but-wrong: mean over tokens. Differs from sum version by ~sqrt(L)."""
    lg = logits[0, :-1, :].double()
    labels = ids[0, 1:]
    lp = torch.log_softmax(lg, dim=-1)
    p = torch.softmax(lg, dim=-1)
    L = lg.shape[0]
    ll = torch.stack([lp[t, labels[t]] for t in range(L)]).mean().item()
    e1 = torch.stack([(p[t] * lp[t]).sum() for t in range(L)])
    e2 = torch.stack([(p[t] * lp[t] ** 2).sum() for t in range(L)])
    return (ll - e1.mean().item()) / math.sqrt((e2 - e1 ** 2).mean().item())


def test_canonical_fastdetect_is_sum_aggregation():
    """The vendored canonical estimator matches the SUM reference and is numerically
    distinct from the mean variant (so the test actually pins the aggregation)."""
    torch.manual_seed(0)
    V, L = 7, 14
    logits = torch.randn(1, L, V)
    ids = torch.randint(0, V, (1, L))
    canon = canon_fd(logits[:, :-1], logits[:, :-1], ids[:, 1:])
    ref_sum = _ref_fast_detect_sum(logits, ids)
    ref_mean = _wrong_fast_detect_mean(logits, ids)
    assert abs(canon - ref_sum) < 1e-4, (canon, ref_sum)
    assert abs(ref_sum - ref_mean) > 0.5, (ref_sum, ref_mean)


def test_canonical_fastdetect_confident_scores_higher():
    """Higher discrepancy == more AI: a too-predictable sequence scores higher."""
    torch.manual_seed(2)
    V, L = 50, 30
    ids = torch.randint(0, V, (1, L))
    conf = torch.full((1, L, V), -5.0)
    for t in range(L - 1):
        conf[0, t, ids[0, t + 1]] = 12.0
    rand = torch.randn(1, L, V)
    s_conf = canon_fd(conf[:, :-1], conf[:, :-1], ids[:, 1:])
    s_rand = canon_fd(rand[:, :-1], rand[:, :-1], ids[:, 1:])
    assert s_conf > s_rand, (s_conf, s_rand)


def test_canonical_binoculars_orientation():
    """Lower raw B (ppl/x_ppl) == machine; our wrapper negates so confident performer
    => higher returned score. Here we check the raw B is smaller for the confident case."""
    torch.manual_seed(1)
    V, L = 50, 20
    ids = torch.randint(0, V, (1, L))
    mask = torch.ones(1, L, dtype=torch.long)
    be = BatchEncoding({"input_ids": ids, "attention_mask": mask})
    obs = torch.randn(1, L, V)

    perf_conf = torch.full((1, L, V), -5.0)
    for t in range(L - 1):
        perf_conf[0, t, ids[0, t + 1]] = 10.0
    B_conf = float(canon_perplexity(be, perf_conf)[0] / canon_entropy(obs, perf_conf, be, -1)[0])

    perf_diffuse = torch.randn(1, L, V) * 0.1
    B_diffuse = float(canon_perplexity(be, perf_diffuse)[0] / canon_entropy(obs, perf_diffuse, be, -1)[0])

    assert B_conf < B_diffuse, (B_conf, B_diffuse)   # confident => lower B => more AI
