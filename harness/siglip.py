"""SigLIP helpers that must stay importable without torch.

Production (projects/youtube-shorts/src/visual_director.py) thresholds
sigmoid(logit_scale * cosine + logit_bias) on label index 0. The census must reproduce
that exactly, keep the label text out of logs and artifacts, and pin the revision.
"""
from __future__ import annotations

import math
import os

SIGLIP_REPO = 'google/siglip2-base-patch16-224'
SIGLIP_REV = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
LABEL_ENV = 'MODESTY_LABEL'
MODEL_NAMES = ('freepik', 'marqo', 'siglip')


def parse_models(spec: str) -> list[str]:
    out = [m.strip().lower() for m in (spec or '').split(',') if m.strip()]
    if not out:
        raise SystemExit('--models is empty')
    bad = [m for m in out if m not in MODEL_NAMES]
    if bad:
        raise SystemExit(f'unknown model(s): {bad}; known: {list(MODEL_NAMES)}')
    return list(dict.fromkeys(out))


def load_label(env_name: str = LABEL_ENV) -> str:
    """Read the label text from the environment (a repository secret in CI).

    The value is never logged, stored in a result file, or hashed into anything that
    leaves the machine: only its length is ever reported.
    """
    val = os.environ.get(env_name, '')
    if not val.strip():
        raise SystemExit(
            f'label secret {env_name!r} is empty or missing; refusing to score with SigLIP')
    return val.strip()


def label_banner(label: str) -> str:
    """Log-safe description of the label: length only, never the text."""
    return f'siglip label loaded from secret (chars={len(label)})'


def sigmoid_prob(cosine: float, logit_scale: float, logit_bias: float) -> float:
    """Production's score: sigmoid(logit_scale * cosine + logit_bias)."""
    z = logit_scale * cosine + logit_bias
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def per_clip_max(scores: dict, keys: list[str], field: str) -> dict:
    out = {}
    for k in keys:
        vals = [v[field] for kk, v in scores.items() if kk.startswith(k + '|')]
        if vals:
            out[k] = max(vals)
    return out


def cross_check(computed: dict, refs: dict, tol: float) -> dict:
    """Compare per-clip maxima against reference values; return summary numbers only."""
    report = {}
    for model, vals in refs.items():
        got = computed.get(model, {})
        common = sorted(set(got) & set(vals))
        deltas = [abs(float(got[k]) - float(vals[k])) for k in common]
        bad = [k for k in common if abs(float(got[k]) - float(vals[k])) > tol]
        report[model] = {
            'clips_compared': len(common),
            'clips_reference_only': len(set(vals) - set(got)),
            'max_abs_delta': round(max(deltas), 6) if deltas else None,
            'n_outside_tol': len(bad),
            'tolerance': tol,
        }
    return report
