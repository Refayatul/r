"""SigLIP path tests: pinned revision, secret label, production sigmoid, log hygiene."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness.siglip as score  # noqa: E402  (pure helpers only: no torch import)


def test_siglip_revision_is_pinned():
    # production ran unpinned; the census must not
    assert len(score.SIGLIP_REV) == 40
    assert all(c in '0123456789abcdef' for c in score.SIGLIP_REV)
    assert score.SIGLIP_REV != 'main'


def test_siglip_outputs_match_production_formula():
    # production: sigmoid(logit_scale * cosine + logit_bias)
    cos, scale, bias = 0.08, 4.724453, -16.771725
    expected = 1 / (1 + math.exp(-(scale * cos + bias)))
    assert score.sigmoid_prob(cos, scale, bias) == pytest.approx(expected, rel=0, abs=1e-12)
    # monotone in cosine and bounded
    lo = score.sigmoid_prob(0.0, scale, bias)
    hi = score.sigmoid_prob(0.5, scale, bias)
    assert 0.0 < lo < hi < 1.0


def test_label_comes_from_env_and_is_required(monkeypatch):
    monkeypatch.delenv('MODESTY_LABEL', raising=False)
    with pytest.raises(SystemExit) as exc:
        score.load_label('MODESTY_LABEL')
    assert 'MODESTY_LABEL' in str(exc.value)
    monkeypatch.setenv('MODESTY_LABEL', '  a private label  ')
    assert score.load_label('MODESTY_LABEL') == 'a private label'


def test_label_is_never_logged(monkeypatch, capsys):
    secret = 'super-secret-label-text'
    monkeypatch.setenv('MODESTY_LABEL', secret)
    label = score.load_label('MODESTY_LABEL')
    print(score.label_banner(label))
    out = capsys.readouterr().out
    assert secret not in out
    assert 'chars=' in out


def test_parse_models_rejects_unknown_and_dedupes():
    assert score.parse_models('siglip,freepik,siglip') == ['siglip', 'freepik']
    with pytest.raises(SystemExit):
        score.parse_models('freepik,nsfw-net')
    with pytest.raises(SystemExit):
        score.parse_models(' , ')


def test_cross_check_reports_summary_only():
    computed = {'siglip': {'pexels:1': 0.0800, 'pexels:2': 0.0400}}
    refs = {'siglip': {'pexels:1': 0.0817, 'pexels:2': 0.0400, 'pexels:3': 0.02}}
    rep = score.cross_check(computed, refs, 1e-3)
    assert rep['siglip']['clips_compared'] == 2
    assert rep['siglip']['clips_reference_only'] == 1
    assert rep['siglip']['n_outside_tol'] == 1  # |0.0800-0.0817| > 1e-3
    assert rep['siglip']['max_abs_delta'] == pytest.approx(0.0017, abs=1e-6)
    assert '0.0817' not in str(rep)  # no reference values echoed


def test_per_clip_max_uses_all_frames():
    scores = {'pexels:1|a': {'cosine': 0.01}, 'pexels:1|b': {'cosine': 0.09},
              'pexels:2|a': {'cosine': 0.02}}
    got = score.per_clip_max(scores, ['pexels:1', 'pexels:2'], 'cosine')
    assert got == {'pexels:1': 0.09, 'pexels:2': 0.02}
