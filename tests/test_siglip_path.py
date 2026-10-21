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
    cos, scale, bias = 0.5, 10.0, -4.0
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
    computed = {'siglip': {'pexels:1': 0.5000, 'pexels:2': 0.4900}}
    refs = {'siglip': {'pexels:1': 0.5010, 'pexels:2': 0.4900, 'pexels:3': 0.02}}
    rep = score.cross_check(computed, refs, 1e-3)
    assert rep['siglip']['clips_compared'] == 2
    assert rep['siglip']['clips_reference_only'] == 1
    assert rep['siglip']['n_outside_tol'] == 1  # |0.5000-0.5010| > 1e-3
    assert rep['siglip']['max_abs_delta'] == pytest.approx(0.0010, abs=1e-6)
    assert '0.5010' not in str(rep)  # no reference values echoed


def test_per_clip_max_uses_all_frames():
    scores = {'pexels:1|a': {'cosine': 0.01}, 'pexels:1|b': {'cosine': 0.09},
              'pexels:2|a': {'cosine': 0.02}}
    got = score.per_clip_max(scores, ['pexels:1', 'pexels:2'], 'cosine')
    assert got == {'pexels:1': 0.09, 'pexels:2': 0.02}


def test_cascade_never_skips_a_flagged_clip():
    cos = {'a': 0.01, 'b': 0.09, 'c': 0.02}
    mq = {'a': 0.1, 'b': 0.2, 'c': 0.95}
    out = score.select_freepik_clips(cos, mq, 0.03, 0.9, 0.0, 'salt')
    assert 'b' in out['run'] and 'c' in out['run']  # cos or marqo clears the cut
    assert out['skipped'] == ['a']


def test_cascade_fails_closed_on_missing_score():
    out = score.select_freepik_clips({'a': 0.01, 'b': None}, {'a': 0.1, 'b': 0.1},
                                     0.03, 0.9, 0.0, 'salt')
    assert 'b' in out['run']


def test_audit_is_deterministic_and_bounded():
    keys = [f'pexels:{i}' for i in range(200)]
    picks = {k for k in keys if score.audit_selected(k, 'census-u', 0.2)}
    assert picks == {k for k in keys if score.audit_selected(k, 'census-u', 0.2)}
    assert 20 <= len(picks) <= 60  # ~20% of 200, generous band for one salt
