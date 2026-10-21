"""Hygiene and arm-indistinguishability tests for public manifest construction."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.manifest import (  # noqa: E402
    assert_hygiene,
    flatten,
    make_rows,
    manifest_bytes,
    opaque_qids,
    query_coverage,
    sha256,
)

QUERIES = ['q08', 'q09', 'q10']
QIDS = opaque_qids(QUERIES)
URLS = ['https://images.pexels.com/videos/1/pictures/preview-0.jpeg']


def _arms():
    return {'used': ['pexels:1', 'pexels:2'], 'candidate': ['pexels:3', 'pexels:4']}


def test_opaque_ids_hide_query_text():
    assert list(QIDS.values()) == ['q01', 'q02', 'q03']
    assert all(k not in QIDS.values() for k in QUERIES)


def test_rows_have_only_the_three_allowed_fields():
    qmap = {c: QUERIES[0] for c in ['pexels:1', 'pexels:2', 'pexels:3', 'pexels:4']}
    rows = flatten(_arms(), qmap, QIDS, {'pexels:1': URLS})
    assert rows and all(set(r) == {'clip', 'query', 'frames'} for r in rows)
    assert_hygiene(rows)


def test_used_and_candidate_rows_are_indistinguishable():
    qmap = {c: QUERIES[1] for c in ['pexels:1', 'pexels:2', 'pexels:3', 'pexels:4']}
    used_only = flatten({'used': ['pexels:1', 'pexels:2']}, qmap, QIDS)
    cand_only = flatten({'candidate': ['pexels:3', 'pexels:4']}, qmap, QIDS)
    all_rows = flatten(_arms(), qmap, QIDS)
    assert [set(r) for r in used_only] == [set(r) for r in cand_only]
    assert len(all_rows) == len(used_only) + len(cand_only)
    assert {r['clip'] for r in all_rows} == {'pexels:1', 'pexels:2', 'pexels:3', 'pexels:4'}


def test_flatten_is_deterministic_for_a_seed():
    qmap = {c: QUERIES[0] for c in ['pexels:1', 'pexels:2', 'pexels:3', 'pexels:4']}
    a = [r['clip'] for r in flatten(_arms(), qmap, QIDS, seed=7)]
    b = [r['clip'] for r in flatten(_arms(), qmap, QIDS, seed=7)]
    c = [r['clip'] for r in flatten(_arms(), qmap, QIDS, seed=8)]
    assert a == b and a != c


def test_flatten_does_not_mutate_the_arm_lists():
    arms = _arms()
    qmap = {c: QUERIES[0] for c in ['pexels:1', 'pexels:2', 'pexels:3', 'pexels:4']}
    flatten(arms, qmap, QIDS)
    assert arms == {'used': ['pexels:1', 'pexels:2'], 'candidate': ['pexels:3', 'pexels:4']}


@pytest.mark.parametrize('bad_row,message', [
    ({'clip': 'pexels:1', 'query': 'q01', 'frames': [], 'mark': 'x'}, 'unexpected fields'),
    ({'clip': '1', 'query': 'q01', 'frames': []}, 'clip key'),
    ({'clip': 'pexels:1', 'query': 'two words', 'frames': []}, 'opaque id'),
    ({'clip': 'pexels:1', 'query': 'q01', 'frames': ['/home/kai/frame.jpg']}, 'public preview URL'),
    ({'clip': 'pexels:1', 'query': 'q01', 'frames': ['https://evil.example/x.jpg']}, 'public preview URL'),
])
def test_hygiene_rejects_leaks(bad_row, message):
    with pytest.raises(ValueError) as e:
        assert_hygiene([bad_row])
    assert message in str(e.value)


def test_hygiene_rejects_foreign_keys_and_text():
    for extra in ('source', 'arm', 'used', 'pool', 'stratum', 'score'):
        with pytest.raises(ValueError):
            assert_hygiene([{'clip': 'pexels:1', 'query': 'q01', 'frames': [], extra: 1}])


def test_manifest_bytes_are_stable_and_hashable():
    qmap = {'pexels:1': QUERIES[0]}
    rows = make_rows(['pexels:1'], qmap, QIDS, {'pexels:1': URLS})
    b1, b2 = manifest_bytes(rows), manifest_bytes(rows)
    assert b1 == b2 and sha256(b1) == sha256(b2)
    assert b'"clips"' in b1
    # no field that could reveal which arm a row came from
    for leak in (b'"arm"', b'"used"', b'"candidate"', b'"score"', b'"mark"', b'"label"'):
        assert leak not in b1


def test_query_coverage_is_opaque():
    q_first = QIDS[QUERIES[0]]
    q_last = QIDS[QUERIES[2]]
    qmap = {c: QUERIES[0] for c in ['pexels:1', 'pexels:2']}
    qmap['pexels:3'] = QUERIES[2]
    rows = flatten({'used': ['pexels:1'], 'candidate': ['pexels:2', 'pexels:3']}, qmap, QIDS)
    cov = query_coverage(rows)
    assert cov[q_first] == 2 and cov[q_last] == 1
    assert all(k.startswith('q') for k in cov)
    assert not any(q in str(cov) for q in QUERIES)
