"""Contract tests for the frame-selection maths (the stage-1 bug lived here)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.frames import (  # noqa: E402
    DEFAULT_HOSTS,
    all_preview_urls,
    frame_indices,
    frame_positions,
    max_index_from_urls,
    preview_set,
    preview_url,
)


def test_known_counts():
    assert frame_indices(15) == [0, 5, 10, 14]
    assert frame_indices(10) == [0, 3, 6, 9]
    assert frame_indices(16) == [0, 5, 10, 15]
    assert frame_indices(1) == [0, 0, 0, 0]
    assert frame_indices(2) == [0, 0, 1, 1]


def test_positions_are_named_and_ordered():
    pos = frame_positions(15)
    assert list(pos) == ['first', 'third', 'twothirds', 'last']
    assert pos['last'] == 14


def test_indices_are_monotonic_and_in_range():
    for n in list(range(1, 200)) + [512]:
        idx = frame_indices(n)
        assert idx == sorted(idx)
        assert idx[0] == 0 and idx[-1] == n - 1 and all(0 <= i < n for i in idx)


def test_rejects_bad_n():
    for bad in (0, -3):
        try:
            frame_indices(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'frame_indices({bad}) should raise')


def test_urls_match_the_documented_shape():
    urls = preview_set('7477551', 15)
    assert urls == [preview_url('7477551', i) for i in (0, 5, 10, 14)]
    for u in urls:
        assert '/videos/7477551/pictures/preview-' in u
        assert u.startswith('https://')
    # every documented host produces the same path shape
    for tpl in DEFAULT_HOSTS:
        u = preview_url('1', 2, tpl)
        assert '/videos/1/pictures/preview-2' in u


def test_all_previews_covers_every_index():
    assert all_preview_urls('5', 4) == [preview_url('5', i) for i in range(4)]


def test_max_index_from_urls():
    assert max_index_from_urls(['.../preview-0.jpeg', '.../preview-14.jpeg']) == 14
    assert max_index_from_urls([]) is None
    assert max_index_from_urls(['not a url']) is None


def test_contract_matches_pilot_and_cached_clips():
    """Clips verified by CDN probing (pilot, 2026-10-20) and by the cached frames we scored.

    This is the regression test for the stage-1 frame-index bug: the indices must come from
    the contract, never from the order files were cached in.
    """
    known = {
        '7477551': (15, [0, 5, 10, 14]),
        '10140736': (15, [0, 5, 10, 14]),
        '34982656': (15, [0, 5, 10, 14]),
        '11760274': (15, [0, 5, 10, 14]),
        '8456422': (15, [0, 5, 10, 14]),
        '35362440': (10, [0, 3, 6, 9]),
    }
    for clip, (n, idx) in known.items():
        assert frame_indices(n) == idx, clip
        assert sorted(int(u.rsplit('preview-', 1)[1].split('.')[0]) for u in preview_set(clip, n)) == idx
