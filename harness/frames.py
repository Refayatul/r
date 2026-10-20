"""Frame-selection contract for the modesty-gate evaluation harness.

Production picks four preview pictures per clip at positions
``0, n/3, 2n/3, n-1`` of the ``n`` available preview pictures. The stage-1
frame-index bug came from assuming the cached file order equalled the preview
index, so this module is the single place where that maths lives and it is
covered by tests.

Preview pictures are public CDN images (``video_pictures[].picture`` in the
Pexels video resource). Only the host/extension differ between what the CDN
serves today and what the pipeline cached; the path shape does not:

    https://images.pexels.com/videos/<id>/pictures/preview-<k>.jpeg
    https://static-videos.pexels.com/videos/<id>/pictures/preview-<k>.jpg
"""

from __future__ import annotations

from typing import Iterable

DEFAULT_HOSTS = (
    'https://images.pexels.com/videos/{clip}/pictures/preview-{index}.jpeg',
    'https://images.pexels.com/videos/{clip}/pictures/preview-{index}.jpg',
    'https://static-videos.pexels.com/videos/{clip}/pictures/preview-{index}.jpg',
)

#: Names of the four contract positions, in order.
POSITION_NAMES = ('first', 'third', 'twothirds', 'last')


def frame_indices(n: int) -> list[int]:
    """Return the four contract indices for a clip with ``n`` preview pictures.

    >>> frame_indices(15)
    [0, 5, 10, 14]
    >>> frame_indices(10)
    [0, 3, 6, 9]
    >>> frame_indices(1)
    [0, 0, 0, 0]
    """
    if n < 1:
        raise ValueError('n must be >= 1')
    return [0, n // 3, (2 * n) // 3, n - 1]


def frame_positions(n: int) -> dict[str, int]:
    """Same as :func:`frame_indices`, keyed by position name."""
    return dict(zip(POSITION_NAMES, frame_indices(n)))


def preview_url(clip: str | int, index: int, template: str = DEFAULT_HOSTS[0]) -> str:
    return template.format(clip=clip, index=index)


def preview_set(clip: str | int, n: int, template: str = DEFAULT_HOSTS[0]) -> list[str]:
    """The four contract URLs for one clip."""
    return [preview_url(clip, i, template) for i in frame_indices(n)]


def all_preview_urls(clip: str | int, n: int, template: str = DEFAULT_HOSTS[0]) -> list[str]:
    """Every preview picture of a clip (used by the sensitivity arm)."""
    return [preview_url(clip, i, template) for i in range(n)]


def max_index_from_urls(urls: Iterable[str]) -> int | None:
    """Largest ``preview-<k>`` index present in a list of URLs, or None."""
    import re

    best = None
    for u in urls:
        m = re.search(r'preview-(\d+)', str(u))
        if m:
            k = int(m.group(1))
            best = k if best is None else max(best, k)
    return best
