"""Manifest construction for the modesty-gate evaluation harness.

Public-repo hygiene is enforced here, not by convention:

* a manifest row is ``{"clip": "pexels:<id>", "query": "qNN", "frames": [urls]}``
  and nothing else — no labels, marks, strata, scores, dates, channel text;
* the ``used`` arm (clips the pipeline has already taken) and the ``candidate``
  arm (clips it could draw) are merged into **one flat, deterministically
  shuffled list** with no field that tells them apart;
* query strings become opaque ids (``q01`` …) and the mapping stays local.

See ``tests/test_manifest.py`` for the properties this module promises.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Iterable, Mapping, Sequence

CLIP_RE = re.compile(r'^(pexels|pixabay):\d+$')
QID_RE = re.compile(r'^q\d{2,}$')
URL_RE = re.compile(r'^https://(images|static-videos)\.pexels\.com/videos/\d+/pictures/preview-\d+\.(jpeg|jpg)$')

#: Words that must never appear anywhere in a public manifest.
FORBIDDEN_WORDS = (
    'modest', 'hijab', 'pray', 'muslim', 'bikini', 'nsfw', 'swimwear',
    'zaitoon', 'channel', 'label', 'mark', 'stratum', 'used', 'candidate',
    'holdout', 'freepik', 'marqo', 'siglip', 'score',
)


def opaque_qids(queries: Iterable[str], width: int = 2) -> dict[str, str]:
    """Map query strings to opaque ids in sorted order (stable across runs)."""
    return {q: f'q{i + 1:0{width}d}' for i, q in enumerate(sorted(set(queries)))}


def make_rows(clips: Sequence[str], query_of: Mapping[str, str], qids: Mapping[str, str],
              frames_of: Mapping[str, Sequence[str]] | None = None) -> list[dict]:
    """Build flat rows for one arm. ``clips`` may be used or candidate — identical shape."""
    frames_of = frames_of or {}
    rows = []
    for c in clips:
        rows.append({'clip': c, 'query': qids[query_of[c]], 'frames': list(frames_of.get(c, []))})
    return rows


def flatten(arms: Mapping[str, Sequence[str]], query_of: Mapping[str, str],
            qids: Mapping[str, str], frames_of: Mapping[str, Sequence[str]] | None = None,
            seed: int = 20261020) -> list[dict]:
    """Merge every arm into one shuffled list; the arm name never leaves this function."""
    rows: list[dict] = []
    for _, clips in arms.items():
        rows.extend(make_rows(clips, query_of, qids, frames_of))
    random.Random(seed).shuffle(rows)
    return rows


def assert_hygiene(rows: Sequence[dict]) -> None:
    """Raise ValueError if a row could leak anything or reveal the arm it came from."""
    allowed = {'clip', 'query', 'frames'}
    for i, r in enumerate(rows):
        extra = set(r) - allowed
        if extra:
            raise ValueError(f'row {i}: unexpected fields {sorted(extra)} (arm leakage risk)')
        if not CLIP_RE.match(str(r['clip'])):
            raise ValueError(f'row {i}: clip key is not <source>:<digits>')
        if not QID_RE.match(str(r['query'])):
            raise ValueError(f'row {i}: query is not an opaque id (raw query text leaked)')
        for u in r['frames']:
            if not URL_RE.match(str(u)):
                raise ValueError(f'row {i}: frame URL is not a public preview URL')
            low = str(u).lower()
            if any(w in low for w in FORBIDDEN_WORDS):
                raise ValueError(f'row {i}: forbidden word in URL')


def manifest_bytes(rows: Sequence[dict], frames_per_clip: int = 4) -> bytes:
    """Canonical JSON bytes for hashing and upload."""
    doc = {'note': 'preview pictures only; identities and arm membership intentionally omitted',
           'frames_per_clip': frames_per_clip, 'clips': rows}
    return json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def query_coverage(rows: Sequence[dict]) -> dict[str, int]:
    """Clips per opaque query id — the only query-level statistic that may be logged."""
    out: dict[str, int] = {}
    for r in rows:
        out[r['query']] = out.get(r['query'], 0) + 1
    return dict(sorted(out.items()))
