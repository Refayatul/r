# nsfw-score — content-free scorer for the modesty gate

A small, self-contained CLI that scores images for "not modest enough for the channel"
against a pinned image classifier, so the heavy work happens on CI instead of on a laptop.

It exists because the gate decision has to be reproducible and auditable: the same pinned
model revision, the same frame-selection contract, the same numbers, on a machine that
never holds the video files themselves. Everything this repo sees is a public preview URL.

## Layout

| path | what it is |
| :-- | :-- |
| `score_nsfw.py` | the scorer: reads `manifest.json`, downloads each preview picture, writes per-frame probabilities |
| `harness/frames.py` | the frame-selection contract (`0, n/3, 2n/3, n-1` of `n` preview pictures) and the public preview-URL shapes |
| `harness/manifest.py` | flat-manifest construction: opaque query ids, one shuffled list, hygiene assertions |
| `tests/` | unit tests for both harness modules (`pytest -q`) |
| `.github/workflows/nsfw-score.yml` | the existing scoring job (runs 1–5, 108 clips / 431 frames) |
| `.github/workflows/tests.yml` | runs the unit tests on every push |
| `manifest.json` | the current job: `{clip key, public preview URLs}` — nothing else |

## Models (pinned)

| role | model | revision |
| :-- | :-- | :-- |
| Freepik NSFW detector | `Freepik/nsfw_image_detector` | `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` |
| Marqo NSFW detector | `Marqo/nsfw-image-detection-384` | `0c26ec22111b83f106d72a55f611ec35962bcb65` |

## Contract

* a manifest row is exactly `{"clip": "<source>:<id>", "query": "qNN", "frames": [urls]}`;
  `harness.manifest.assert_hygiene` fails the build if any other field appears, if a query
  is not opaque, or if a frame URL is not a public Pexels preview picture;
* the clips the pipeline has already used and the clips it could draw are merged into **one
  shuffled list** — the manifest must not say which is which;
* frame indices are `0, n/3, 2n/3, n-1` for `n` preview pictures, never the order files
  happen to be cached in (that mistake cost us one full re-run in October);
* logs are aggregate only: counts and timings, never per-clip scores.

## Running the tests

```bash
python -m pytest -q tests
```
