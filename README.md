# r — content-free scoring jobs

This repo exists only so heavy compute can run on GitHub Actions without touching the
owner's laptop and without ever holding content-bearing data.

Current job: `nsfw-score` (`.github/workflows/nsfw-score.yml`).

## Content-free guarantee

The only input is `manifest.json`:

```json
{"note": "...", "frames_per_clip": {"pexels:123": 4},
 "clips": [{"key": "pexels:123", "frames": ["https://images.pexels.com/videos/123/pictures/preview-0.jpeg"]}]}
```

It contains clip keys and public preview URLs — nothing else. No owner labels or marks,
no strata, no sealed keys, no API keys, no channel-identifying text, no frame images.
The output artifact contains per-frame model probabilities keyed by clip key + URL, and
is joined to labels locally.

## Pinned models

| model | revision | license |
|---|---|---|
| `Freepik/nsfw_image_detector` | `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` | MIT |
| `Marqo/nsfw-image-detection-384` | `0c26ec22111b83f106d72a55f611ec35962bcb65` | Apache-2.0 |

Dependencies are pinned in the workflow (torch 2.14.1 CPU, transformers 5.20.1,
timm 1.0.30, huggingface_hub 2.2.0, safetensors 0.8.0, pillow 13.0.0, requests 2.32.3).

## Run

```bash
gh workflow run nsfw-score -R Refayatul/r -f manifest=manifest.json
gh run watch <run-id> -R Refayatul/r
gh run download <run-id> -R Refayatul/r -n nsfw-scores
```

## Failure policy

Any non-200 / non-image response, any frame shorter than 1 kB, any exception, or any
clip missing a frame aborts the job with a non-zero exit code. There are no silent
skips and no fail-open paths. The job prints `FREEPIK_S_PER_FRAME` and
`MARQO_S_PER_FRAME` so runner throughput is recorded with the artifacts.
