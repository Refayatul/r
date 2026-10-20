#!/usr/bin/env python3
"""Score a content-free manifest (clip keys + public preview URLs) with two pinned
NSFW detectors and emit per-frame probabilities.

Models (revisions pinned; see README):
  - Freepik/nsfw_image_detector @ 15b85477e4fd2000db76ae9aae0f89a72f95e2e3  (MIT)
  - Marqo/nsfw-image-detection-384 @ 0c26ec22111b83f106d72a55f611ec35962bcb65 (Apache-2.0)

Fails loudly: any unresolved URL, any non-image response, any missing frame, any
exception aborts the job with a non-zero exit code. No silent skips.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import torch
from huggingface_hub import snapshot_download
from PIL import Image

FREEPIK_REPO = 'Freepik/nsfw_image_detector'
FREEPIK_REV = '15b85477e4fd2000db76ae9aae0f89a72f95e2e3'
MARQO_REPO = 'Marqo/nsfw-image-detection-384'
MARQO_REV = '0c26ec22111b83f106d72a55f611ec35962bcb65'


def _candidates(url: str) -> list[str]:
    base = url.rsplit('.', 1)[0]
    out = [url]
    for ext in ('jpeg', 'jpg', 'png'):
        alt = f'{base}.{ext}'
        if alt not in out:
            out.append(alt)
    return out


def fetch(url: str, dest: Path) -> None:
    """Download and validate one frame. Never skips: raises if no candidate decodes."""
    reasons = []
    for candidate in _candidates(url):
        for attempt in range(3):
            try:
                r = requests.get(candidate, timeout=60, headers={'User-Agent': 'nsfw-score/1.0'})
                if r.status_code != 200:
                    reasons.append(f'{candidate}: HTTP {r.status_code}')
                    time.sleep(2 * (attempt + 1))
                    continue
                ctype = r.headers.get('Content-Type', '')
                if not ctype.startswith('image/'):
                    reasons.append(f'{candidate}: content-type {ctype!r}')
                    time.sleep(2 * (attempt + 1))
                    continue
                dest.write_bytes(r.content)
                with Image.open(dest) as im:
                    im.load()
                    w, h = im.size
                if min(w, h) < 64:
                    reasons.append(f'{candidate}: too small {w}x{h}')
                    break
                return
            except Exception as e:  # network/decode error: retry, never silently skip
                reasons.append(f'{candidate}: {e!r}')
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'failed to fetch {url}: ' + '; '.join(reasons[-4:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 2)
    man = json.load(open(args.manifest))
    clips = man['clips']
    assert clips, 'empty manifest'
    keys = [c['key'] for c in clips]
    assert len(keys) == len(set(keys)), 'duplicate clip keys in manifest'
    total = sum(len(c['frames']) for c in clips)

    work = Path('/tmp/frames')
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) download every frame (hard fail on any problem)
    items = []  # (clip key, url, local path), in manifest order
    t0 = time.time()
    n = 0
    for c in clips:
        for url in c['frames']:
            p = work / f"{c['key'].replace(':', '_')}__{url.rsplit('/', 1)[-1]}"
            fetch(url, p)
            items.append((c['key'], url, p))
            n += 1
            if n % 50 == 0 or n == total:
                print(f'  downloaded {n}/{total} ({time.time()-t0:.0f}s)', flush=True)
    assert n == total, (n, total)

    # 2) Freepik
    print('loading Freepik', FREEPIK_REPO, FREEPIK_REV, flush=True)
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    fp_dir = snapshot_download(FREEPIK_REPO, revision=FREEPIK_REV)
    proc = AutoImageProcessor.from_pretrained(fp_dir)
    fp = AutoModelForImageClassification.from_pretrained(fp_dir, dtype=torch.float32).eval()
    fp_labels = {int(k): v for k, v in fp.config.id2label.items()}
    print('Freepik labels:', fp_labels, flush=True)

    fp_scores = {}
    t0 = time.time()
    for i, (key, url, path) in enumerate(items, 1):
        im = Image.open(path).convert('RGB')
        with torch.inference_mode():
            logits = fp(**proc(images=[im], return_tensors='pt')).logits
        probs = torch.softmax(logits.float(), dim=-1)[0].tolist()
        fp_scores[f'{key}|{url}'] = [round(float(x), 6) for x in probs]
        if i % 25 == 0 or i == total:
            el = time.time() - t0
            print(f'  freepik {i}/{total} ({el:.0f}s, {el/i:.2f}s/frame)', flush=True)
    fp_elapsed = time.time() - t0
    print(f'FREEPIK_S_PER_FRAME={fp_elapsed/total:.3f} frames={total}', flush=True)

    # 3) Marqo
    print('loading Marqo', MARQO_REPO, MARQO_REV, flush=True)
    import timm
    from safetensors.torch import load_file
    mq_dir = snapshot_download(MARQO_REPO, revision=MARQO_REV)
    mq = timm.create_model('vit_tiny_patch16_384', pretrained=False, num_classes=2).eval()
    mq.load_state_dict(load_file(os.path.join(mq_dir, 'model.safetensors')), strict=True)
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(mq), is_training=False)

    mq_scores = {}
    t0 = time.time()
    for i, (key, url, path) in enumerate(items, 1):
        im = Image.open(path).convert('RGB')
        with torch.inference_mode():
            probs = torch.softmax(mq(tf(im).unsqueeze(0)), -1)[0].tolist()
        mq_scores[f'{key}|{url}'] = [round(float(x), 6) for x in probs]
        if i % 50 == 0 or i == total:
            el = time.time() - t0
            print(f'  marqo {i}/{total} ({el:.0f}s, {el/i:.2f}s/frame)', flush=True)
    mq_elapsed = time.time() - t0
    print(f'MARQO_S_PER_FRAME={mq_elapsed/total:.3f} frames={total}', flush=True)

    # 4) hard completeness check
    assert len(fp_scores) == total, (len(fp_scores), total)
    assert len(mq_scores) == total, (len(mq_scores), total)
    for c in clips:
        for url in c['frames']:
            k = f"{c['key']}|{url}"
            assert k in fp_scores and k in mq_scores, f'missing {k}'

    result = {
        'freepik': {'repo': FREEPIK_REPO, 'revision': FREEPIK_REV,
                    'labels': {str(k): v for k, v in fp_labels.items()},
                    's_per_frame': round(fp_elapsed / total, 4),
                    'frames': fp_scores},
        'marqo': {'repo': MARQO_REPO, 'revision': MARQO_REV,
                  'labels': ['NSFW', 'SFW'],
                  's_per_frame': round(mq_elapsed / total, 4),
                  'frames': mq_scores},
        'manifest': args.manifest,
        'manifest_sha256': os.environ.get('MANIFEST_SHA256'),
        'clips': keys,
        'frames_total': total,
        'elapsed_s': round(fp_elapsed + mq_elapsed, 1),
    }
    dest = out / 'scores.json'
    dest.write_text(json.dumps(result, indent=1))
    print('wrote', dest, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
