#!/usr/bin/env python3
"""Score a content-free manifest (clip keys + public preview URLs) with pinned
modesty/NSFW detectors and emit per-frame probabilities.

Models (revisions pinned; see README):
  - Freepik/nsfw_image_detector @ 15b85477e4fd2000db76ae9aae0f89a72f95e2e3  (MIT)
  - Marqo/nsfw-image-detection-384 @ 0c26ec22111b83f106d72a55f611ec35962bcb65 (Apache-2.0)
  - google/siglip2-base-patch16-224 @ 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
    (the model production uses; the label text comes from a repository secret and is
     never written to logs, results or artifacts)

SigLIP outputs, matching production (visual_director.py, label index 0):
  cosine  = normalised image embedding . normalised text embedding
  logit   = logit_scale * cosine + logit_bias      (logit_scale/bias from the model)
  prob    = sigmoid(logit)                          (this is what production thresholds)

Fails loudly: any unresolved URL, any non-image response, any missing frame, any
exception aborts the job with a non-zero exit code. No silent skips.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests  # preflight needs only this; torch/PIL are imported where used

from harness.siglip import (  # pure helpers, importable without torch
    LABEL_ENV,
    select_freepik_clips,
    MODEL_NAMES,
    SIGLIP_REPO,
    SIGLIP_REV,
    cross_check,
    label_banner,
    load_label,
    parse_models,
    per_clip_max,
    sigmoid_prob,
)

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
                from PIL import Image
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


def preflight(clips: list[dict], out: Path, fail_fraction: float) -> int:
    """HEAD every preview URL. No downloads, no API key. Writes out/preflight.json."""
    missing, total = [], 0
    for c in clips:
        for url in c['frames']:
            total += 1
            ok = False
            for candidate in _candidates(url):
                try:
                    r = requests.head(candidate, timeout=20, allow_redirects=True,
                                      headers={'User-Agent': 'census-preflight/1.0'})
                    if r.status_code == 200:
                        ok = True
                        break
                except Exception:
                    continue
            if not ok:
                missing.append({'key': c['key'], 'url': url})
    frac = (len(missing) / total) if total else 1.0
    out.mkdir(parents=True, exist_ok=True)
    (out / 'preflight.json').write_text(json.dumps(
        {'frames': total, 'missing': len(missing), 'missing_fraction': round(frac, 4),
         'missing_keys': sorted({m['key'] for m in missing})}, indent=1))
    print(f'preflight: {total - len(missing)}/{total} frames reachable '
          f'({frac:.2%} missing, limit {fail_fraction:.0%})', flush=True)
    if frac > fail_fraction:
        print('preflight FAILED: too many unreachable previews', flush=True)
        return 1
    return 0


def load_siglip(device: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, SiglipModel
    d = snapshot_download(SIGLIP_REPO, revision=SIGLIP_REV)
    proc = AutoProcessor.from_pretrained(d)
    model = SiglipModel.from_pretrained(d, dtype=torch.float32).eval().to(device)
    return proc, model


def score_siglip(items, proc, model, label, device):
    import torch
    from PIL import Image
    """Returns (per-frame list, seconds, meta). Batches the 4 frames of a clip."""
    import numpy as _np  # heavy import kept local; numpy ships with torch
    from collections import defaultdict, OrderedDict

    frames = OrderedDict()
    for key, url, path in items:
        frames.setdefault(key, []).append((url, path))

    with torch.inference_mode():
        tok = proc(text=[label], padding='max_length', truncation=True,
                   return_tensors='pt').to(device)
        text = model.get_text_features(**tok)
        text = text / text.norm(dim=-1, keepdim=True)

    logit_scale = float(model.logit_scale.exp().to('cpu'))
    logit_bias = float(model.logit_bias.to('cpu')) if model.logit_bias is not None else 0.0

    out = {}
    t0 = time.time()
    done = 0
    for key, pairs in frames.items():
        imgs = [Image.open(p).convert('RGB') for _, p in pairs]
        with torch.inference_mode():
            inp = proc(images=imgs, return_tensors='pt').to(device)
            emb = model.get_image_features(**inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            cos = (emb @ text.T).squeeze(-1).to('cpu').tolist()
        for (url, _), c in zip(pairs, cos):
            out[f'{key}|{url}'] = {'cosine': round(float(c), 6),
                                   'prob': round(sigmoid_prob(float(c), logit_scale, logit_bias), 6)}
        done += len(pairs)
        if done % 40 < 4 or done == len(items):
            el = time.time() - t0
            print(f'  siglip {done}/{len(items)} ({el:.0f}s, {el/done:.2f}s/frame)', flush=True)
    return out, time.time() - t0, {'logit_scale': round(logit_scale, 6),
                                   'logit_bias': round(logit_bias, 6)}






def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='out')
    ap.add_argument('--models', default='freepik,marqo,siglip')
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--chunks', type=int, default=1)
    ap.add_argument('--limit-clips', type=int, default=0,
                    help='score only the first N clips (calibration)')
    ap.add_argument('--report-timing-only', action='store_true',
                    help='write only seconds/frame and counts, no per-frame scores')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--fail-on-missing-fraction', type=float, default=0.30)
    ap.add_argument('--label-env', default='MODESTY_LABEL')
    ap.add_argument('--cascade-cos', type=float, default=None,
                    help='run Freepik only where cheap models cannot clear the clip')
    ap.add_argument('--cascade-marqo', type=float, default=0.9)
    ap.add_argument('--audit-fraction', type=float, default=0.20)
    ap.add_argument('--audit-salt', default='census-u')
    ap.add_argument('--video-budget-mb', type=float, default=100.0)
    ap.add_argument('--cross-check-ref', default='',
                    help="env var name holding a JSON map {model: {clip: value}}")
    ap.add_argument('--tolerance', type=float, default=1e-3)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    clips = man['clips']
    assert clips, 'empty manifest'
    if args.chunks > 1:
        clips = [c for i, c in enumerate(clips) if i % args.chunks == args.chunk]
        print(f'chunk {args.chunk}/{args.chunks}: {len(clips)} clips', flush=True)
    if args.limit_clips:
        clips = clips[:args.limit_clips]
    keys = [c['key'] for c in clips]
    assert len(keys) == len(set(keys)), 'duplicate clip keys in manifest'
    total = sum(len(c['frames']) for c in clips)
    out = Path(args.out)

    if args.preflight:
        return preflight(clips, out, args.fail_on_missing_fraction)

    import torch
    torch.set_num_threads(os.cpu_count() or 2)

    models = parse_models(args.models)
    work = Path('/tmp/frames')
    work.mkdir(parents=True, exist_ok=True)
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

    # 1b) optional video arm: download a capped set of clips, extract 2 fps frames, delete the video
    video_items, video_bytes = [], 0
    budget = args.video_budget_mb * 1024 * 1024
    for c in clips:
        v = c.get('video')
        if not v:
            continue
        dest = work / f"{c['key'].replace(':', '_')}.mp4"
        with requests.get(v['url'], stream=True, timeout=120,
                          headers={'User-Agent': 'census-video/1.0'}) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    video_bytes += len(chunk)
                    if video_bytes > budget:
                        raise RuntimeError(f'video budget exceeded at {c["key"]} '
                                           f'({video_bytes/1e6:.1f} MB > {args.video_budget_mb} MB)')
        outdir = work / f"vid_{c['key'].replace(':', '_')}"
        outdir.mkdir(exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(dest), '-vf', f"fps={v.get('fps', 2)}",
                        '-frames:v', str(v.get('max_frames', 20)), str(outdir / 'f_%03d.jpg')],
                       check=True)
        dest.unlink(missing_ok=True)  # never keep the video
        frames = sorted(outdir.glob('f_*.jpg'))
        if not frames:
            raise RuntimeError(f'no frames extracted for {c["key"]}')
        for i, fp in enumerate(frames):
            video_items.append((c['key'], f'video:{c["key"]}:{i}', fp))
    if video_items:
        items.extend(video_items)
        total = len(items)
        print(f'  video arm: {video_bytes/1e6:.1f} MB downloaded and deleted, '
              f'{len(video_items)} frames (total {total})', flush=True)

    result = {'manifest': args.manifest, 'models': models, 'clips': keys,
              'frames_total': total, 'frames_scored': len(items) + len(video_items)}
    per_model_clip = {}

    # 3) Marqo
    mq_scores, mq_elapsed = {}, 0.0
    if 'marqo' in models:
        print('loading Marqo', MARQO_REPO, MARQO_REV, flush=True)
        import torch
        from huggingface_hub import snapshot_download
        from PIL import Image
        import timm
        from safetensors.torch import load_file
        mq_dir = snapshot_download(MARQO_REPO, revision=MARQO_REV)
        mq = timm.create_model('vit_tiny_patch16_384', pretrained=False, num_classes=2).eval()
        mq.load_state_dict(load_file(os.path.join(mq_dir, 'model.safetensors')), strict=True)
        tf = timm.data.create_transform(**timm.data.resolve_model_data_config(mq), is_training=False)
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
        result['marqo'] = {'repo': MARQO_REPO, 'revision': MARQO_REV,
                           'labels': ['NSFW', 'SFW'],
                           's_per_frame': round(mq_elapsed / total, 4)}
        per_model_clip['marqo'] = {
            k: max(v[0] for kk, v in mq_scores.items() if kk.startswith(k + '|'))
            for k in keys}

    # 4) SigLIP (production model; label from secret)
    sg_scores, sg_elapsed, sg_meta = {}, 0.0, {}
    if 'siglip' in models:
        label = load_label(args.label_env)
        print(label_banner(label), flush=True)
        print('loading SigLIP', SIGLIP_REPO, SIGLIP_REV, flush=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        proc, model = load_siglip(device)
        sg_scores, sg_elapsed, sg_meta = score_siglip(items, proc, model, label, device)
        print(f'SIGLIP_S_PER_FRAME={sg_elapsed/total:.3f} frames={total}', flush=True)
        result['siglip'] = {'repo': SIGLIP_REPO, 'revision': SIGLIP_REV,
                            'label_source': f'secret:{args.label_env}',
                            'signal': 'sigmoid(logit_scale*cosine+logit_bias)',
                            **sg_meta,
                            's_per_frame': round(sg_elapsed / total, 4)}
        per_model_clip['siglip'] = {
            k: max(v['cosine'] for kk, v in sg_scores.items() if kk.startswith(k + '|'))
            for k in keys}
        result['siglip_probs'] = {
            k: max(v['prob'] for kk, v in sg_scores.items() if kk.startswith(k + '|'))
            for k in keys}

    # 2) Freepik (last when the cascade is on: it needs the cheap scores first)
    fp_scores, fp_elapsed, fp_labels = {}, 0.0, {}
    fp_items = items
    if 'freepik' in models and args.cascade_cos is not None:
        cos_by = per_clip_max(sg_scores, keys, 'cosine') if 'siglip' in models else {}
        mq_by = {}
        for k in keys:
            vals = [v[0] for kk, v in mq_scores.items() if kk.startswith(k + '|')]
            if vals:
                mq_by[k] = max(vals)
        sel = select_freepik_clips(cos_by, mq_by, args.cascade_cos, args.cascade_marqo,
                                   args.audit_fraction, args.audit_salt)
        run_set = set(sel['run'])
        fp_items = [it for it in items if it[0] in run_set]
        cascade = {'clips_total': len(cos_by), 'clips_freepik': len(sel['run']),
                   'clips_skipped': len(sel['skipped']), 'clips_audited': len(sel['audited']),
                   'cos_cut': args.cascade_cos, 'marqo_cut': args.cascade_marqo,
                   'audit_fraction': args.audit_fraction, 'audit_salt': args.audit_salt,
                   'frames_freepik': len(fp_items)}
        print('CASCADE ' + json.dumps(cascade), flush=True)

    if 'freepik' in models:
        print('loading Freepik', FREEPIK_REPO, FREEPIK_REV, flush=True)
        import torch
        from huggingface_hub import snapshot_download
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        fp_dir = snapshot_download(FREEPIK_REPO, revision=FREEPIK_REV)
        proc = AutoImageProcessor.from_pretrained(fp_dir)
        fp = AutoModelForImageClassification.from_pretrained(fp_dir, dtype=torch.float32).eval()
        fp_labels = {int(k): v for k, v in fp.config.id2label.items()}
        print('Freepik labels:', fp_labels, flush=True)
        t0 = time.time()
        for i, (key, url, path) in enumerate(fp_items, 1):
            im = Image.open(path).convert('RGB')
            with torch.inference_mode():
                logits = fp(**proc(images=[im], return_tensors='pt')).logits
            probs = torch.softmax(logits.float(), dim=-1)[0].tolist()
            fp_scores[f'{key}|{url}'] = [round(float(x), 6) for x in probs]
            if i % 25 == 0 or i == len(fp_items):
                el = time.time() - t0
                print(f'  freepik {i}/{len(fp_items)} ({el:.0f}s, {el/i:.2f}s/frame)', flush=True)
        fp_elapsed = time.time() - t0
        print(f'FREEPIK_S_PER_FRAME={fp_elapsed/max(1,len(fp_items)):.3f} frames={len(fp_items)}', flush=True)
        result['freepik'] = {'repo': FREEPIK_REPO, 'revision': FREEPIK_REV,
                             'labels': {str(k): v for k, v in fp_labels.items()},
                             's_per_frame': round(fp_elapsed / max(1, len(fp_items)), 4)}
        per_model_clip['freepik'] = {
            k: max(1.0 - v[0] for kk, v in fp_scores.items() if kk.startswith(k + '|'))
            for k in keys if any(kk.startswith(k + '|') for kk in fp_scores)}


    # 5) hard completeness check
    for c in clips:
        for url in c['frames']:
            k = f"{c['key']}|{url}"
            if 'freepik' in models and k in fp_scores:
                pass  # scored
            elif 'freepik' in models and args.cascade_cos is not None:
                pass  # skipped by the cascade (audited separately)
            elif 'freepik' in models:
                assert k in fp_scores, f'missing freepik {k}'
            if 'marqo' in models:
                assert k in mq_scores, f'missing marqo {k}'
            if 'siglip' in models:
                assert k in sg_scores, f'missing siglip {k}'

    if args.cascade_cos is not None and fp_items is not items:
        result['cascade'] = cascade

    # 6) cross-check against reference maxima (numbers only; no reference values echoed)
    if args.cross_check_ref:
        raw = os.environ.get(args.cross_check_ref, '')
        if not raw.strip():
            raise SystemExit(f'reference secret {args.cross_check_ref!r} is empty')
        refs = json.loads(raw)
        report = cross_check(per_model_clip, refs, args.tolerance)
        (out / 'cross_check.json').write_text(json.dumps(report, indent=1))
        print('CROSS_CHECK ' + json.dumps(report), flush=True)
        for m, r in report.items():
            if r['clips_compared'] and r['max_abs_delta'] is not None:
                print(f"  {m}: {r['clips_compared']} clips, max|delta|={r['max_abs_delta']}, "
                      f"outside tol: {r['n_outside_tol']}", flush=True)

    if args.report_timing_only:
        timing = {m: {k: v for k, v in result[m].items()
                      if k in ('repo', 'revision', 's_per_frame', 'logit_scale', 'logit_bias')}
                  for m in models if m in result}
        timing['frames_total'] = total
        timing['clips'] = len(keys)
        (out / 'calibration.json').write_text(json.dumps(timing, indent=1))
        print('wrote', out / 'calibration.json', flush=True)
        return 0

    result['elapsed_s'] = round(fp_elapsed + mq_elapsed + sg_elapsed, 1)
    result['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if 'freepik' in models:
        result['freepik']['frames'] = fp_scores
    if 'marqo' in models:
        result['marqo']['frames'] = mq_scores
    if 'siglip' in models:
        result['siglip']['frames'] = sg_scores
    dest = out / 'scores.json'
    dest.write_text(json.dumps(result, indent=1))
    print('wrote', dest, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
