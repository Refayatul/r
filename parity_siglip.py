#!/usr/bin/env python3
"""Parity: 14 audit anchors + 20 used clips under transformers 5.20.1 / 5.20.2.

Exact production contract for the cosine modesty cut:
  model  = google/siglip2-base-patch16-224 (pinned revision)
  label  = MODESTY_LABEL (secret; never printed)
  frames = up to 4 public Pexels preview stills per clip
  cos    = (logit_scale * cosine + logit_bias - logit_bias) / logit_scale
  max cosine per clip over its frames

The two versions score the same downloaded frames. Logs are aggregate only;
per-clip values go to the artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
ANCHORS = [
    "pexels:6275797", "pexels:34410295", "pexels:11048627", "pexels:8788897",
    "pexels:9015701", "pexels:35202010", "pexels:7477551", "pexels:10140736",
    "pexels:35362440", "pexels:7361599", "pexels:20450431", "pexels:11666242",
    "pexels:38883117", "pexels:8165554",
]
EXTRA = [
    "pexels:10082474", "pexels:14940677", "pexels:19594913", "pexels:27462147",
    "pexels:30493727", "pexels:31815449", "pexels:33453211", "pexels:34452563",
    "pexels:34796460", "pexels:35081872", "pexels:35461323", "pexels:36271956",
    "pexels:36629871", "pexels:37741486", "pexels:39536271", "pexels:4727543",
    "pexels:6317855", "pexels:7249050", "pexels:8165167", "pexels:8488725",
]
CLIPS = ANCHORS + EXTRA
FRAMES = Path("frames")
OUT = Path("out")


def _download(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45, headers={"User-Agent": "siglip-parity/1.0"})
            if r.status_code != 200:
                return False
            dest.write_bytes(r.content)
            from PIL import Image
            with Image.open(dest) as im:
                im.load()
                if min(im.size) < 64:
                    dest.unlink(missing_ok=True)
                    return False
            return True
        except Exception:
            time.sleep(1 + attempt)
    return False


def fetch() -> None:
    FRAMES.mkdir(exist_ok=True)
    missing = []
    for clip in CLIPS:
        vid = clip.split(":", 1)[1]
        got = 0
        for idx in range(0, 20):
            if got >= 4:
                break
            dest = FRAMES / f"{vid}_{idx}.jpg"
            if dest.exists():
                got += 1
                continue
            for ext in ("jpeg", "jpg"):
                if _download(f"https://images.pexels.com/videos/{vid}/pictures/preview-{idx}.{ext}", dest):
                    got += 1
                    break
            else:
                continue
        if got < 4:
            missing.append((clip, got))
        print(f"frames {clip}: {got}", flush=True)
    if missing:
        raise SystemExit(f"insufficient frames: {missing}")


def score(out_path: Path) -> None:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    label = os.environ.get("MODESTY_LABEL", "").strip()
    if not label:
        raise SystemExit("MODESTY_LABEL is empty")
    print(f"transformers={transformers.__version__} torch={torch.__version__}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).to("cpu").eval()
    commit = getattr(model.config, "_commit_hash", None) or getattr(processor, "_commit_hash", None)
    print(f"model_revision={commit}", flush=True)
    scale = float(model.logit_scale.exp().item())
    bias = float(model.logit_bias.item())
    result = {"transformers": transformers.__version__, "torch": torch.__version__,
              "model_revision": commit, "model_id": MODEL_ID, "clips": {}}
    with torch.no_grad():
        for clip in CLIPS:
            vid = clip.split(":", 1)[1]
            paths = sorted(FRAMES.glob(f"{vid}_*.jpg"))[:4]
            if len(paths) < 4:
                raise SystemExit(f"missing frames for {clip}")
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = processor(text=[label], images=images, padding="max_length",
                               return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits_per_image[:, 0]
            cos = [(float(x) - bias) / scale for x in logits]
            result["clips"][clip] = {"max_cos": max(cos), "frames": len(cos)}
            print(f"scored {clip}", flush=True)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path}", flush=True)


def compare(a_path: Path, b_path: Path, out_path: Path) -> None:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    deltas = {k: abs(a["clips"][k]["max_cos"] - b["clips"][k]["max_cos"])
              for k in a["clips"] if k in b["clips"]}
    print(f"clips compared: {len(deltas)}")
    print(f"max_abs_delta: {max(deltas.values()):.3e}")
    print(f"outside_1e-3: {sum(v > 1e-3 for v in deltas.values())}")
    print(f"a_revision: {a.get('model_revision')} a_transformers: {a.get('transformers')}")
    print(f"b_revision: {b.get('model_revision')} b_transformers: {b.get('transformers')}")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(
        {"transformers_a": a["transformers"], "transformers_b": b["transformers"],
         "revision_a": a.get("model_revision"), "revision_b": b.get("model_revision"),
         "clips": len(deltas), "max_abs_delta": max(deltas.values()),
         "outside_1e-3": sum(v > 1e-3 for v in deltas.values()),
         "deltas": deltas}, indent=1))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fetch", "score", "compare"), required=True)
    ap.add_argument("--out", type=Path, default=OUT / "scores.json")
    ap.add_argument("--a", type=Path, default=OUT / "scores_5201.json")
    ap.add_argument("--b", type=Path, default=OUT / "scores_5202.json")
    args = ap.parse_args()
    if args.mode == "fetch":
        fetch()
    elif args.mode == "score":
        score(args.out)
    else:
        compare(args.a, args.b, args.out)
