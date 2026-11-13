"""Store the I-JEPA ViT-H/14 checkpoint's target encoder as safetensors and record its identity (task.md 3b897b0 gate:
checkpoint access and weight identity).

First recomputes the publisher's S3 multipart ETag of the downloaded file (MD5 of the concatenated MD5s of 8 MiB
parts, "-<parts>"; the server reports b0e94c3423adcbda5a2108a6c1544263-1235) and requires it to equal ``--etag``:
this checks the bytes against the server's own record. Then reads the official checkpoint with ``torch.load(weights_only=True)`` (no arbitrary pickle code), records its top-level
keys, epoch, sha256 (computed here) and the per-part tensor counts; removes the DistributedDataParallel ``module.``
prefix from ``target_encoder``; loads it strictly into the authors' ``vit_huge(img_size=[224], patch_size=14)``
(src/encoders/ijepa.py: every key and shape must match); saves the state dict as safetensors and its sha256. Also
records parameter counts and the largest difference between the target and context encoders (they are different
exponential-moving-average states; equality would mean a mislabelled part).

Usage: ijepa_target_weights.py --checkpoint IN1K-vit.h.14-300e.pth.tar --etag ETAG --out WEIGHTS.safetensors
       --record RECORD.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.encoders.ijepa import build_vith14


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def s3_etag(path: Path, part: int = 8 * 2 ** 20) -> str:
    digests = []
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(part), b""):
            digests.append(hashlib.md5(block).digest())
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def strip(sd: dict) -> dict:
    return {k.removeprefix("module."): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--etag", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    a = ap.parse_args()
    etag = s3_etag(a.checkpoint)
    if etag != a.etag:
        raise SystemExit(f"downloaded file's multipart ETag {etag} != server ETag {a.etag}")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    target, context = strip(ck["target_encoder"]), strip(ck["encoder"])
    vit = build_vith14()
    vit.load_state_dict(target, strict=True)  # raises on any missing, unexpected or mis-shaped key
    tensors = {k: v.contiguous() for k, v in vit.state_dict().items()}
    assert all(torch.equal(tensors[k], target[k]) for k in target), "loaded weights differ from the checkpoint"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(a.out))
    diff = max(float((target[k].float() - context[k].float()).abs().max()) for k in target)
    rec = {"checkpoint": str(a.checkpoint), "checkpoint_bytes": a.checkpoint.stat().st_size,
           "checkpoint_sha256": sha256(a.checkpoint), "checkpoint_s3_etag_recomputed": etag,
           "server_etag": a.etag, "top_level_keys": sorted(ck),
           "epoch": ck.get("epoch"), "parts_tensor_counts": {k: len(v) for k, v in ck.items() if isinstance(v, dict)},
           "target_encoder_tensors": len(target), "target_encoder_parameters": int(sum(v.numel() for v in target.values())),
           "dtypes": sorted({str(v.dtype) for v in target.values()}), "strict_load": "ok",
           "max_abs_diff_target_vs_context_encoder": diff, "out": str(a.out), "out_sha256": sha256(a.out),
           "torch": torch.__version__}
    a.record.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
