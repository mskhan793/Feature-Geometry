"""Read-only: which candidate encoder interfaces the current Python environment supports (no weights loaded).

Reports library versions, whether transformers has the model classes the candidates' configs name, whether timm
registers Perception Encoder models, whether open_clip is importable, and which candidate repositories are in the
Hugging Face cache (``$HF_HOME/hub``) at which revisions.

Usage: check_encoder_support.py --out result.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

TRANSFORMERS_CLASSES = ("Dinov2Model", "DINOv3ViTModel", "SiglipModel", "SiglipVisionModel", "Siglip2Model",
                        "Aimv2VisionModel", "Aimv2Model", "CLIPImageProcessor")
CACHE_REPOS = ("imageomics/bioclip-2", "facebook/dinov2-large", "facebook/dinov3-vitl16-pretrain-lvd1689m",
               "facebook/dinov3-vith16plus-pretrain-lvd1689m", "google/siglip2-so400m-patch14-384",
               "google/siglip2-so400m-patch16-384", "google/siglip2-giant-opt-patch16-256",
               "google/siglip2-giant-opt-patch16-384", "facebook/PE-Core-G14-448",
               "facebook/vit_pe_core_gigantic_patch14_448_timm", "nvidia/C-RADIOv4-H", "apple/aimv2-huge-patch14-448")


def version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except Exception as e:  # recorded, not hidden
        return f"not importable: {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    report = {"versions": {m: version(m) for m in ("torch", "torchvision", "transformers", "timm", "open_clip", "huggingface_hub")}}
    try:
        import transformers
        report["transformers_classes"] = {c: hasattr(transformers, c) for c in TRANSFORMERS_CLASSES}
    except Exception as e:
        report["transformers_classes"] = f"error {e!r}"
    try:
        import timm
        report["timm_pe_models"] = timm.list_models("*pe_core*")
    except Exception as e:
        report["timm_pe_models"] = f"error {e!r}"
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    report["hf_hub_dir"] = str(hub)
    report["cached"] = {repo: sorted(p.name for p in (hub / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
                        for repo in CACHE_REPOS if (hub / f"models--{repo.replace('/', '--')}" / "snapshots").is_dir()}
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
