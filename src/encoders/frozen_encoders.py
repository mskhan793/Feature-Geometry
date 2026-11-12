"""The frozen encoders. The first three exactly as in Lightning_Aqua_Vision_Language_v1
``scripts/encode_frozen_backbone.py`` and ``scripts/run_bioscan.sh`` (model ids, preprocessing, output); SigLIP 2
added for the eight-dataset benchmark (task.md 8538df5; docs/evidence/benchmark_readiness/encoders/).

- ``bioclip2``: open_clip ``hf-hub:imageomics/bioclip-2``, its own eval preprocess, feature ``model.visual(x)``
  (open_clip ``encode_image`` without normalisation).
- ``dinov2_l``: transformers ``facebook/dinov2-large``, official DINOv2 eval transform (Resize 256 bicubic,
  CenterCrop 224, ImageNet mean/std), feature ``pooler_output`` = final-layernorm CLS token.
- ``dinov3_l``: transformers ``facebook/dinov3-vitl16-pretrain-lvd1689m``, official DINOv3 README transform for the
  LVD-1689M weights (ToImage, Resize((256, 256)), ToDtype, ImageNet mean/std; no crop), feature ``pooler_output``.
- ``siglip2``: transformers ``google/siglip2-so400m-patch16-256`` (the official 256 variant, preferred over 384 by the
  task's checkpoint policy), the checkpoint's own ``SiglipImageProcessor`` (256 x 256 bilinear, rescale, mean/std 0.5),
  feature ``SiglipVisionModel`` ``pooler_output`` = the attention-pooled image embedding that
  ``SiglipModel.get_image_features`` returns (SigLIP has no further projection).

Transformers checkpoints are loaded at the pinned ``REVISIONS``; BioCLIP 2 through open_clip's hf-hub loader, whose
cached ``refs/main`` is 2957b322 on LUMI (open_clip takes no revision argument).

Features are the raw model outputs (not unit-normalised). Weights are loaded from the Hugging Face cache; set
``HF_HUB_OFFLINE=1`` so that nothing is downloaded or silently substituted.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MODEL_IDS = {
    "bioclip2": "hf-hub:imageomics/bioclip-2",
    "dinov2_l": "facebook/dinov2-large",
    "dinov3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "siglip2": "google/siglip2-so400m-patch16-256",
}
REVISIONS = {"dinov2_l": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", "dinov3_l": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
             "siglip2": "e8708ab72d125807e45b36fb7d4e0aacbb59f379"}


class PooledOutput(nn.Module):
    """A transformers vision model whose output is ``pooler_output``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


class ProcessorTransform:
    """A Hugging Face image processor as a per-image transform (PIL image -> 3 x H x W tensor)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt").pixel_values[0]

    def __repr__(self) -> str:
        return f"ProcessorTransform({self.processor!r})"


def load_encoder(name: str) -> tuple[nn.Module, Callable, dict]:
    """Return ``(encoder, transform, info)``; ``encoder(batch)`` gives raw features, ``info`` records provenance."""
    if name not in MODEL_IDS:
        raise ValueError(f"unknown encoder {name}; known: {sorted(MODEL_IDS)}")
    model_id = MODEL_IDS[name]
    if name == "bioclip2":
        import open_clip

        model, _, transform = open_clip.create_model_and_transforms(model_id)
        encoder, library = model.visual, f"open_clip {open_clip.__version__}"
        feature = "open_clip model.visual(x), not normalised"
    else:
        import transformers
        from torchvision import transforms as T
        from torchvision.transforms import v2

        revision = REVISIONS[name]
        feature = "pooler_output (final-norm CLS token)"
        if name == "dinov2_l":  # v1 'dinov2_eval': classic transforms, PIL resize before ToTensor
            transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        elif name == "dinov3_l":  # v1 'dinov3_lvd'
            transform = v2.Compose([v2.ToImage(), v2.Resize((256, 256), antialias=True),
                                    v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        if name == "siglip2":
            processor = transformers.AutoImageProcessor.from_pretrained(model_id, revision=revision)
            transform = ProcessorTransform(processor)
            model = transformers.SiglipVisionModel.from_pretrained(model_id, revision=revision)
            feature = "SiglipVisionModel pooler_output (attention-pooled; = get_image_features), not normalised"
        else:
            model = transformers.AutoModel.from_pretrained(model_id, revision=revision)
        encoder = PooledOutput(model)
        library = f"transformers {transformers.__version__} @ {revision}"
    encoder.eval().requires_grad_(False)
    info = {"encoder": name, "model_id": model_id, "library": library, "feature": feature,
            "transform": repr(transform), "torch": torch.__version__}
    return encoder, transform, info
