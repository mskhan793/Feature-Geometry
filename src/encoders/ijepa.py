"""Frozen I-JEPA ViT-H/14 target encoder from the authors' unmodified code (third_party/ijepa, commit 52c1ae9; task.md
3b897b0).

- Model: ``vit_huge(img_size=[224], patch_size=14)`` exactly as the authors' ``src/helper.init_model`` builds it for
  ``configs/in1k_vith14_ep300.yaml``; weights = the checkpoint's ``target_encoder`` (paper App. A.1: "We use the
  target-encoder for evaluation and average pool its output"), stored once as safetensors by
  scripts/benchmark/ijepa_target_weights.py (DDP ``module.`` prefix removed; loaded strictly).
- Feature: mean over the 256 patch tokens of the encoder's output (its forward ends with the final LayerNorm), i.e.
  the "average-pooled patch representation of the last layer" (paper App. A.2), 1280-d, not normalised further. No
  masks, no predictor, no [cls] token (the model has none).
- The three source files are executed as separate in-memory modules; their ``src.utils.tensors`` /
  ``src.masks.utils`` imports are resolved to those modules (this repository's own ``src`` package is never touched),
  after checking every file's sha256 against third_party/ijepa/README.md.
"""

from __future__ import annotations

import builtins
import functools
import hashlib
import os
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2] / "third_party" / "ijepa" / "src"
SOURCES = {"src.utils.tensors": ("utils/tensors.py", "66cac858005d01e93ad6f481f384a9d75cb1fb8ad2bff0fd7e9a127f384429c0"),
           "src.masks.utils": ("masks/utils.py", "124da04a7513329a14793389d89a98db87498caf7623a10bc772b097c74d97ec"),
           "src.models.vision_transformer": ("models/vision_transformer.py",
                                             "69be453a6367460fd0c94588af6531c022a2469ff72e0a793c16fd110dd498c0")}
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
WEIGHTS = os.environ.get("FG_IJEPA_WEIGHTS", "/scratch/project_462001534/feature_geometry/weights/ijepa/"
                                             "IN1K-vit.h.14-300e_target_encoder.safetensors")


def _source(name: str) -> str:
    rel, sha = SOURCES[name]
    raw = (ROOT / rel).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise RuntimeError(f"third_party/ijepa/src/{rel} differs from the authors' pinned file")
    return raw.decode()


@functools.cache
def official_module() -> ModuleType:
    """The authors' vision_transformer module, its two imports resolved to the vendored helper modules."""
    helpers = {}
    for name in ("src.utils.tensors", "src.masks.utils"):
        m = ModuleType("ijepa_" + name.replace(".", "_"))
        exec(compile(_source(name), str(ROOT / SOURCES[name][0]), "exec"), m.__dict__)
        helpers[name] = m

    def resolve(name, globals=None, locals=None, fromlist=(), level=0):
        return helpers[name] if name in helpers else builtins.__import__(name, globals, locals, fromlist, level)

    vt = ModuleType("ijepa_vision_transformer")
    vt.__dict__["__builtins__"] = {**builtins.__dict__, "__import__": resolve}
    exec(compile(_source("src.models.vision_transformer"), str(ROOT / SOURCES["src.models.vision_transformer"][0]),
                 "exec"), vt.__dict__)
    return vt


def build_vith14() -> nn.Module:
    return official_module().vit_huge(img_size=[224], patch_size=14)


class MeanPatchTokens(nn.Module):
    """Mean of the final-layer (LayerNorm-ed) patch tokens of the I-JEPA encoder."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values).mean(dim=1)


def load_target_encoder(weights: str | Path = WEIGHTS) -> tuple[nn.Module, str]:
    """The pooled target encoder with the stored weights (strict load) and the weights file's sha256."""
    from safetensors.torch import load_file

    vit = build_vith14()
    vit.load_state_dict(load_file(str(weights)), strict=True)
    sha = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    return MeanPatchTokens(vit), sha
