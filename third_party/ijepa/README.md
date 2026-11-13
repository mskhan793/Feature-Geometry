# I-JEPA — authors' original model code (unmodified)

- Source: https://github.com/facebookresearch/ijepa, commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
  (2023-06-13), files `src/models/vision_transformer.py` and its two imports `src/utils/tensors.py`,
  `src/masks/utils.py`, in the same layout.
- Licence: CC BY-NC 4.0 (`LICENSE`, copied unchanged); the released weights are under the same terms. Use here is
  non-commercial research.
- Paper: M. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
  CVPR 2023, arXiv:2301.08243 (local copy `~/Documents/Research/methods/ijepa_assran2023/paper.pdf`).
- Checkpoint (task.md 3b897b0): `https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar` (ViT-H/14, 224 px,
  ImageNet-1K, 300 epochs; config `configs/in1k_vith14_ep300.yaml`: `model_name: vit_huge`, patch 14).

| File | sha256 |
|---|---|
| `src/models/vision_transformer.py` | `69be453a6367460fd0c94588af6531c022a2469ff72e0a793c16fd110dd498c0` |
| `src/utils/tensors.py` | `66cac858005d01e93ad6f481f384a9d75cb1fb8ad2bff0fd7e9a127f384429c0` |
| `src/masks/utils.py` | `124da04a7513329a14793389d89a98db87498caf7623a10bc772b097c74d97ec` |
| `LICENSE` | `336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf` |

The files are not edited. `vision_transformer.py` imports `src.utils.tensors` and `src.masks.utils`, which would
collide with this repository's own `src` package; `src/encoders/ijepa.py` therefore executes the three files as
separate in-memory modules and resolves those two imports to them (no text is changed; hashes checked at load).
