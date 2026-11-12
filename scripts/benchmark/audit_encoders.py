"""Audit the four frozen encoders of the benchmark (task.md 8538df5) from their Hugging Face repositories at pinned
revisions: revision sha, licence / gating, parameter count and weight size (API metadata), architecture and
preprocessing (config files only; no weights are downloaded), the feature this project extracts, whether that feature is
L2-normalised, and whether a final LayerNorm makes it rank-deficient by construction. Gated configs (DINOv3) are read
from the local cache. The LayerNorm statement is the expectation from the architecture; it is confirmed empirically
per dataset on extracted features (scripts/audit_feature_null_space.py).

Usage: audit_encoders.py --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ENCODERS = {
    "dinov3_l": {"repo": "facebook/dinov3-vitl16-pretrain-lvd1689m", "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
                 "files": ["config.json", "preprocessor_config.json"], "library": "transformers DINOv3ViTModel",
                 "feature": "pooler_output = final-LayerNorm CLS token (existing aquatic caches)",
                 "preprocessing_used": "ToImage, Resize((256, 256)), ImageNet mean/std, no crop (src/encoders/frozen_encoders.py)",
                 "layernorm_rank_deficit_expected": True,
                 "note": "rank 1023 of 1024 measured on AquaMonitor and FIN-Benthic2 caches (native_null_space audit)"},
    "dinov2_l": {"repo": "facebook/dinov2-large", "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                 "files": ["config.json", "preprocessor_config.json"], "library": "transformers Dinov2Model",
                 "feature": "pooler_output = final-LayerNorm CLS token",
                 "preprocessing_used": "Resize 256 bicubic, CenterCrop 224, ImageNet mean/std (v1 transform)",
                 "layernorm_rank_deficit_expected": True, "note": "same final-norm construction as DINOv3; to measure"},
    "bioclip2": {"repo": "imageomics/bioclip-2", "revision": "2957b322090f9cb17ae72c71981c7218a28d81e0",
                 "files": ["open_clip_config.json"], "library": "open_clip (hf-hub:imageomics/bioclip-2)",
                 "feature": "model.visual(x) = proj(ln_post(CLS)), not L2-normalised",
                 "preprocessing_used": "open_clip eval transform of the checkpoint (224, bicubic, OpenAI CLIP mean/std)",
                 "layernorm_rank_deficit_expected": False,
                 "note": "linear projection after ln_post maps the 1023-dim LayerNorm hyperplane into 768 dims: no exact deficit expected; to measure"},
    "siglip2": {"repo": "google/siglip2-so400m-patch16-256", "revision": None,
                "files": ["config.json", "preprocessor_config.json"], "library": "transformers SiglipModel",
                "feature": "get_image_features = attention-pooled (MAP head) vision output, not L2-normalised",
                "preprocessing_used": "official processor: resize to 256 x 256 bilinear, rescale, mean/std 0.5, no crop",
                "layernorm_rank_deficit_expected": False,
                "note": "pooled output leaves an MLP residual after the head LayerNorm: no exact deficit expected; to measure. "
                        "Checkpoint choice: official 256 variant preferred over 384 by the task's checkpoint policy; "
                        "alternative google/siglip2-so400m-patch16-384 (dd658faa) recorded in docs/encoder_feasibility_audit.md"},
}


def config_files(repo: str, revision: str, files: list[str]) -> dict:
    out = {}
    for f in files:
        try:
            path = hf_hub_download(repo, f, revision=revision, local_files_only=True)
        except Exception:
            try:
                path = hf_hub_download(repo, f, revision=revision)
            except Exception as e:  # gated or absent
                out[f] = {"error": type(e).__name__}
                continue
        out[f] = json.loads(Path(path).read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    rows = {}
    for name, e in ENCODERS.items():
        info = api.model_info(e["repo"], revision=e["revision"], files_metadata=True)
        weights = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin"))}
        card = info.card_data.to_dict() if info.card_data else {}
        rec = {"encoder": name, "repo": e["repo"], "revision": info.sha, "pinned_revision_requested": e["revision"],
               "gated": info.gated, "license": card.get("license"), "license_name": card.get("license_name"),
               "params_total": info.safetensors.total if info.safetensors else None, "weight_files_bytes": weights,
               "library": e["library"], "feature": e["feature"], "preprocessing_used": e["preprocessing_used"],
               "feature_l2_normalised": False, "layernorm_rank_deficit_expected": e["layernorm_rank_deficit_expected"],
               "note": e["note"], "configs": config_files(e["repo"], info.sha, e["files"])}
        rows[name] = rec
        (a.out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: {f: v[f] for f in ("repo", "revision", "gated", "license", "params_total")} for k, v in rows.items()},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
