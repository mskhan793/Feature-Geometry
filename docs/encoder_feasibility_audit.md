# Encoder feasibility audit — nine-encoder shortlist (`task.md` 17f864f)

**Date:** 2026-11-07. **Scope:** metadata only. No candidate weights were downloaded, no model was run, no job was submitted.

**Evidence** (all in `docs/evidence/encoder_candidates/`):
- `candidates.json`: summary per repository.
- `raw/<repo>.json`: the Hugging Face API answer, config files and model card, verbatim.
- `encoder_support_dell.json`: which interfaces the local environment supports.

These were produced by `scripts/encoder_candidate_audit.py` and `scripts/check_encoder_support.py`.

**What was checked against what:**
- **Verified:** read from the immutable Hugging Face revision listed (API `sha`), or from the local cache.
- **Not verified:** left as such; nothing is inferred from a model's name.

## 1. Verification matrix

Parameter counts are the safetensors total of the whole checkpoint. For SigLIP 2 this includes the text tower, so it is not the vision encoder alone. "Feature" means the single vector per image that the official interface returns.

| Role | Repository (revision) | Licence / access | Params (checkpoint) | Input / preprocessing (from config) | Feature, dim | Interface here | Status |
|---|---|---|---|---|---|---|---|
| Biology VL | `imageomics/bioclip-2` (`2957b322`) | MIT, open | not in API (open_clip files) | 224, bicubic shortest-side resize, OpenAI CLIP mean/std (open_clip config) | `model.visual(x)`, 768 (`embed_dim`) | open_clip 3.3.0 | **Implemented**; v1 equivalence check pending |
| SSL | `facebook/dinov2-large` (`47b73eef`) | Apache-2.0, open | 304,368,640 | v1 transform: 256 bicubic → crop 224, ImageNet mean/std | `pooler_output`, 1024 (`hidden_size`) | transformers `Dinov2Model` | **Implemented**; check pending |
| SSL | `facebook/dinov3-vitl16-pretrain-lvd1689m` (`ea8dc286`; local cache same) | DINOv3 licence, gated (manual; access already granted) | 303,129,600 | v1 transform: Resize((256, 256)), ImageNet mean/std | `pooler_output`, 1024 (local `config.json`: patch 16, 4 register tokens) | transformers `DINOv3ViTModel` | **Implemented**; check pending |
| Capacity scaling | `facebook/dinov3-vith16plus-pretrain-lvd1689m` (`c807c9ee`, the revision v1 pinned and copied to LUMI) | DINOv3 licence, gated (manual; v1 has the weights on LUMI) | 840,592,640 | same family as DINOv3-L; config gated, to read from the LUMI cache | `pooler_output`; dim **not yet verified** | `DINOv3ViTModel` | **Feasible**; dim pending |
| General image–text | `google/siglip2-so400m-patch16-384` (`dd658faa`) or `…-patch14-384` (`e8e48729`) | Apache-2.0, open | 1,136,039,602 / 1,136,008,498 | 384 × 384, bilinear (`resample` 2), mean/std 0.5, no crop | `get_image_features` (attention-pooled), 1152 (`vision_config.hidden_size`) | transformers `SiglipModel` | **Feasible**; **variant decision** (patch 16 or 14) |
| Large image–text | `google/siglip2-giant-opt-patch16-384` (`a713301b`) or `…-256` (`46d7129e`) | Apache-2.0, open | 1,871,885,426 / 1,871,393,906 | 384 or 256 square, bilinear, mean/std 0.5 | `get_image_features`, 1536 | `SiglipModel` | **Feasible**; **resolution decision** |
| Perception encoder | `facebook/PE-Core-G14-448` (`a6046680`, `.pt` + empty `config.yaml`) or timm port `facebook/vit_pe_core_gigantic_patch14_448_timm` (`f537e434`) | Apache-2.0, open | not in API / 1,882,032,640 | 448, patch 14; official transform from `core.vision_encoder.transforms` | CLIP image embedding, 1280 (timm config `output_dim`; attention pooling, no CLS token) | official: needs Meta `perception_models` (not installed); timm port: `vit_pe_core_gigantic_patch14_448` is in timm 1.0.31 (Dell); LUMI's timm 1.0.28 not yet checked | **Feasible with an interface decision** |
| Multi-teacher | `nvidia/C-RADIOv4-H` (`0057b339`) | NVIDIA Open Model Licence, open | 651,645,448 | preferred 512 (tested 256–2048 in steps of 16); processor resizes the shortest side to 512, no crop, **no normalisation** (done inside the model) | `summary` (CLS-like); dim "some number of channels", **not stated, to measure** | `AutoModel(trust_remote_code=True)`: runs the repository's own Python code (26 files) | **Feasible with conditions** (code review, fixed input size, dim) |
| Autoregressive multimodal | `apple/aimv2-*`: 4 sizes (large 309.2 M … 3B 2,720.7 M) × 224 / 336 / 448, plus native / distilled / LiT variants (audited: the 224 and 448 sizes and large-native; the 336 variants are listed on Hugging Face but not audited) | Apple AMLR licence, open | e.g. huge-448 682,031,616 | e.g. huge-448: shortest side 448 bicubic, crop 448, OpenAI CLIP mean/std | **No CLS / pooled output in the vision model** (`Aimv2VisionModel` returns patch tokens); hidden 1024 / 1536 / 2048 / 3072 | transformers native `Aimv2VisionModel` (no remote code needed) | **Variant and pooling decision needed** |

## 2. Known overlaps and dependence between models

These are recorded as stated by the model cards, not as assumptions.
- **C-RADIOv4-H** lists SigLIP2-g (`google/siglip2-giant-opt-patch16-384`), DINOv3-7B and SAM3 among its sources. So it is **not independent** of the SigLIP 2 Giant and DINOv3 candidates.
- **DINOv3-L and DINOv3-H+** are the same family and corpus (LVD-1689M).
- **BioCLIP-2** is CLIP ViT-L/14 (LAION-2B) fine-tuned on TreeOfLife-200M. Whether TreeOfLife-200M contains AquaMonitor images is **not verified**.
- **SigLIP 2** is pre-trained on WebLI. C-RADIOv4 is trained on "NV-CC-Img-Text-Dataset". Overlap with AquaMonitor is unknown for both.

## 3. Points needing the author's decision before adapters are written

1. **SigLIP 2 So400M at 384:** patch 16 (`dd658faa`) or patch 14 (`e8e48729`). Both have a 1152-d feature and the same preprocessing.
2. **SigLIP 2 Giant:** 384 (`a713301b`, also C-RADIOv4's teacher) or 256 (`46d7129e`, cheaper).
3. **PE-Core-G:**
   - the official checkpoint through Meta's `perception_models` code, which would need installing and reviewing;
   - or Meta's own timm port (safetensors, timm). This would need one numerical check against the official code on a few images before use.
4. **C-RADIOv4-H:**
   - accept running its repository code (pinned to `0057b339`, reviewed before use);
   - fix one input size, because batching needs equal sizes and its processor keeps the aspect ratio. Candidates: 512 × 512 square resize, or shortest side 512 with a centre crop;
   - the `summary` dimension is measured at the smoke test.
5. **AIMv2:**
   - size and resolution, e.g. huge-448 (682 M, similar capacity to DINOv3-H+ and C-RADIOv4-H) against large-224 (cheapest) or 1B/3B;
   - a feature: the vision model has no pooled output. The candidates are the mean of the final patch tokens, or another documented pooling. This choice changes the feature and must be stated.

**Not checked yet** (LUMI refused SSH connections on 2026-11-07):
- the `timm` and transformers class support in the LUMI container;
- the DINOv3-H+ config from the LUMI cache;
- which candidate weights are already cached on LUMI.

## 4. Compute (qualitative only, nothing measured)

The baselines run at 224–256 px with about 300 M parameters. Most candidates are larger (0.65–1.9 B checkpoint parameters) and run at 384–512 px. The number of tokens grows with the square of the resolution, so their extraction time per image will be several times that of the baselines. Exact images per second must be measured in the smoke test before any estimate is given.
