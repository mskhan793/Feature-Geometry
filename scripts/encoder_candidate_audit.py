"""Metadata audit of candidate encoder checkpoints on Hugging Face (no weights downloaded).

For each repository: the immutable revision (commit sha), last modification, gating, licence, library, parameter
count from the safetensors index, file list, and the fields of ``config.json`` / preprocessing configs that fix the
feature interface (hidden size, projection dim, image size, patch size, resize/crop/normalisation). Raw API answers are
kept next to the summary so every field can be traced. Fields that are absent are recorded as absent, not guessed.

Usage: encoder_candidate_audit.py --out DIR REPO [REPO ...]
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models/{repo}"
RAW = "https://huggingface.co/{repo}/resolve/{rev}/{name}"
CONFIGS = ("config.json", "preprocessor_config.json", "open_clip_config.json", "config.yaml")
CARD = "README.md"  # model card, saved verbatim as evidence
INTERFACE_KEYS = ("architectures", "model_type", "hidden_size", "projection_dim", "embed_dim", "image_size", "patch_size",
                  "num_hidden_layers", "num_attention_heads", "intermediate_size", "num_register_tokens", "vision_config",
                  "size", "crop_size", "do_center_crop", "resample", "image_mean", "image_std", "do_normalize",
                  "rescale_factor", "auto_map", "preprocess_cfg", "model_cfg", "max_resolution", "preferred_resolution")


def get_text(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode()
    except urllib.error.HTTPError:
        return None


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code}
    except json.JSONDecodeError:
        return {"_not_json": True}


def pick(d: dict, keys=INTERFACE_KEYS) -> dict:
    out = {k: d[k] for k in keys if k in d}
    if isinstance(out.get("vision_config"), dict):
        out["vision_config"] = pick(out["vision_config"])
    return out


def audit(repo: str) -> tuple[dict, dict]:
    info = get_json(API.format(repo=repo))
    if "_http_error" in info:
        return {"repo": repo, "error": info}, {"api": info}
    rev = info.get("sha")
    files = sorted(s["rfilename"] for s in info.get("siblings", []))
    configs = {name: (get_text if name.endswith(".yaml") else get_json)(RAW.format(repo=repo, rev=rev, name=name))
               for name in CONFIGS if name in files}
    card_text = get_text(RAW.format(repo=repo, rev=rev, name=CARD)) if CARD in files else None
    card = info.get("cardData") or {}
    safetensors = info.get("safetensors") or {}
    summary = {
        "repo": repo, "revision": rev, "last_modified": info.get("lastModified"), "gated": info.get("gated"),
        "private": info.get("private"), "license": card.get("license"), "license_name": card.get("license_name"),
        "license_link": card.get("license_link"), "library": info.get("library_name"), "pipeline": info.get("pipeline_tag"),
        "parameters_safetensors": safetensors.get("total"), "parameters_by_dtype": safetensors.get("parameters"),
        "weight_files": [f for f in files if f.endswith((".safetensors", ".bin", ".pt", ".pth", ".npz"))],
        "code_files": [f for f in files if f.endswith(".py")],
        "config_fields": {name: (pick(c) if isinstance(c, dict) else c) for name, c in configs.items()},
    }
    summary["model_card_saved"] = card_text is not None
    return summary, {"api": info, "configs": configs, "model_card": card_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("repos", nargs="+")
    a = ap.parse_args()
    (a.out / "raw").mkdir(parents=True, exist_ok=True)
    summaries = []
    for repo in a.repos:
        summary, raw = audit(repo)
        (a.out / "raw" / (repo.replace("/", "__") + ".json")).write_text(json.dumps(raw, indent=1))
        summaries.append(summary)
        print(json.dumps({k: summary.get(k) for k in ("repo", "revision", "gated", "license", "library", "parameters_safetensors")}), flush=True)
    (a.out / "candidates.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
