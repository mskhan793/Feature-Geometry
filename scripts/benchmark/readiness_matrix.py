"""8 x 4 readiness matrix of the benchmark (task.md 8538df5) with extraction workload, storage and time estimates.

Inputs: the dataset summaries of scripts/benchmark/build_manifests.py (image counts), the frozen aquatic selections
(images actually encoded: up to 10 frames per camera), and the recorded facts below (image location, checkpoint
availability on LUMI, existing caches; each with its evidence in docs/evidence/benchmark_readiness/). Estimates:
- throughput: measured DINOv3-L extraction, AquaMonitor, LUMI job 23642060 (1 node, 8 GCDs): 836,159 images in
  20:25 = 682.6 images/s per node (startup included). ViT-L-sized encoders (DINOv2-L, BioCLIP 2: 256 patch tokens
  as DINOv3-L at 256 px) are assumed equal; SigLIP 2 so400m (vision tower ~1.4x a ViT-L's parameters, 256 tokens at
  256 px) is assumed 1.4x slower. Both are EXTRAPOLATIONS, to be replaced by measured values.
- GPU-hours = node-hours x 4 (one LUMI node = 4 MI250X = 8 GCDs); storage = images x dim x 4 bytes (float32).

Usage: readiness_matrix.py --summaries DIR --aquatic-selection finbenthic2=SEL aquamonitor=SEL --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

THROUGHPUT_NODE = 836159 / (20 * 60 + 25)  # images/s per node, job 23642060
ENCODERS = {"dinov3_l": {"dim": 1024, "slowdown": 1.0, "checkpoint_on_lumi": True},
            "dinov2_l": {"dim": 1024, "slowdown": 1.0, "checkpoint_on_lumi": True},
            "bioclip2": {"dim": 768, "slowdown": 1.0, "checkpoint_on_lumi": True},
            "siglip2": {"dim": 1152, "slowdown": 1.4, "checkpoint_on_lumi": False}}
IMAGES_ON_LUMI = {"finbenthic2": "/flash/.../finbenthic2/finbenthic2_images.h5 (byte-verified)",
                  "aquamonitor": "/flash/.../data_hdf_aqua/aquamonitor_complete.h5 (byte-identical to the release)",
                  "bioscan5m": "/scratch/.../aqua_v1/bioscan/zips (official cropped_256 train + eval)",
                  "df24": None,
                  "fgvc_aircraft": "/scratch/.../benchmark_sources/fgvc_aircraft (archive, sha256 e4e323d4)",
                  "food101": "/scratch/.../benchmark_sources/food101 (archive, sha256 d97d15e4)",
                  "cifar100": "/scratch/.../benchmark_sources/cifar100 (archive, publisher MD5 verified)",
                  "oxford_pets": "/scratch/.../benchmark_sources/oxford_pets (archive, sha256 67195c5e)"}
EXISTING = {("dinov3_l", "finbenthic2"): "cached (Feature-Geometry, verified)",
            ("dinov3_l", "aquamonitor"): "cached (Feature-Geometry, verified)",
            ("dinov3_l", "bioscan5m"): "cached in v1 (same checkpoint/transform; equivalence check pending)",
            ("dinov2_l", "bioscan5m"): "cached in v1 (same checkpoint/transform; equivalence check pending)",
            ("bioclip2", "bioscan5m"): "cached in v1 (same checkpoint/transform; equivalence check pending)"}
DATASETS = ["finbenthic2", "aquamonitor", "bioscan5m", "df24", "fgvc_aircraft", "food101", "cifar100", "oxford_pets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True, type=Path)
    ap.add_argument("--aquatic-selection", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    images = {d: json.loads((a.summaries / f"{d}.json").read_text())["rows"] for d in DATASETS}
    for s in a.aquatic_selection:
        name, path = s.split("=", 1)
        images[name] = len(pd.read_parquet(path, columns=["img"]))
    rows = []
    for e, spec in ENCODERS.items():
        for d in DATASETS:
            existing = EXISTING.get((e, d))
            missing = [m for m, ok in (("images", IMAGES_ON_LUMI[d] is not None), ("checkpoint", spec["checkpoint_on_lumi"])) if not ok]
            status = existing or ("ready" if not missing else "missing " + " + ".join(missing))
            todo = 0 if existing and existing.startswith("cached (Feature") else images[d]
            node_h = todo * spec["slowdown"] / THROUGHPUT_NODE / 3600
            rows.append({"encoder": e, "dataset": d, "status": status, "images": images[d], "images_to_encode": todo,
                         "est_node_hours": round(node_h, 3), "est_gpu_hours": round(4 * node_h, 3),
                         "storage_gb_float32": round(images[d] * spec["dim"] * 4 / 1e9, 3)})
    m = pd.DataFrame(rows)
    m.to_csv(a.out / "readiness_matrix.csv", index=False)
    pivot = m.pivot(index="dataset", columns="encoder", values="status").loc[DATASETS, list(ENCODERS)]
    totals = m.groupby("encoder")[["images_to_encode", "est_node_hours", "est_gpu_hours", "storage_gb_float32"]].sum().round(3)
    summary = {"throughput_images_per_s_per_node": round(THROUGHPUT_NODE, 1), "images_per_dataset": images,
               "totals_per_encoder": totals.to_dict("index"), "total_gpu_hours": round(float(m.est_gpu_hours.sum()), 3),
               "total_storage_gb": round(float(m.storage_gb_float32.sum()), 3),
               "status_counts": m.status.value_counts().to_dict()}
    (a.out / "readiness_summary.json").write_text(json.dumps(summary, indent=1))
    (a.out / "readiness_matrix.md").write_text("Status (dataset x encoder):\n\n" + pivot.to_string() + "\n\nTotals per encoder:\n\n"
                                               + totals.to_string() + "\n")
    print(pivot.to_string()); print(totals.to_string()); print(json.dumps({k: v for k, v in summary.items() if k != "totals_per_encoder"}, indent=1))


if __name__ == "__main__":
    main()
