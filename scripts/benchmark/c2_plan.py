"""C2 extraction plan (task.md d57a630): which of the 32 encoder x dataset train+val combinations are reused from
verified caches, done in C1, or newly extracted; image counts from the train+val extraction manifests; measured C1
throughput (FGVC-Aircraft, 8 GCDs, startup excluded) -> estimated encode time, a Slurm time limit, GPU-hours and
float32 storage. Read-only; writes plan.tsv (consumed by scripts/lumi/submit_c2.sh) and plan.json.

GPU-hours: one standard-g node = 8 GCDs = 4 MI250X GPUs, billed 4 GPU-h per node-hour. The estimate charges the
GPU job's whole run (encode / 0.9 + OVERHEAD_MIN of startup); verification and the CPU 256-row check run in a separate
CPU job (check_extraction.sh). Time limit: 1.5 x the estimated encode / 0.9 + 20 min, rounded up to 15 min.

Usage: c2_plan.py --rates extraction_summary.csv --local-root FEATURE_GEOMETRY_DATA --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ENCODERS = ("dinov3_l", "dinov2_l", "bioclip2", "siglip2")
DIMS = {"dinov3_l": 1024, "dinov2_l": 1024, "bioclip2": 768, "siglip2": 1152}
F = "/flash/project_462001534/feature_geometry"
# dataset: (local extraction manifest relative to --local-root, LUMI manifest, source flag, LUMI source)
DATASETS = {
    "finbenthic2": ("full_specimen/finbenthic2/selection.parquet", f"{F}/finbenthic2/full_selection.parquet",
                    "--hdf5", f"{F}/finbenthic2/finbenthic2_images.h5"),
    "aquamonitor": ("full_specimen/aquamonitor/selection.parquet", f"{F}/aquamonitor/full_selection.parquet",
                    "--hdf5", "/flash/project_462001534/data_hdf_aqua/aquamonitor_complete.h5"),
    "bioscan5m": ("benchmark/extract/bioscan5m_trainval.parquet", f"{F}/benchmark/extract/bioscan5m_trainval.parquet",
                  "--hdf5", f"{F}/benchmark/bioscan5m/bioscan5m_trainval.h5"),
    "df24": ("benchmark/extract/df24_trainval.parquet", f"{F}/benchmark/extract/df24_trainval.parquet",
             "--hdf5", f"{F}/benchmark/df24/df24_trainval.h5"),
    "food101": ("benchmark/extract/food101_trainval.parquet", f"{F}/benchmark/extract/food101_trainval.parquet",
                "--hdf5", f"{F}/benchmark/food101/food101_trainval.h5"),
    "oxford_pets": ("benchmark/extract/oxford_pets_trainval.parquet", f"{F}/benchmark/extract/oxford_pets_trainval.parquet",
                    "--hdf5", f"{F}/benchmark/oxford_pets/oxford_pets_trainval.h5"),
    "cifar100": ("benchmark/extract/cifar100_trainval.parquet", f"{F}/benchmark/extract/cifar100_trainval.parquet",
                 "--cifar-dir", f"{F}/benchmark/cifar100/cifar-100-python"),
    "fgvc_aircraft": ("benchmark/extract/fgvc_aircraft_trainval.parquet",
                      f"{F}/benchmark/extract/fgvc_aircraft_trainval.parquet",
                      "--hdf5", f"{F}/benchmark/fgvc_aircraft/fgvc_aircraft_trainval.h5"),
}
S = "/scratch/project_462001534"
# Combinations not newly extracted in C2, with the evidence that allows it.
NOT_NEW = {
    ("dinov3_l", "finbenthic2"): ("reuse", "full-specimen cache (jobs 23623689 + 23642052; two caches covering the "
                                  "selection), verified and used in tasks 92a605a onward",
                                  f"{S}/feature_geometry/finbenthic2_fold0/23623689/features_gpu/dinov3_l + "
                                  f"{S}/feature_geometry/full_specimen/finbenthic2/extract/features_gpu/dinov3_l"),
    ("dinov3_l", "aquamonitor"): ("reuse", "full-specimen cache (job 23642060), verified and used in tasks 92a605a onward",
                                  f"{S}/feature_geometry/full_specimen/aquamonitor/extract/features_gpu/dinov3_l"),
    ("dinov3_l", "bioscan5m"): ("reuse", "v1 cache equivalent to fresh extraction (C1, max rel 1.5e-5 <= 1e-4)",
                                f"{S}/aqua_v1/bioscan/cache_dinov3"),
    ("bioclip2", "bioscan5m"): ("reuse", "v1 cache equivalent to fresh extraction (C1, max rel 3.3e-5 <= 1e-4)",
                                f"{S}/aqua_v1/bioscan/cache_bioclip2"),
}
for e in ENCODERS:
    NOT_NEW[(e, "fgvc_aircraft")] = ("c1", "C1 throughput pilot = full train+val (6,667 images), verified",
                                     f"{S}/feature_geometry/benchmark_c1/aircraft/{e}")
OVERHEAD_MIN = 2  # GPU-job startup (C1 Aircraft jobs: 1.3-1.8 min from start to end of encoding of 6,667 images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", required=True, type=Path)
    ap.add_argument("--local-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    r = pd.read_csv(a.rates)
    rate = r[r.run.str.startswith("aircraft_")].set_index("encoder").images_per_s_node_8gcd
    rows = []
    for d, (local, manifest, flag, source) in DATASETS.items():
        m = pd.read_parquet(a.local_root / local, columns=["img"])
        assert m.img.is_unique, d
        for e in ENCODERS:
            status, why, path = NOT_NEW.get((e, d), ("new", "", f"{S}/feature_geometry/benchmark_c2/{d}/{e}"))
            encode_s = len(m) / rate[e]
            limit_min = 15 * math.ceil((1.5 * encode_s / 0.9 / 60 + 20) / 15)
            rows.append({"encoder": e, "dataset": d, "status": status, "images": len(m), "dim": DIMS[e],
                         "float32_gb": round(len(m) * DIMS[e] * 4 / 1e9, 3), "node_rate": rate[e],
                         "est_encode_min": round(encode_s / 60, 1),
                         "est_gpu_h": round(4 * (encode_s / 0.9 / 3600 + OVERHEAD_MIN / 60), 2) if status == "new" else 0,
                         "time_limit": f"{limit_min // 60:02d}:{limit_min % 60:02d}:00", "manifest": manifest,
                         "source_flag": flag, "source": source, "out": path, "evidence": why})
    t = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    new = t[t.status == "new"]
    new[["encoder", "dataset", "time_limit", "manifest", "source_flag", "source", "out"]].to_csv(
        a.out_dir / "plan.tsv", sep="\t", index=False, header=False)
    t.to_csv(a.out_dir / "plan_all.csv", index=False)
    summary = {"combinations": len(t), "by_status": t.status.value_counts().to_dict(), "new_jobs": len(new),
               "new_images": int(new.images.sum()), "new_float32_gb": round(new.float32_gb.sum(), 2),
               "est_gpu_h_new": round(new.est_gpu_h.sum(), 1),
               "est_encode_node_h": round(new.est_encode_min.sum() / 60, 2)}
    (a.out_dir / "plan.json").write_text(json.dumps(summary, indent=1))
    print(t[["encoder", "dataset", "status", "images", "est_encode_min", "est_gpu_h", "time_limit", "float32_gb"]]
          .to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
