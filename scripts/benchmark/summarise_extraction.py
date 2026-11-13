"""Per-combination summary of extract_encoder.sh outputs (task.md d57a630, C1-C3), read-only: rows and chunks written,
dimension / dtype / all chunks finite, feature-norm range, encode seconds summed over chunks, measured images per
second per GCD (rows / encode seconds; startup excluded) and the implied full-node (8 GCD) rate, the cache verification
(verify.json) and the 256-row CPU-vs-GPU comparison (gpu_vs_cpu256.json), and peak GPU memory / host RSS per task.

Usage: summarise_extraction.py --run NAME=OUT_DIR ... --out FILE.csv   (OUT_DIR of one extract_encoder.sh job)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarise(name: str, out: Path) -> dict:
    caches = [d for d in (out / "features_gpu").iterdir() if d.is_dir()]
    assert len(caches) == 1, f"{out}: expected one encoder cache"
    metas = [json.loads(p.read_text()) for p in sorted(caches[0].glob("chunk_*.json"))]
    peaks = []
    for log in sorted(out.glob("extract_task*.log")):
        for line in log.read_text().splitlines():
            if line.startswith("{") and "peak_gpu_memory_bytes" in line:
                peaks.append(json.loads(line))
    verify = json.loads((out / "verify.json").read_text()) if (out / "verify.json").exists() else {}
    cpu = json.loads((out / "gpu_vs_cpu256.json").read_text()) if (out / "gpu_vs_cpu256.json").exists() else {}
    rows = sum(m["rows"] for m in metas)
    seconds = sum(m["seconds"] for m in metas)
    return {"run": name, "encoder": metas[0]["encoder"], "model_id": metas[0]["model_id"], "library": metas[0]["library"],
            "chunks": len(metas), "rows": rows, "dim": metas[0]["dim"], "dtype": metas[0]["dtype"],
            "all_finite": all(m["finite"] for m in metas), "norm_min": min(m["norm_min"] for m in metas),
            "norm_max": max(m["norm_max"] for m in metas), "encode_seconds": round(seconds, 1),
            "images_per_s_per_gcd": round(rows / seconds, 1), "images_per_s_node_8gcd": round(8 * rows / seconds, 1),
            "verify_pass": bool(verify) and not verify["problems"] and verify["rows_covered"] == verify["manifest_rows"],
            "verify_problems": len(verify.get("problems", [])), "verify": {k: v for k, v in verify.items() if k != "chunks"},
            "cpu_vs_gpu256_max_abs": cpu.get("max_abs_diff"), "cpu_vs_gpu256_max_rel_norm": cpu.get("max_rel_norm_diff"),
            "peak_gpu_memory_gb": round(max((p["peak_gpu_memory_bytes"] or 0) for p in peaks) / 1e9, 2) if peaks else None,
            "peak_host_rss_gb": round(max(p["peak_host_rss_kib"] for p in peaks) / 2 ** 20, 2) if peaks else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = [summarise(*s.split("=", 1)[:1], Path(s.split("=", 1)[1])) for s in a.run]
    t = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.drop(columns=["verify"]).to_csv(a.out, index=False)
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str))
    print(t.drop(columns=["verify", "library", "model_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
