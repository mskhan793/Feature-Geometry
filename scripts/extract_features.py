"""Extract frozen-encoder features for manifest rows into a chunked cache (one process per GPU).

This process encodes chunks ``task, task + n_tasks, ...`` of the manifest (rows in manifest order) and skips chunks
that are already complete. ``--limit-rows`` restricts the run to the first N manifest rows (smoke tests only; use a
separate cache directory).

Usage: extract_features.py --manifest M.parquet --hdf5 H5 --encoder bioclip2|dinov2_l|dinov3_l --cache-dir DIR
       [--chunk-size 65536] [--task K --n-tasks N] [--batch-size 128] [--num-workers 6] [--limit-rows N]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path

import pandas as pd
import torch

from src.encoders.frozen_encoders import load_encoder
from src.features import chunk_store
from src.features.extract import extract_chunks
from src.lightning_utils.trainer import build_trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=65536)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    a = ap.parse_args()

    manifest = pd.read_parquet(a.manifest, columns=["img", "hdf5_row"])
    if a.limit_rows:
        manifest = manifest.iloc[:a.limit_rows]
    chunks = range(a.task, chunk_store.n_chunks(len(manifest), a.chunk_size), a.n_tasks)
    encoder, transform, info = load_encoder(a.encoder)
    info["manifest_sha256"] = chunk_store.file_sha256(a.manifest)
    trainer = build_trainer([], max_epochs=1, accelerator=a.accelerator, devices=1, deterministic="warn",
                            enable_progress_bar=False, enable_model_summary=False)
    print(json.dumps({"task": a.task, "n_tasks": a.n_tasks, "rows": len(manifest), "chunks": list(chunks), **info}), flush=True)
    extract_chunks(trainer, encoder, transform, info, manifest, a.hdf5, a.cache_dir / a.encoder, a.chunk_size, chunks,
                   a.batch_size, a.num_workers, log=lambda s: print(f"[task {a.task}] {s}", flush=True))
    print(json.dumps({"task": a.task, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                      "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)


if __name__ == "__main__":
    main()
