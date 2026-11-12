"""Does a frozen feature cache lie in an affine hyperplane fixed by the encoder's final LayerNorm? (read-only audit,
2026-11-12; motivated by the native-space coupled RSCM run, whose pooled covariance had rank p - 1).

LayerNorm output x = w * (h - mean(h)) / std(h) + b satisfies sum_j (x_j - b_j) / w_j = 0 for every vector, so every
covariance of such features is singular along v = (1/w) / ||1/w||. The script reads the cached features (float32 as
stored, computed in float64), the final-norm weight and bias from the model's local safetensors, and reports:
the constraint value per row relative to the row norm, the eigenvalues of the feature covariance (smallest and
largest), and |cos| between the smallest-eigenvalue eigenvector and v.

Usage: audit_feature_null_space.py --cache DIR (chunk_*.npy) --weights model.safetensors [--key-prefix norm] --out FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--key-prefix", default="norm")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    x = np.concatenate([np.load(f) for f in sorted(a.cache.glob("chunk_*.npy"))]).astype(np.float64)
    tensors = load_file(a.weights)
    w = tensors[f"{a.key_prefix}.weight"].astype(np.float64)
    b = tensors[f"{a.key_prefix}.bias"].astype(np.float64)
    assert w.shape == (x.shape[1],), (w.shape, x.shape)
    constraint = ((x - b) / w).sum(1)
    rel = np.abs(constraint) / np.linalg.norm((x - b) / w, axis=1)
    cov = np.cov(x, rowvar=False)
    eig, vec = np.linalg.eigh(cov)
    v = (1 / w) / np.linalg.norm(1 / w)
    out = {"rows": int(len(x)), "dim": int(x.shape[1]), "weights": str(a.weights), "key": f"{a.key_prefix}.weight/bias",
           "constraint_rel_abs_max": float(rel.max()), "constraint_rel_abs_median": float(np.median(rel)),
           "eigenvalues_smallest_4": eig[:4].tolist(), "eigenvalue_largest": float(eig[-1]),
           "smallest_over_second_smallest": float(eig[0] / eig[1]),
           "abs_cos_smallest_eigvec_vs_inv_w": float(abs(vec[:, 0] @ v)),
           "variance_along_inv_w": float(v @ cov @ v), "numerical_rank_1e-12": int((eig > 1e-12 * eig[-1]).sum())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
