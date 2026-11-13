"""Does a frozen feature cache lie in an affine hyperplane fixed by the encoder's final LayerNorm? (read-only audit,
2026-11-12; motivated by the native-space coupled RSCM run, whose pooled covariance had rank p - 1).

LayerNorm output x = w * (h - mean(h)) / std(h) + b satisfies sum_j (x_j - b_j) / w_j = 0 for every vector, so every
covariance of such features is singular along v = (1/w) / ||1/w||. The script reads the cached features (float32 as
stored, computed in float64), the final-norm weight and bias from the model's local safetensors, and reports:
the constraint value per row relative to the row norm, the eigenvalues of the feature covariance (smallest and
largest), and |cos| between the smallest-eigenvalue eigenvector and v.

Without ``--weights`` (encoders whose output is not a final LayerNorm, e.g. BioCLIP 2, SigLIP 2) only the eigenvalues
and the numerical rank are reported.

Usage: audit_feature_null_space.py --cache DIR (chunk_*.npy) [--weights model.safetensors --key-prefix norm] --out FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from src.features.null_space import layernorm_constraint, spectrum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--key-prefix", default="norm")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    x = np.concatenate([np.load(f) for f in sorted(a.cache.glob("chunk_*.npy"))])
    out, cov, vec = spectrum(x)
    if a.weights:
        tensors = load_file(a.weights)
        out |= {"weights": str(a.weights), "key": f"{a.key_prefix}.weight/bias"}
        out |= layernorm_constraint(x, tensors[f"{a.key_prefix}.weight"], tensors[f"{a.key_prefix}.bias"], cov, vec)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
