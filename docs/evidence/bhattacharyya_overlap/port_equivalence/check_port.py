"""Equivalence of the Feature-Geometry numpy port (src/methods/bhattacharyya.py, stable_transfer_*) with the authors'
original TensorFlow gbc.py (google-research stable_transfer @ 5b09c22, copied unchanged; sha256 checked), on
inputs.npz from prepare_inputs.py. Run with the TensorFlow test environment: tf_env/bin/python check_port.py REPO.

Compared for both gaussian types (diagonal, spherical) and both datasets: per-class mean and floored variance, every
ordered-pair Bhattacharyya distance, and the GBC score. The original's import of transfer_experiment (only used by a
decorator for experiment caching) is replaced by a stub that returns the function unchanged. Writes result.json.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ORIGINAL = HERE.parent / "code_candidate" / "gbc.py"
SHA = "4f2dd55ca2068b75dd986b51115db0eccac0d511edff5fb336176c7df1ea1ead"
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == SHA, "original gbc.py changed"

stub = Path(tempfile.mkdtemp())
pkg = stub / "stable_transfer" / "transferability"
pkg.mkdir(parents=True)
(stub / "stable_transfer" / "__init__.py").write_text("")
(pkg / "__init__.py").write_text("")
(pkg / "transfer_experiment.py").write_text("def load_or_compute(f):\n    return f\n")
shutil.copy(ORIGINAL, pkg / "gbc.py")
sys.path[:0] = [str(stub), sys.argv[1]]

import tensorflow as tf  # noqa: E402
from stable_transfer.transferability import gbc as original  # noqa: E402

from src.methods import bhattacharyya as port  # noqa: E402

data = np.load(HERE / "inputs.npz")
result = {"original_sha256": SHA, "tensorflow": tf.__version__, "numpy": np.__version__, "checks": {}}
ok = True
for name in ("dummy", "real"):
    x, y = data[f"{name}_x"], data[f"{name}_y"]
    labels = list(dict.fromkeys(y.tolist()))
    stats_tf = original.compute_per_class_mean_and_variance(tf.constant(x), tf.constant(y), labels)
    for gtype in ("diagonal", "spherical"):
        mean_diff = var_diff = dist_diff = 0.0
        dist_max = 0.0
        for c in labels:
            mu, var = port.stable_transfer_class_stats(x[y == c])
            mean_diff = max(mean_diff, float(np.abs(mu - stats_tf[c]["mean"].numpy()).max()))
            var_diff = max(var_diff, float(np.abs(var - stats_tf[c]["variance"].numpy()).max()))
        for c1 in labels:
            for c2 in labels:
                if c1 == c2:
                    continue
                d_tf = float(original.get_bhattacharyya_distance(stats_tf, c1, c2, gtype))
                m1, v1 = port.stable_transfer_class_stats(x[y == c1])
                m2, v2 = port.stable_transfer_class_stats(x[y == c2])
                d_np = float(port.stable_transfer_distance(m1, m2, v1, v2, gtype))
                dist_diff = max(dist_diff, abs(d_tf - d_np) / max(1.0, abs(d_tf)))
                dist_max = max(dist_max, abs(d_tf))
        g_tf = float(original.get_gbc_score(tf.constant(x), tf.constant(y), gtype))
        g_np = port.stable_transfer_gbc(x, y, gtype)
        check = {"classes": len(labels), "rows": int(len(y)), "max_abs_diff_class_mean": mean_diff,
                 "max_abs_diff_class_variance": var_diff, "max_rel_diff_pair_distance": dist_diff,
                 "max_abs_pair_distance": dist_max, "gbc_original": g_tf, "gbc_port": g_np,
                 "gbc_rel_diff": abs(g_tf - g_np) / abs(g_tf)}
        check["pass"] = bool(dist_diff < 1e-5 and check["gbc_rel_diff"] < 1e-5 and mean_diff < 1e-5 and var_diff < 1e-4)
        ok &= check["pass"]
        result["checks"][f"{name}_{gtype}"] = check
result["pass"] = bool(ok)
(HERE / "result.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
sys.exit(0 if ok else 1)
