"""Exploratory, in-sample geometry of one pooled cohort (task.md 92a605a, output A): every train and val specimen of
one fold (default fold0), so that fold's test specimens stay unread. NOT held-out evidence: everything is fitted and
described on the same specimens; held-out comparisons are in class_geometry_pilot.py runs per fold (output B).

Unit: one embedding per specimen (mean of its camera-level means). Classes with at least 15 pooled specimens.
Predeclared statistics, no tuning:
- within class (native 1024-d): eigenvalue summary on all pooled specimens and on the first 30 in sha1(seed:individual)
  order, camera-pair re-identification, frame spread (src.statistics.specimen_geometry);
- between classes (native, pooled): radius separation, in-sample d' along the centroid direction, centroid distance,
  top-5 subspace overlap, 2-D PCA radius separation;
- per class at each of ``--pcs`` (global PCA on the pool, whitened, reg_covar 1e-3): one full Gaussian vs a
  2-component tied mixture by in-sample BIC; component share of the smaller group; both camera views in the same
  component; association of the components with metadata: ARI with ``--source-column`` (if given), AUC of each
  ``--numeric`` field (``log_image_area`` = log(width * height)).

Writes per_class_geometry.csv, class_pairs.csv, per_class_mixture.csv, eigenspectra.png and results.json to ``--out``.

Usage: descriptive_geometry.py --cohort selection.parquet --caches MANIFEST=CACHE ... --out DIR [--fold fold0]
       [--pcs 8 16] [--source-column dataset|none] [--numeric log_image_area]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from threadpoolctl import threadpool_limits

from src.features.chunk_store import file_sha256
from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.statistics.class_geometry import principal_axes
from src.statistics.class_modality import eligible, fit_mixture
from src.statistics.specimen_geometry import between_classes, paired_views, within_class

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REG, MIN_SPECIMENS, N_FIXED, K_SUB = 1e-3, 15, 30, 5


def mixture_row(c, d, z, views, groups, numeric, seed) -> dict:
    with threadpool_limits(1):
        row = {"taxon_group": c, "pcs": d, "n": len(z)}
        if not (eligible(1, d, "full", len(z)) and eligible(2, d, "tied", len(z))):
            return {**row, "status": "ineligible"}
        g1, g2 = fit_mixture(z, 1, "full", REG, seed), fit_mixture(z, 2, "tied", REG, seed)
        lab = g2.predict(z)
        row.update({"status": "ok", "bic_gain_G2_tied": g1.bic(z) - g2.bic(z),
                    "smaller_component_share": float(np.bincount(lab, minlength=2).min() / len(lab)),
                    "both_views_same_component": float(np.mean(g2.predict(views[0]) == g2.predict(views[1])))})
        if groups is not None:
            row["ari_source"] = adjusted_rand_score(groups, lab) if len(set(groups)) > 1 else None
        for f, v in numeric.items():
            row[f"auc_{f}"] = roc_auc_score(lab == 1, v) if len(set(lab)) == 2 else None
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+", help="MANIFEST=CACHE_DIR pairs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--pcs", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--source-column", default="dataset", help="'none' if the dataset has no source field")
    ap.add_argument("--numeric", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    S = a.fold
    cohort = pd.read_parquet(a.cohort)
    rows_in = len(cohort)
    cohort = cohort[cohort[S].isin(["train", "val"])].reset_index(drop=True)  # this fold's test rows are never read
    if "log_image_area" in a.numeric:
        cohort["log_image_area"] = np.log(cohort.width * cohort.height)
    x, meta = features_by_name([tuple(Path(p) for p in s.split("=", 1)) for s in a.caches], cohort.img.tolist())
    cam_keys, cam_means, frame_rms = specimen_camera_means(x, cohort, S)
    sp_keys, sp = specimen_means(cam_keys, cam_means, S)
    cam = cam_keys.assign(row=np.arange(len(cam_keys)))
    n = sp_keys.taxon_group.value_counts()
    classes = sorted(n.index[n >= MIN_SPECIMENS])

    geo = within_class(classes, sp_keys, sp, cam, cam_means, frame_rms, a.seed, N_FIXED)
    pca2 = PCA(n_components=2, svd_solver="full").fit(sp)
    pairs = between_classes(classes, sp_keys, sp, pca2.transform(sp), k_sub=K_SUB).rename(
        columns={"dprime_train": "dprime_in_sample"})
    pca = PCA(n_components=max(a.pcs), whiten=True, svd_solver="full").fit(sp)
    z_all, zc_all = pca.transform(sp), pca.transform(cam_means)
    source = None if a.source_column == "none" else cohort.groupby("individual")[a.source_column].first()
    numeric = cohort.groupby("individual")[a.numeric].mean() if a.numeric else None
    jobs = []
    for d in a.pcs:
        for c in classes:
            m = (sp_keys.taxon_group == c).values
            ids = sp_keys.individual[m]
            r1, r2 = paired_views(cam, ids)
            groups = None if source is None else source.reindex(ids).values
            num = {} if numeric is None else {f: numeric[f].reindex(ids).values for f in a.numeric}
            jobs.append(delayed(mixture_row)(c, d, z_all[m, :d], (zc_all[r1, :d], zc_all[r2, :d]), groups, num, a.seed))
    mix = pd.DataFrame(Parallel(n_jobs=a.jobs)(jobs))

    geo.to_csv(a.out / "per_class_geometry.csv", index=False)
    pairs.sort_values("dprime_in_sample").to_csv(a.out / "class_pairs.csv", index=False)
    mix.to_csv(a.out / "per_class_mixture.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c in classes:
        lam = principal_axes(sp[(sp_keys.taxon_group == c).values])[0]
        ax.plot(np.arange(1, 16), lam[:15] / lam.sum(), c="0.6", lw=0.6)
    ax.set(yscale="log", xlabel="principal direction (within class, native)", ylabel="share of class variance",
           title=f"Within-class spectra, {S} train + val pooled (in-sample)")
    fig.tight_layout()
    fig.savefig(a.out / "eigenspectra.png", dpi=130)
    plt.close(fig)

    by_d = {}
    for d, g in mix.groupby("pcs"):
        ok = g[g.status == "ok"]
        s = {"classes_fitted": len(ok), "classes_ineligible": int((g.status != "ok").sum()),
             "bic_prefers_G2_tied": int((ok.bic_gain_G2_tied > 0).sum()),
             "both_views_same_component_median": float(ok.both_views_same_component.median())}
        if "ari_source" in ok:
            s["source_ari_at_least_0.5"] = sorted(ok.taxon_group[ok.ari_source >= 0.5])
        for f in a.numeric:
            s[f"{f}_auc_outside_0.2_0.8"] = sorted(ok.taxon_group[(ok[f"auc_{f}"] - 0.5).abs() >= 0.3])
        by_d[str(d)] = s
    result = {
        "status": "exploratory, in-sample: fitted and described on the same specimens; not held-out evidence",
        "pool": f"{S} train + val specimens; {S} test rows not read", "cohort_sha256": file_sha256(a.cohort),
        "cohort_rows": {"in_file": rows_in, "pool": len(cohort), "not_read": rows_in - len(cohort)},
        "encoder": meta["encoder"], "model_id": meta["model_id"], "transform": meta["transform"], "tf32": meta.get("tf32"),
        "seed": a.seed, "specimens": len(sp_keys), "classes": len(classes),
        "classes_below_min": {c: int(n[c]) for c in sorted(set(n.index) - set(classes))},
        "specimens_per_class_min_median_max": [int(n[classes].min()), float(n[classes].median()), int(n[classes].max())],
        "pca_explained_variance": {str(d): float(pca.explained_variance_ratio_[:d].sum()) for d in a.pcs},
        "within_class_native_min_median_max": {k: [float(geo[k].min()), float(geo[k].median()), float(geo[k].max())]
                                              for k in ("rms_radius", "top1_share", "effective_rank", "effective_rank_n30",
                                                        "camera_reid", "frame_rms_within_specimen_median")},
        "between_classes": {"pairs": len(pairs), "dprime_in_sample_below_2": int((pairs.dprime_in_sample < 2).sum()),
                            "dprime_in_sample_median": float(pairs.dprime_in_sample.median()),
                            "closest_pairs": json.loads(pairs.nsmallest(10, "dprime_in_sample")[
                                ["class_a", "class_b", "dprime_in_sample"]].to_json(orient="records"))},
        "mixture_in_sample_by_pcs": by_d, "wall_seconds": round(time.time() - t0, 1),
    }
    (a.out / "results.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
