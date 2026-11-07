"""Camera-conditioned Gaussian pilot (task.md 2026-11-07): do per-class frozen DINOv3-L embeddings look like one
Gaussian, one Gaussian per camera, or a 2-3 component mixture, judged on held-out official fold0 val specimens?

Unit: one mean embedding per specimen and camera (native 1024-d, no normalisation). Every cohort specimen is imaged by
both cameras, so each contributes one observation per camera. Fitting (PCA and all densities) uses fold0 train only;
fold0 val is only scored; the test split is not in the cohort.

Density comparison, per projection setting (covariance type, number of PCs):
- PCA fitted on all train observations (all classes), whitened; densities live in this space, not native space.
- Per class: models of src.statistics.class_density_models; K=2 only with >= 20 train specimens, K=3 with >= 30.
- Score = mean log density of a val specimen's two camera observations; class score = mean over its val specimens;
  overall = mean over classes. Differences to A_pooled with 95 % bootstrap intervals over val specimens (within class).
Supporting evidence (train unless stated): camera mean distance / within-camera spread and a paired sign-flip test in
native space, replication of the camera-difference direction on val, within-specimen frame spread, BIC (train only),
2-component / camera agreement and specimen coherence on val, PCA plots.

Usage: camera_gaussian_pilot.py --cohort cohort.parquet --cache FEATURE_DIR --out DIR [--seed 20261107]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.chunk_store import file_sha256, load_cache
from src.features.specimen_means import specimen_camera_means
from src.statistics.class_density_models import UNCONDITIONAL, component_camera_purity, fit_class_models, log_densities

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETTINGS = [("diag", 8), ("diag", 16), ("full", 8)]
REG = 1e-3  # reg_covar added to every variance, in whitened PCA units (unit total-train variance per PC)
MIN_TRAIN, MIN_VAL = 15, 5
K_MIN_SPECIMENS = {2: 20, 3: 30}
N_BOOT, N_PERM = 2000, 999
MODELS = (*UNCONDITIONAL, "B_camera_conditional")


def paired_difference(keys: pd.DataFrame, x: np.ndarray, cameras) -> np.ndarray:
    """Per specimen: camera-1 minus camera-2 embedding (rows in specimen order)."""
    a = keys[keys.camera == cameras[0]].sort_values("individual")
    b = keys[keys.camera == cameras[1]].sort_values("individual")
    assert (a.individual.values == b.individual.values).all()
    return x[a.index] - x[b.index]


def camera_separation(keys, x, frame_spread, cameras, rng) -> pd.DataFrame:
    rows = []
    for (cls, split), k in keys.groupby(["taxon_group", "fold0"]):
        d = paired_difference(k, x, cameras)
        stat = np.linalg.norm(d.mean(0))
        signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
        null = np.linalg.norm(signs @ d / len(d), axis=1)
        within = np.sqrt(np.mean([((x[k.index[k.camera == c]] - x[k.index[k.camera == c]].mean(0)) ** 2).sum(1).mean()
                                  for c in cameras]))
        rows.append({"taxon_group": cls, "split": split, "specimens": len(d), "camera_mean_distance": stat,
                     "within_camera_rms": within, "ratio": stat / within,
                     "paired_signflip_p": (1 + (null >= stat).sum()) / (1 + N_PERM),
                     "mean_frame_rms_within_specimen": float(frame_spread[k.index].mean()),
                     "direction": d.mean(0)})
    return pd.DataFrame(rows)


def bootstrap(scores: dict, classes: list, rng) -> tuple[dict, dict]:
    """Macro-mean differences to A_pooled: point, 95 % interval (resampling val specimens within class), and per class."""
    idx = {c: rng.integers(0, len(scores[c]["A_pooled"]), size=(N_BOOT, len(scores[c]["A_pooled"]))) for c in classes}
    overall, per_class = {}, {}
    for m in MODELS[1:]:
        have = [c for c in classes if m in scores[c]]
        if not have:
            continue
        point = {c: float((scores[c][m] - scores[c]["A_pooled"]).mean()) for c in have}
        boots = {c: (scores[c][m] - scores[c]["A_pooled"])[idx[c]].mean(1) for c in have}
        macro = np.mean([boots[c] for c in have], axis=0)
        overall[f"{m}-A_pooled"] = {"classes": len(have), "mean": float(np.mean(list(point.values()))),
                                    "ci95": [float(np.quantile(macro, 0.025)), float(np.quantile(macro, 0.975))],
                                    "classes_better": int(sum(v > 0 for v in point.values())),
                                    "classes_ci_above_0": int(sum(np.quantile(boots[c], 0.025) > 0 for c in have)),
                                    "classes_ci_below_0": int(sum(np.quantile(boots[c], 0.975) < 0 for c in have))}
        for c in have:
            per_class.setdefault(c, {})[f"{m}-A_pooled"] = point[c]
            per_class[c][f"{m}-A_pooled_ci95"] = [float(np.quantile(boots[c], 0.025)), float(np.quantile(boots[c], 0.975))]
    return overall, per_class


def run_setting(cov, dims, z_tr, z_va, k_tr, k_va, classes, n_train, seed, rng):
    scores, rows, failures = {}, [], {}
    for c in classes:
        tr, va = (k_tr.taxon_group == c).values, (k_va.taxon_group == c).values
        ks = [k for k, need in K_MIN_SPECIMENS.items() if n_train[c] >= need]
        models, fail = fit_class_models(z_tr[tr, :dims], k_tr.camera.values[tr], ks, cov, REG, seed)
        if fail:
            failures[c] = fail
        ld = log_densities(models, z_va[va, :dims], k_va.camera.values[va])
        spec = k_va[va].individual.values
        scores[c] = {m: pd.Series(v).groupby(spec).mean().sort_index().values for m, v in ld.items()}
        row = {"taxon_group": c, **{m: float(s.mean()) for m, s in scores[c].items()}}
        bic = {name: models[name].bic(z_tr[tr, :dims]) for name in ("A_pooled", "C_gmm2", "C_gmm3") if name in models}
        row["bic_best_train"] = min(bic, key=bic.get) if bic else None
        if "C_gmm2" in models:
            comp = models["C_gmm2"].predict(z_va[va, :dims])
            row["gmm2_camera_purity_val"] = component_camera_purity(comp, k_va.camera.values[va])
            row["gmm2_same_component_both_cameras_val"] = float(pd.Series(comp).groupby(spec).nunique().eq(1).mean())
            comp_tr = models["C_gmm2"].predict(z_tr[tr, :dims])
            row["gmm2_camera_purity_train"] = component_camera_purity(comp_tr, k_tr.camera.values[tr])
        rows.append(row)
    overall, per_class = bootstrap(scores, classes, rng)
    table = pd.DataFrame(rows)
    for c, d in per_class.items():
        for key, v in d.items():
            table.loc[table.taxon_group == c, key] = str(v) if isinstance(v, list) else v
    summary = {"covariance": cov, "pcs": dims, "reg_covar": REG, "differences_vs_A_pooled": overall,
               "mean_log_density": {m: float(table[m].mean()) for m in MODELS if m in table},
               "bic_best_train_counts": table.bic_best_train.value_counts().to_dict(),
               "gmm2_camera_purity_val_mean": float(table.gmm2_camera_purity_val.mean()),
               "gmm2_same_component_both_cameras_val_mean": float(table.gmm2_same_component_both_cameras_val.mean()),
               "fit_failures": failures}
    return summary, table.assign(covariance=cov, pcs=dims)


def plots(out, z_tr, k_tr, classes, table, cameras):
    colors = {cameras[0]: "#1f77b4", cameras[1]: "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for cam in cameras:
        m = (k_tr.camera == cam).values
        ax.scatter(z_tr[m, 0], z_tr[m, 1], s=3, alpha=0.4, c=colors[cam], label=f"camera {cam}")
    ax.set(xlabel="PC1 (whitened, train)", ylabel="PC2", title="Train specimen-camera means, all 41 classes")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(out / "pca_all_classes.png", dpi=130)
    plt.close(fig)

    n = len(classes)
    cols = 7
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(cols * 2.3, -(-n // cols) * 2.3))
    for ax, c in zip(axes.ravel(), classes):
        k = k_tr[k_tr.taxon_group == c]
        a = k[k.camera == cameras[0]].sort_values("individual").index
        b = k[k.camera == cameras[1]].sort_values("individual").index
        for i, j in zip(a, b):
            ax.plot(z_tr[[i, j], 0], z_tr[[i, j], 1], c="0.8", lw=0.5, zorder=0)
        for cam, ids in ((cameras[0], a), (cameras[1], b)):
            ax.scatter(z_tr[ids, 0], z_tr[ids, 1], s=6, c=colors[cam])
        ax.set_title(c, fontsize=7)
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per class, train: PC1 vs PC2 of global PCA; blue camera 1, orange camera 2, grey = same specimen")
    fig.tight_layout()
    fig.savefig(out / "pca_by_class.png", dpi=130)
    plt.close(fig)

    t = table[(table.covariance == "diag") & (table.pcs == 16)].sort_values("B_camera_marginal-A_pooled")
    fig, ax = plt.subplots(figsize=(7, 9))
    y = np.arange(len(t))
    for off, m, col in ((-0.2, "B_camera_marginal", "#2ca02c"), (0.2, "C_gmm2", "#9467bd")):
        ci = np.array([json.loads(v) for v in t[f"{m}-A_pooled_ci95"]])
        ax.errorbar(t[f"{m}-A_pooled"], y + off, xerr=[t[f"{m}-A_pooled"] - ci[:, 0], ci[:, 1] - t[f"{m}-A_pooled"]],
                    fmt="o", ms=3, c=col, label=f"{m} - A_pooled")
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y, t.taxon_group, fontsize=6)
    ax.set(xlabel="held-out mean log density difference per val specimen (diag, 16 PCs); 95 % bootstrap")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "density_differences_diag16.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    t0 = time.process_time()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cohort = pd.read_parquet(a.cohort)
    if not set(cohort.fold0) <= {"train", "val"}:
        raise RuntimeError("cohort holds rows outside fold0 train/val")
    x, meta = load_cache(a.cache, cohort.img.tolist())
    cameras = sorted(cohort.camera.unique())
    keys, means, frame_spread = specimen_camera_means(x, cohort)
    if set(keys[keys.fold0 == "train"].individual) & set(keys[keys.fold0 == "val"].individual):
        raise RuntimeError("specimen in both train and val")
    if not (keys.groupby("individual").camera.nunique() == len(cameras)).all():
        raise RuntimeError("a specimen lacks a camera")
    runs = cohort.groupby(["individual", "camera"]).imaging_run.agg(frozenset).unstack()
    meta_cols = cohort.groupby("individual")[["site", "year", "lake", "dataset"]].nunique()

    support = keys[keys.camera == cameras[0]].groupby(["taxon_group", "fold0"]).size().unstack(fill_value=0)
    classes = sorted(support.index[(support.train >= MIN_TRAIN) & (support.val >= MIN_VAL)])
    insufficient = sorted(set(support.index) - set(classes))
    n_train = support.train.to_dict()

    sep = camera_separation(keys, means, frame_spread, cameras, rng)
    tr_dir = sep[sep.split == "train"].set_index("taxon_group").direction
    va_dir = sep[sep.split == "val"].set_index("taxon_group").direction
    cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))  # noqa: E731
    sep["train_val_direction_cosine"] = sep.taxon_group.map({c: cos(tr_dir[c], va_dir[c]) for c in tr_dir.index})
    d = np.stack([v / np.linalg.norm(v) for v in tr_dir.values])
    cross_class_cos = float((d @ d.T)[~np.eye(len(d), dtype=bool)].mean())
    between = []
    for (c, cam), k in keys[keys.fold0 == "train"].groupby(["taxon_group", "camera"]):
        between.append(np.sqrt(((means[k.index] - means[k.index].mean(0)) ** 2).sum(1).mean()))
    sep.drop(columns="direction").to_csv(a.out / "camera_separation.csv", index=False)

    k_tr = keys[keys.fold0 == "train"].reset_index(drop=True)
    k_va = keys[keys.fold0 == "val"].reset_index(drop=True)
    pca = PCA(n_components=max(p for _, p in SETTINGS), whiten=True, svd_solver="full").fit(means[keys.fold0 == "train"])
    z_tr, z_va = pca.transform(means[keys.fold0 == "train"]), pca.transform(means[keys.fold0 == "val"])

    summaries, tables = [], []
    for cov, dims in SETTINGS:
        s, t = run_setting(cov, dims, z_tr, z_va, k_tr, k_va, classes, n_train, a.seed, rng)
        summaries.append(s)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(a.out / "per_class_density.csv", index=False)
    plots(a.out, z_tr, k_tr, classes, table, cameras)

    sep_tr, sep_va = sep[sep.split == "train"], sep[sep.split == "val"]
    result = {
        "cohort": str(a.cohort), "cohort_sha256": file_sha256(a.cohort), "encoder": meta["encoder"], "model_id": meta["model_id"],
        "feature": meta["feature"], "transform": meta["transform"], "seed": a.seed,
        "observations": {"train": len(k_tr), "val": len(k_va)}, "images": {s: int((cohort.fold0 == s).sum()) for s in ("train", "val")},
        "specimens": {s: int(cohort[cohort.fold0 == s].individual.nunique()) for s in ("train", "val")},
        "frames_per_specimen_camera": keys.frames.value_counts().sort_index().to_dict(),
        "support_rule": {"min_train_specimens": MIN_TRAIN, "min_val_specimens": MIN_VAL, "k_min_train_specimens": K_MIN_SPECIMENS},
        "classes_fitted": len(classes), "classes_insufficient": insufficient,
        "specimens_same_imaging_runs_both_cameras": int((runs[cameras[0]] == runs[cameras[1]]).sum()),
        "specimens_with_one_site_year_lake_dataset": int((meta_cols == 1).all(axis=1).sum()), "specimens_total": len(meta_cols),
        "pca_explained_variance_cumulative": {p: float(pca.explained_variance_ratio_[:p].sum()) for _, p in SETTINGS},
        "camera_separation_native": {
            "train_ratio_median": float(sep_tr.ratio.median()), "val_ratio_median": float(sep_va.ratio.median()),
            "train_classes_p_below_0.05": int((sep_tr.paired_signflip_p < 0.05).sum()),
            "val_classes_p_below_0.05": int((sep_va.paired_signflip_p < 0.05).sum()), "classes": len(sep_tr),
            "train_val_direction_cosine_median": float(sep_tr.train_val_direction_cosine.median()),
            "cross_class_direction_cosine_mean_train": cross_class_cos,
            "frame_rms_within_specimen_median_train": float(sep_tr.mean_frame_rms_within_specimen.median()),
            "specimen_rms_within_class_camera_median_train": float(np.median(between)),
        },
        "settings": summaries, "cpu_seconds": round(time.process_time() - t0, 1),
    }
    (a.out / "results.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
