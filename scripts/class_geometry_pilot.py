"""Within-class and between-class geometry of DINOv3-L specimen embeddings (task.md fb26afc), local and descriptive.

Inputs: the camera-pilot cohort (official fold0 train/val, capped) and its verified feature cache. No new features.
Unit: one embedding per independent specimen = mean of its two camera-level means (each the mean of its frames).
Everything that is fitted (PCA, centroids, covariances, mixtures, directions) uses train specimens only; val
specimens are only scored or projected. Test is not in the cohort.

1. Within class (native 1024-d, train): eigenvalue summary (src.statistics.class_geometry), and camera-pair
   re-identification: share of specimens whose camera-1 view's nearest camera-2 view (same class, train) is its own
   (chance 1/n).
2. Between classes (native, train centroids): radius separation (centroid distance / RMS radius), d' along the
   train centroid direction measured on val specimens (primary closeness), d' on train, top-5 subspace overlap; the
   radius separation is repeated on L2-normalised embeddings and in the global 2-D PCA to show scale/projection effects.
3. One Gaussian or several (global PCA fitted on train specimens, whitened, 8 PCs, reg_covar 1e-3, 5 k-means + 5 random-specimen + one principal-direction-split start per PC, best train likelihood, seed):
   models in src.statistics.class_modality.MODELS, eligibility n*d/2, unreliable if a component has < d+1 specimens.
   Score: log density of each val specimen; differences to G1_full with 95 % bootstrap intervals over val specimens.
   16 PCs is not used: a single full Gaussian there has 152 parameters for 30 train specimens.
   Predeclared per-class verdict over eligible, converged, reliable full/tied mixtures:
     credible multiple groups: some mixture's interval lies above 0 AND the 2-component tied split is stable
       (median val ARI over 80 % train subsamples >= 0.5, both components used by val);
     one Gaussian favoured: every mixture's interval lies below 0;
     inconclusive: otherwise; insufficient evidence: no usable mixture or fewer than 5 val specimens.
   Diagonal models are reported as a diagnostic only.
Plot/montage selection uses train BIC only: top 3 classes by BIC gain of the best 2-component full/tied mixture over
G1_full ("candidate groups") and the bottom 2 ("candidate single cloud").

Usage: class_geometry_pilot.py --cohort cohort.parquet --cohort-summary cohort_summary.json --cache FEATURE_DIR --out DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.features.chunk_store import file_sha256, load_cache
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.statistics.class_geometry import (direction, direction_separation, eigen_summary, principal_axes,
                                           radius_separation, subspace_overlap)
from src.statistics.class_modality import (MODELS, condition_number, eligible, fit_mixture, free_parameters, smallest_component,
                                           split_stability)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PCS, REG, N_BOOT, N_STAB, K_SUB, STABLE_ARI, MIN_VAL = 8, 1e-3, 2000, 200, 5, 0.5, 5
FT = [m for m, (k, c) in MODELS.items() if k > 1 and c != "diag"]


def ci(a: np.ndarray) -> list[float]:
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def reidentification(a: np.ndarray, b: np.ndarray) -> float:
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.mean(d.argmin(1) == np.arange(len(a))))


def within_class(classes, sp_tr, x_tr, cam_tr, xc) -> pd.DataFrame:
    rows = []
    for c in classes:
        k = sp_tr[sp_tr.taxon_group == c]
        ids = k.individual.values
        cams = sorted(cam_tr.camera.unique())
        views = [xc[cam_tr[(cam_tr.taxon_group == c) & (cam_tr.camera == cam)].set_index("individual").loc[ids, "row"].values]
                 for cam in cams]
        rows.append({"taxon_group": c, **eigen_summary(x_tr[k.index]), "camera_reid": reidentification(*views),
                     "camera_reid_chance": 1 / len(ids)})
    return pd.DataFrame(rows)


def between_classes(classes, sp_tr, x_tr, sp_va, x_va, z2_tr) -> pd.DataFrame:
    xn_tr = x_tr / np.linalg.norm(x_tr, axis=1, keepdims=True)
    tr = {c: sp_tr.index[sp_tr.taxon_group == c] for c in classes}
    va = {c: sp_va.index[sp_va.taxon_group == c] for c in classes}
    rows = []
    for a, b in itertools.combinations(classes, 2):
        u = direction(x_tr[tr[a]], x_tr[tr[b]])
        rows.append({"class_a": a, "class_b": b,
                     "dprime_val": direction_separation(x_va[va[a]], x_va[va[b]], u),
                     "dprime_train": direction_separation(x_tr[tr[a]], x_tr[tr[b]], u),
                     "radius_sep": radius_separation(x_tr[tr[a]], x_tr[tr[b]]),
                     "radius_sep_l2norm": radius_separation(xn_tr[tr[a]], xn_tr[tr[b]]),
                     "radius_sep_pca2d": radius_separation(z2_tr[tr[a]], z2_tr[tr[b]]),
                     "centroid_distance": float(np.linalg.norm(x_tr[tr[a]].mean(0) - x_tr[tr[b]].mean(0))),
                     "subspace_overlap_top5": subspace_overlap(x_tr[tr[a]], x_tr[tr[b]], K_SUB)})
    return pd.DataFrame(rows)


def modality(classes, sp_tr, z_tr, sp_va, z_va, cam_tr, zc, seed, rng):
    rows, scores, fits = [], {}, {}
    for c in classes:
        zt, zv = z_tr[sp_tr.taxon_group == c], z_va[sp_va.taxon_group == c]
        row = {"taxon_group": c, "n_train": len(zt), "n_val": len(zv)}
        scores[c], fits[c] = {}, {}
        for m, (k, cov) in MODELS.items():
            row[f"{m}_params"] = free_parameters(k, PCS, cov)
            if not eligible(k, PCS, cov, len(zt)):
                row[f"{m}_status"] = "ineligible"
                continue
            try:
                gm = fit_mixture(zt, k, cov, REG, seed)
            except (RuntimeError, ValueError) as e:
                row[f"{m}_status"] = f"failed: {e}"
                continue
            small = smallest_component(gm, zt)
            row.update({f"{m}_status": "unreliable" if small < PCS + 1 else "ok", f"{m}_smallest_component": small,
                        f"{m}_condition": condition_number(gm), f"{m}_bic_train": gm.bic(zt)})
            scores[c][m], fits[c][m] = gm.score_samples(zv), gm
        idx = rng.integers(0, len(zv), size=(N_BOOT, len(zv)))
        for m in [m for m in MODELS if m in scores[c] and m != "G1_full"]:
            base = "G1_diag" if m == "G2_diag" else "G1_full"
            diff = scores[c][m] - scores[c][base]
            row[f"{m}-{base}"] = float(diff.mean())
            row[f"{m}-{base}_ci95"] = ci(diff[idx].mean(1))
            scores[c][f"{m}-{base}_boot"] = diff[idx].mean(1)
        st = split_stability(zt, zv, 2, "tied", REG, seed, N_STAB)
        row.update({f"G2_tied_split_{key}": v for key, v in st.items()})
        ref = fits[c]["G2_tied"]
        views = [zc[cam_tr[(cam_tr.taxon_group == c) & (cam_tr.camera == cam)].sort_values("individual").row.values]
                 for cam in sorted(cam_tr.camera.unique())]
        la, lb = ref.predict(views[0]), ref.predict(views[1])
        row["G2_tied_both_views_same_component_train"] = float(np.mean(la == lb))
        usable = [m for m in FT if row.get(f"{m}_status") == "ok"]
        row["usable_mixtures"] = ",".join(usable)
        if not usable or len(zv) < MIN_VAL:
            verdict = "insufficient evidence"
        elif any(row[f"{m}-G1_full_ci95"][0] > 0 for m in usable) and st["ari_val_median"] >= STABLE_ARI \
                and st["val_components_used"] == 2:
            verdict = "credible multiple groups"
        elif all(row[f"{m}-G1_full_ci95"][1] < 0 for m in usable):
            verdict = "one Gaussian favoured"
        else:
            verdict = "inconclusive"
        row["verdict"] = verdict
        two = [row[f"{m}_bic_train"] for m in ("G2_full", "G2_tied") if f"{m}_bic_train" in row]
        row["bic_gain_best2_train"] = row["G1_full_bic_train"] - min(two)
        rows.append(row)
    return pd.DataFrame(rows), scores, fits


def plots(out, classes, sel, per_class, pairs, sp_tr, x_tr, sp_va, x_va, cam_tr, xc, fits, z_tr, z_va, eig):
    # 1. eigenvalue spectra (share of total variance), all classes; selected classes highlighted
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c in classes:
        lam = eig[c]
        style = dict(lw=1.8, label=c) if c in sel["all"] else dict(c="0.75", lw=0.6)
        ax.plot(np.arange(1, 16), lam[:15] / lam.sum(), **style)
    ax.set(yscale="log", xlabel="principal direction (within class, native, train)", ylabel="share of class variance",
           title="Within-class eigenvalue spectra (30 train specimens per class)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "eigenspectra.png", dpi=130)
    plt.close(fig)

    # 2. internal geometry of the selected classes: class-local PCA; top = G2_tied components, bottom = camera views
    show = sel["all"]
    fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.4))
    for j, c in enumerate(show):
        it, iv = sp_tr.index[sp_tr.taxon_group == c], sp_va.index[sp_va.taxon_group == c]
        mu = x_tr[it].mean(0)
        axes_c = principal_axes(x_tr[it])[1][:, :2]
        pt, pv = (x_tr[it] - mu) @ axes_c, (x_va[iv] - mu) @ axes_c
        lt, lv = fits[c]["G2_tied"].predict(z_tr[it]), fits[c]["G2_tied"].predict(z_va[iv])
        ax = axes[0, j]
        for comp, col in ((0, "#2ca02c"), (1, "#9467bd")):
            ax.scatter(*pt[lt == comp].T, s=14, c=col, label=f"train comp {comp}")
            ax.scatter(*pv[lv == comp].T, s=24, facecolors="none", edgecolors=col, label=f"val comp {comp}")
        ax.set_title(f"{c}\n{per_class.set_index('taxon_group').loc[c, 'verdict']}", fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6)
        ax = axes[1, j]
        k = cam_tr[cam_tr.taxon_group == c]
        v1 = xc[k[k.camera == k.camera.min()].sort_values("individual").row.values]
        v2 = xc[k[k.camera == k.camera.max()].sort_values("individual").row.values]
        p1, p2 = (v1 - mu) @ axes_c, (v2 - mu) @ axes_c
        for a, b in zip(p1, p2):
            ax.plot([a[0], b[0]], [a[1], b[1]], c="0.8", lw=0.6, zorder=0)
        ax.scatter(*p1.T, s=10, c="#1f77b4", label="camera 1 view")
        ax.scatter(*p2.T, s=10, c="#ff7f0e", label="camera 2 view")
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6)
    fig.suptitle("Selected classes, class-local PC1 vs PC2 (native, train axes). Top: 2-component tied mixture "
                 "(fitted in global 8-PC space). Bottom: camera views of the same train specimens", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "class_internal_geometry.png", dpi=130)
    plt.close(fig)

    # 3. pair separation heatmap (d' on val along train direction) and radius separation: native vs 2-D PCA
    n = len(classes)
    mat = np.zeros((n, n))
    pos = {c: i for i, c in enumerate(classes)}
    for r in pairs.itertuples():
        mat[pos[r.class_a], pos[r.class_b]] = mat[pos[r.class_b], pos[r.class_a]] = r.dprime_val
    cent = np.stack([x_tr[sp_tr.index[sp_tr.taxon_group == c]].mean(0) for c in classes])
    order = leaves_list(linkage(cent, "average"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    im = axes[0].imshow(np.minimum(mat[np.ix_(order, order)], 6), cmap="viridis")
    axes[0].set_xticks(range(n), [classes[i] for i in order], rotation=90, fontsize=5)
    axes[0].set_yticks(range(n), [classes[i] for i in order], fontsize=5)
    axes[0].set_title("d' between classes (val specimens, train centroid direction; capped at 6)", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    axes[1].scatter(pairs.radius_sep, pairs.radius_sep_pca2d, s=5, c=np.minimum(pairs.dprime_val, 6), cmap="viridis")
    lim = max(pairs.radius_sep.max(), pairs.radius_sep_pca2d.max())
    axes[1].plot([0, lim], [0, lim], c="k", lw=0.6)
    axes[1].set(xlabel="radius separation, native 1024-d", ylabel="radius separation, global 2-D PCA",
                title="Same pairs: 2-D projection vs native (colour = d' val)")
    fig.tight_layout()
    fig.savefig(out / "class_pair_separation.png", dpi=130)
    plt.close(fig)

    # 4. closest pairs: projections on the train centroid direction
    close = pairs.nsmallest(6, "dprime_val")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, r in zip(axes.ravel(), close.itertuples()):
        ta, tb = sp_tr.index[sp_tr.taxon_group == r.class_a], sp_tr.index[sp_tr.taxon_group == r.class_b]
        va, vb = sp_va.index[sp_va.taxon_group == r.class_a], sp_va.index[sp_va.taxon_group == r.class_b]
        u = direction(x_tr[ta], x_tr[tb])
        bins = np.histogram_bin_edges(np.concatenate([x_tr[ta] @ u, x_tr[tb] @ u, x_va[va] @ u, x_va[vb] @ u]), 20)
        ax.hist(x_tr[ta] @ u, bins, alpha=0.5, color="#1f77b4", label=f"{r.class_a} train")
        ax.hist(x_tr[tb] @ u, bins, alpha=0.5, color="#d62728", label=f"{r.class_b} train")
        ax.plot(x_va[va] @ u, np.full(len(va), -0.4), "|", c="#1f77b4", ms=12, label="val")
        ax.plot(x_va[vb] @ u, np.full(len(vb), -0.4), "|", c="#d62728", ms=12)
        ax.set_title(f"d' val {r.dprime_val:.2f}, d' train {r.dprime_train:.2f}", fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Closest class pairs: specimens projected on the train centroid-difference direction (native)")
    fig.tight_layout()
    fig.savefig(out / "closest_pairs.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--cohort-summary", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cohort = pd.read_parquet(a.cohort)
    if not set(cohort.fold0) <= {"train", "val"}:
        raise RuntimeError("cohort holds rows outside fold0 train/val")
    x, meta = load_cache(a.cache, cohort.img.tolist())
    cam_keys, cam_means, _ = specimen_camera_means(x, cohort)
    sp_keys, sp = specimen_means(cam_keys, cam_means)
    if (sp_keys.cameras != 2).any():
        raise RuntimeError("a specimen lacks a camera")
    if set(sp_keys[sp_keys.fold0 == "train"].individual) & set(sp_keys[sp_keys.fold0 == "val"].individual):
        raise RuntimeError("specimen in both train and val")
    tr, va = (sp_keys.fold0 == "train").values, (sp_keys.fold0 == "val").values
    sp_tr, sp_va = sp_keys[tr].reset_index(drop=True), sp_keys[va].reset_index(drop=True)
    x_tr, x_va = sp[tr], sp[va]
    ctr = (cam_keys.fold0 == "train").values
    cam_tr = cam_keys[ctr].reset_index(drop=True).assign(row=lambda d: np.arange(len(d)))
    xc = cam_means[ctr]
    support = sp_keys.groupby(["taxon_group", "fold0"]).size().unstack(fill_value=0)
    classes = sorted(support.index[(support.train >= 15) & (support.val >= MIN_VAL)])
    summary_in = json.loads(a.cohort_summary.read_text())

    geo = within_class(classes, sp_tr, x_tr, cam_tr, xc)
    pca2 = PCA(n_components=2, svd_solver="full").fit(x_tr)
    pairs = between_classes(classes, sp_tr, x_tr, sp_va, x_va, pca2.transform(x_tr))
    pca = PCA(n_components=PCS, whiten=True, svd_solver="full").fit(x_tr)
    z_tr, z_va, zc = pca.transform(x_tr), pca.transform(x_va), pca.transform(xc)
    mod, scores, fits = modality(classes, sp_tr, z_tr, sp_va, z_va, cam_tr, zc, a.seed, rng)
    per_class = geo.merge(mod, on="taxon_group")
    comp_rows = []
    for c in classes:
        for keys, z in ((sp_tr, z_tr), (sp_va, z_va)):
            m = (keys.taxon_group == c).values
            resp = fits[c]["G2_tied"].predict_proba(z[m])
            comp_rows.append(keys[m][["taxon_group", "fold0", "individual"]].assign(
                g2_tied_component=resp.argmax(1), g2_tied_responsibility=resp.max(1),
                g1_full_log_density=fits[c]["G1_full"].score_samples(z[m])))
    pd.concat(comp_rows).to_csv(a.out / "specimen_components.csv", index=False)

    nearest = []
    for c in classes:
        p = pairs[(pairs.class_a == c) | (pairs.class_b == c)].copy()
        p["other"] = np.where(p.class_a == c, p.class_b, p.class_a)
        r = p.nsmallest(1, "dprime_val").iloc[0]
        q = p.nsmallest(1, "radius_sep").iloc[0]
        nearest.append({"taxon_group": c, "nearest_by_dprime_val": r.other, "dprime_val": r.dprime_val,
                        "dprime_train": r.dprime_train, "radius_sep": r.radius_sep, "subspace_overlap_top5": r.subspace_overlap_top5,
                        "nearest_by_radius_sep": q.other, "radius_sep_min": q.radius_sep})
    nearest = pd.DataFrame(nearest)
    per_class = per_class.merge(nearest[["taxon_group", "nearest_by_dprime_val", "dprime_val"]], on="taxon_group")

    rank = per_class.sort_values(["bic_gain_best2_train", "taxon_group"], ascending=[False, True]).taxon_group.tolist()
    sel = {"candidate_groups": rank[:3], "candidate_single": rank[-2:]}
    sel["all"] = sel["candidate_groups"] + sel["candidate_single"]
    eig = {c: principal_axes(x_tr[sp_tr.taxon_group == c])[0] for c in classes}
    montage = {}
    for c in sel["all"]:
        it = sp_tr.index[sp_tr.taxon_group == c]
        resp = fits[c]["G2_tied"].predict_proba(z_tr[it])
        ids = sp_tr.individual.values[it]
        montage[c] = {f"component_{k}": [ids[i] for i in np.lexsort((ids, -resp[:, k]))[:4]] for k in (0, 1)}
        g1 = fits[c]["G1_full"].score_samples(z_tr[it])
        montage[c]["outliers_lowest_G1_density"] = [ids[i] for i in np.argsort(g1)[:2]]
    (a.out / "selection.json").write_text(json.dumps({**sel, "montage_specimens": montage}, indent=1))

    plots(a.out, classes, sel, per_class, pairs, sp_tr, x_tr, sp_va, x_va, cam_tr, xc, fits, z_tr, z_va, eig)
    per_class.to_csv(a.out / "per_class.csv", index=False)
    pairs.sort_values("dprime_val").to_csv(a.out / "class_pairs.csv", index=False)
    nearest.to_csv(a.out / "nearest_class.csv", index=False)

    diffs = {}
    for key in [f"{m}-G1_full" for m in FT] + ["G2_diag-G1_diag"]:
        have = [c for c in classes if f"{key}_boot" in scores[c]]
        if not have:  # model ineligible or failed in every class
            continue
        boot = np.mean([scores[c][f"{key}_boot"] for c in have], axis=0)
        col = per_class.set_index("taxon_group").loc[have, key]
        civ = per_class.set_index("taxon_group").loc[have, f"{key}_ci95"]
        diffs[key] = {"classes": len(have), "mean": float(col.mean()), "ci95": ci(boot),
                      "classes_better": int((col > 0).sum()), "classes_ci_above_0": int(sum(v[0] > 0 for v in civ)),
                      "classes_ci_below_0": int(sum(v[1] < 0 for v in civ))}
    status = {m: per_class[f"{m}_status"].fillna("not fitted").value_counts().to_dict() for m in MODELS}
    result = {
        "cohort_sha256": file_sha256(a.cohort), "encoder": meta["encoder"], "model_id": meta["model_id"], "seed": a.seed,
        "unit": "specimen = mean of its two camera-level means", "classes": len(classes),
        "excluded_upstream": summary_in["excluded_classes"],
        "specimens": {"train": len(sp_tr), "val": len(sp_va)},
        "val_specimens_per_class_min_max": [int(per_class.n_val.min()), int(per_class.n_val.max())],
        "pca_8_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_2_explained_variance": float(pca2.explained_variance_ratio_.sum()),
        "within_class_native": {k: [float(per_class[k].min()), float(per_class[k].median()), float(per_class[k].max())]
                                for k in ("rms_radius", "top1_share", "top5_share", "effective_rank", "participation_ratio", "camera_reid")},
        "within_class_note": "[min, median, max] over classes; ranks bounded by n-1 = 29",
        "between_classes": {
            "pairs": len(pairs), "dprime_val_below_1": int((pairs.dprime_val < 1).sum()),
            "dprime_val_below_2": int((pairs.dprime_val < 2).sum()),
            "radius_sep_below_1": int((pairs.radius_sep < 1).sum()), "radius_sep_pca2d_below_1": int((pairs.radius_sep_pca2d < 1).sum()),
            "spearman_radius_native_vs_l2norm": float(spearmanr(pairs.radius_sep, pairs.radius_sep_l2norm)[0]),
            "spearman_radius_native_vs_pca2d": float(spearmanr(pairs.radius_sep, pairs.radius_sep_pca2d)[0]),
            "spearman_dprime_val_vs_radius_native": float(spearmanr(pairs.dprime_val, pairs.radius_sep)[0]),
            "closest_pairs": json.loads(pairs.nsmallest(10, "dprime_val").to_json(orient="records")),
        },
        "modality": {"pcs": PCS, "reg_covar": REG, "eligibility": "free parameters <= n_train * d / 2",
                     "params": {m: free_parameters(k, PCS, cov) for m, (k, cov) in MODELS.items()},
                     "status_counts": status, "differences": diffs,
                     "verdicts": per_class.verdict.value_counts().to_dict(),
                     "verdict_classes": {v: sorted(g.taxon_group) for v, g in per_class.groupby("verdict")},
                     "G2_tied_split_ari_val_median_over_classes": float(per_class.G2_tied_split_ari_val_median.median()),
                     "G2_tied_split_stable_classes": int(((per_class.G2_tied_split_ari_val_median >= STABLE_ARI)
                                                          & (per_class.G2_tied_split_val_components_used == 2)).sum()),
                     "G2_tied_both_views_same_component_median": float(per_class.G2_tied_both_views_same_component_train.median()),
                     "condition_max": {m: float(per_class[f"{m}_condition"].max()) for m in MODELS if f"{m}_condition" in per_class}},
        "selection_train_bic": sel, "wall_seconds": round(time.time() - t0, 1),
    }
    (a.out / "results.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
