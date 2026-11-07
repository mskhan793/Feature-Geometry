"""Within-class and between-class geometry of DINOv3-L specimen embeddings, and one Gaussian vs a small mixture per
class (task.md fb26afc on the 30-specimen cohort; task.md fb46e1f on the expanded cohort). Local and descriptive.

Inputs: a cohort (official fold0 train/val rows only) and the verified feature caches holding its images (looked up by
image name, src.features.feature_lookup). Unit: one embedding per independent specimen = mean of its camera-level means
(each the mean of its frames; a specimen imaged by one camera uses that camera). Everything fitted (PCA, centroids,
covariances, mixtures, directions) uses train specimens only; val specimens are only scored or projected. Classes need
at least 15 train and 5 val specimens; the others are listed with their counts.

1. Within class (native 1024-d, train): eigenvalue summary (src.statistics.class_geometry) on all train specimens and
   on the first 30 in sha1(seed:individual) order (ranks depend on n; the n=30 values compare across classes and runs);
   camera-pair re-identification on specimens with both cameras (share whose camera-1 view's nearest camera-2 view is
   its own; chance 1/n); median RMS distance of a specimen-camera's frames to their mean.
2. Between classes (native, train centroids): radius separation, d' along the train centroid direction measured on val
   specimens (primary closeness), d' on train, top-5 subspace overlap; radius separation also on L2-normalised
   embeddings and in the global 2-D PCA.
3. One Gaussian or several, for each of ``--pcs`` (global PCA fitted on train specimens, whitened, reg_covar 1e-3,
   src.statistics.class_modality.fit_mixture starts, seed): models in MODELS, eligible if free parameters <= n*d/2,
   unreliable if a component has < d+1 specimens. A class is analysed at d only if G1_full and G2_tied are eligible. Score: log density of each val specimen; differences to G1_full with
   95 % bootstrap intervals over val specimens (per-class generator seeded by seed, class index and d).
   Sensitivity (not used in the verdict): GD_tied, one Gaussian per source ``dataset`` with shared covariance, mixed
   with train proportions (SourceGaussians).
   Predeclared per-class verdict over eligible, converged, reliable full/tied mixtures:
     credible multiple groups: some mixture's interval lies above 0 AND the 2-component tied split is stable
       (median val ARI over ``--n-stab`` 80 % train subsamples >= 0.5, both components used by val);
     one Gaussian favoured: every mixture's interval lies below 0;
     inconclusive: otherwise; insufficient evidence: no usable mixture.
   Diagonal models are a diagnostic only.
Plot/montage selection uses train BIC only, at the largest ``--pcs``: top 3 classes by BIC gain of the best 2-component
full/tied mixture over G1_full ("candidate groups") and the bottom 2 ("candidate single cloud").

Usage: class_geometry_pilot.py --cohort cohort.parquet --caches MANIFEST=CACHE_DIR ... --out DIR [--pcs 8 16]
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
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from src.data.expanded_cohort import stable_order
from src.features.chunk_store import file_sha256
from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.statistics.class_geometry import (direction, direction_separation, eigen_summary, principal_axes,
                                           radius_separation, subspace_overlap)
from src.statistics.class_modality import (MODELS, SourceGaussians, condition_number, eligible, fit_mixture,
                                           free_parameters, smallest_component, split_stability)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REG, N_BOOT, K_SUB, STABLE_ARI, MIN_TRAIN, MIN_VAL, N_FIXED = 1e-3, 2000, 5, 0.5, 15, 5, 30
FT = [m for m, (k, c) in MODELS.items() if k > 1 and c != "diag"]


def ci(a: np.ndarray) -> list[float]:
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def reidentification(a: np.ndarray, b: np.ndarray) -> float:
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.mean(d.argmin(1) == np.arange(len(a))))


def paired_views(cam_keys: pd.DataFrame, ids) -> tuple[np.ndarray, np.ndarray]:
    """Rows of camera 1 and camera 2 views for the given specimens that have both, in the same specimen order."""
    cams = sorted(cam_keys.camera.unique())
    k = cam_keys[cam_keys.individual.isin(ids)]
    both = k.groupby("individual").camera.nunique() == len(cams)
    keep = sorted(both.index[both])
    rows = [k[k.camera == c].set_index("individual").loc[keep, "row"].values for c in cams]
    return rows[0], rows[1]


def within_class(classes, sp_tr, x_tr, cam_tr, xc, frame_rms, seed) -> pd.DataFrame:
    rows = []
    for c in classes:
        k = sp_tr[sp_tr.taxon_group == c]
        first = k.index[np.argsort(stable_order(k.individual, seed).values)[:N_FIXED]]
        r1, r2 = paired_views(cam_tr, k.individual)
        fixed = {f"{key}_n30": v for key, v in eigen_summary(x_tr[first]).items() if key != "n"}
        rows.append({"taxon_group": c, **eigen_summary(x_tr[k.index]), **fixed,
                     "camera_reid": reidentification(xc[r1], xc[r2]), "camera_reid_chance": 1 / len(r1),
                     "paired_specimens": len(r1), "frame_rms_within_specimen_median": float(np.median(
                         frame_rms[cam_tr.index[cam_tr.taxon_group == c]]))})
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


def fit_class(c, ci_, zt, zv, groups_t, views, d, seed, n_stab):
    """All models of one class at dimension d. Returns the table row, bootstrap differences and the two fits needed
    later (G1_full, G2_tied)."""
    with threadpool_limits(1):
        row = {"taxon_group": c, "pcs": d, "n_train": len(zt), "n_val": len(zv)}
        scores, fits = {}, {}
        for m, (k, cov) in MODELS.items():
            row[f"{m}_params"] = free_parameters(k, d, cov)
            if not eligible(k, d, cov, len(zt)):
                row[f"{m}_status"] = "ineligible"
                continue
            try:
                gm = fit_mixture(zt, k, cov, REG, seed)
            except (RuntimeError, ValueError) as e:
                row[f"{m}_status"] = f"failed: {e}"
                continue
            small = smallest_component(gm, zt)
            row.update({f"{m}_status": "unreliable" if small < d + 1 else "ok", f"{m}_smallest_component": small,
                        f"{m}_condition": condition_number(gm), f"{m}_bic_train": gm.bic(zt)})
            scores[m], fits[m] = gm.score_samples(zv), gm
        gd = SourceGaussians(REG).fit(zt, groups_t)
        row.update({"GD_tied_groups": len(gd.groups_), "GD_tied_params": gd.n_parameters(),
                    "GD_tied_status": "ok" if eligible(1, d, "full", len(zt)) else "ineligible"})
        if row["GD_tied_status"] == "ok":
            scores["GD_tied"] = gd.score_samples(zv)
        rng = np.random.default_rng([seed, ci_, d])
        idx = rng.integers(0, len(zv), size=(N_BOOT, len(zv)))
        boots = {}
        for m in [m for m in [*MODELS, "GD_tied"] if m in scores and m != "G1_full"]:
            base = "G1_diag" if m == "G2_diag" else "G1_full"
            diff = scores[m] - scores[base]
            boots[f"{m}-{base}"] = diff[idx].mean(1)
            row[f"{m}-{base}"] = float(diff.mean())
            row[f"{m}-{base}_ci95"] = ci(boots[f"{m}-{base}"])
        st = split_stability(zt, zv, 2, "tied", REG, seed, n_stab)
        row.update({f"G2_tied_split_{key}": v for key, v in st.items()})
        la, lb = fits["G2_tied"].predict(views[0]), fits["G2_tied"].predict(views[1])
        row["G2_tied_both_views_same_component_train"] = float(np.mean(la == lb))
        usable = [m for m in FT if row.get(f"{m}_status") == "ok"]
        row["usable_mixtures"] = ",".join(usable)
        if not usable:
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
    return row, boots, {"G1_full": fits["G1_full"], "G2_tied": fits["G2_tied"]}


def summarise(per_d: pd.DataFrame, boots: dict, classes, d) -> dict:
    diffs = {}
    for key in [f"{m}-G1_full" for m in [*FT, "GD_tied"]] + ["G2_diag-G1_diag"]:
        have = [c for c in classes if key in boots[c]]
        if not have:
            continue
        t = per_d.set_index("taxon_group").loc[have]
        diffs[key] = {"classes": len(have), "mean": float(t[key].mean()), "ci95": ci(np.mean([boots[c][key] for c in have], axis=0)),
                      "classes_better": int((t[key] > 0).sum()), "classes_ci_above_0": int(sum(v[0] > 0 for v in t[f"{key}_ci95"])),
                      "classes_ci_below_0": int(sum(v[1] < 0 for v in t[f"{key}_ci95"]))}
    return {"pcs": d, "params": {m: free_parameters(k, d, cov) for m, (k, cov) in MODELS.items()},
            "status_counts": {m: per_d[f"{m}_status"].fillna("not fitted").value_counts().to_dict() for m in [*MODELS, "GD_tied"]},
            "differences": diffs, "verdicts": per_d.verdict.value_counts().to_dict(),
            "verdict_classes": {v: sorted(g.taxon_group) for v, g in per_d.groupby("verdict")},
            "G2_tied_split_ari_val_median_over_classes": float(per_d.G2_tied_split_ari_val_median.median()),
            "G2_tied_split_stable_classes": int(((per_d.G2_tied_split_ari_val_median >= STABLE_ARI)
                                                 & (per_d.G2_tied_split_val_components_used == 2)).sum()),
            "G2_tied_both_views_same_component_median": float(per_d.G2_tied_both_views_same_component_train.median()),
            "condition_max": {m: float(per_d[f"{m}_condition"].max()) for m in MODELS if f"{m}_condition" in per_d}}


def plots(out, classes, sel, verdict, pairs, sp_tr, x_tr, sp_va, x_va, cam_tr, xc, fits, z_tr, z_va, d):
    # 1. eigenvalue spectra (share of total variance), all classes; selected classes highlighted
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c in classes:
        lam = principal_axes(x_tr[sp_tr.taxon_group == c])[0]
        style = dict(lw=1.8, label=c) if c in sel["all"] else dict(c="0.75", lw=0.6)
        ax.plot(np.arange(1, 16), lam[:15] / lam.sum(), **style)
    ax.set(yscale="log", xlabel="principal direction (within class, native, train)", ylabel="share of class variance",
           title="Within-class eigenvalue spectra (all train specimens per class)")
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
            ax.scatter(*pt[lt == comp].T, s=8, c=col, label=f"train comp {comp}")
            ax.scatter(*pv[lv == comp].T, s=14, facecolors="none", edgecolors=col, label=f"val comp {comp}")
        ax.set_title(f"{c}\n{verdict[c]}", fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6)
        ax = axes[1, j]
        r1, r2 = paired_views(cam_tr, sp_tr.individual[it])
        p1, p2 = (xc[r1] - mu) @ axes_c, (xc[r2] - mu) @ axes_c
        for a, b in zip(p1, p2):
            ax.plot([a[0], b[0]], [a[1], b[1]], c="0.8", lw=0.4, zorder=0)
        ax.scatter(*p1.T, s=6, c="#1f77b4", label="camera 1 view")
        ax.scatter(*p2.T, s=6, c="#ff7f0e", label="camera 2 view")
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6)
    fig.suptitle(f"Selected classes, class-local PC1 vs PC2 (native, train axes). Top: 2-component tied mixture "
                 f"(fitted in global {d}-PC space). Bottom: camera views of the same train specimens", fontsize=9)
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
        bins = np.histogram_bin_edges(np.concatenate([x_tr[ta] @ u, x_tr[tb] @ u, x_va[va] @ u, x_va[vb] @ u]), 30)
        ax.hist(x_tr[ta] @ u, bins, alpha=0.5, color="#1f77b4", density=True, label=f"{r.class_a} train")
        ax.hist(x_tr[tb] @ u, bins, alpha=0.5, color="#d62728", density=True, label=f"{r.class_b} train")
        ax.hist(x_va[va] @ u, bins, histtype="step", color="#1f77b4", density=True, label="val (outline)")
        ax.hist(x_va[vb] @ u, bins, histtype="step", color="#d62728", density=True)
        ax.set_title(f"d' val {r.dprime_val:.2f}, d' train {r.dprime_train:.2f}", fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Closest class pairs: specimens projected on the train centroid-difference direction (native)")
    fig.tight_layout()
    fig.savefig(out / "closest_pairs.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+", help="MANIFEST=CACHE_DIR pairs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pcs", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--n-stab", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(a.cohort)
    if not set(cohort.fold0) <= {"train", "val"}:
        raise RuntimeError("cohort holds rows outside fold0 train/val")
    pairs_in = [tuple(Path(p) for p in s.split("=", 1)) for s in a.caches]
    x, meta = features_by_name(pairs_in, cohort.img.tolist())
    cam_keys, cam_means, frame_rms = specimen_camera_means(x, cohort)
    sp_keys, sp = specimen_means(cam_keys, cam_means)
    if set(sp_keys[sp_keys.fold0 == "train"].individual) & set(sp_keys[sp_keys.fold0 == "val"].individual):
        raise RuntimeError("specimen in both train and val")
    source = cohort.groupby("individual").dataset.first()
    tr, va = (sp_keys.fold0 == "train").values, (sp_keys.fold0 == "val").values
    sp_tr, sp_va = sp_keys[tr].reset_index(drop=True), sp_keys[va].reset_index(drop=True)
    x_tr, x_va = sp[tr], sp[va]
    ctr = (cam_keys.fold0 == "train").values
    cam_tr = cam_keys[ctr].reset_index(drop=True).assign(row=lambda d: np.arange(len(d)))
    xc, frame_rms_tr = cam_means[ctr], frame_rms[ctr]
    support = sp_keys.groupby(["taxon_group", "fold0"]).size().unstack(fill_value=0)
    ok = (support.get("train", 0) >= MIN_TRAIN) & (support.get("val", 0) >= MIN_VAL)
    classes = sorted(support.index[ok])
    insufficient = {c: support.loc[c].to_dict() for c in support.index[~ok]}

    geo = within_class(classes, sp_tr, x_tr, cam_tr, xc, frame_rms_tr, a.seed)
    pca2 = PCA(n_components=2, svd_solver="full").fit(x_tr)
    pairs = between_classes(classes, sp_tr, x_tr, sp_va, x_va, pca2.transform(x_tr))
    pca = PCA(n_components=max(a.pcs), whiten=True, svd_solver="full").fit(x_tr)
    z_tr_all, z_va_all, zc_all = pca.transform(x_tr), pca.transform(x_va), pca.transform(xc)

    tables, summaries, fits_by_d, comp_rows, classes_by_d = [], {}, {}, [], {}
    n_tr = sp_tr.taxon_group.value_counts()
    for d in a.pcs:
        z_tr, z_va, zc = z_tr_all[:, :d], z_va_all[:, :d], zc_all[:, :d]
        classes_by_d[d] = [c for c in classes if eligible(1, d, "full", n_tr[c]) and eligible(2, d, "tied", n_tr[c])]
        jobs = []
        for i, c in enumerate(classes_by_d[d]):
            mt, mv = (sp_tr.taxon_group == c).values, (sp_va.taxon_group == c).values
            r1, r2 = paired_views(cam_tr, sp_tr.individual[mt])
            jobs.append(delayed(fit_class)(c, i, z_tr[mt], z_va[mv], source.reindex(sp_tr.individual[mt]).values,
                                           (zc[r1], zc[r2]), d, a.seed, a.n_stab))
        out = Parallel(n_jobs=a.jobs)(jobs)
        per_d = pd.DataFrame([o[0] for o in out])
        boots = {c: o[1] for c, o in zip(classes_by_d[d], out)}
        fits_by_d[d] = {c: o[2] for c, o in zip(classes_by_d[d], out)}
        summaries[str(d)] = summarise(per_d, boots, classes_by_d[d], d)
        summaries[str(d)]["classes_not_analysed_too_few_train"] = sorted(set(classes) - set(classes_by_d[d]))
        tables.append(per_d)
        for c in classes_by_d[d]:
            for keys, z in ((sp_tr, z_tr), (sp_va, z_va)):
                m = (keys.taxon_group == c).values
                resp = fits_by_d[d][c]["G2_tied"].predict_proba(z[m])
                comp_rows.append(keys[m][["taxon_group", "fold0", "individual"]].assign(
                    pcs=d, g2_tied_component=resp.argmax(1), g2_tied_responsibility=resp.max(1),
                    g1_full_log_density=fits_by_d[d][c]["G1_full"].score_samples(z[m])))
    modality = pd.concat(tables, ignore_index=True)
    modality.to_csv(a.out / "per_class_modality.csv", index=False)
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
    geo = geo.merge(nearest[["taxon_group", "nearest_by_dprime_val", "dprime_val"]], on="taxon_group")
    geo.to_csv(a.out / "per_class_geometry.csv", index=False)

    d_sel = max(a.pcs)
    m_sel = modality[modality.pcs == d_sel]
    rank = m_sel.sort_values(["bic_gain_best2_train", "taxon_group"], ascending=[False, True]).taxon_group.tolist()
    sel = {"pcs": d_sel, "candidate_groups": rank[:3], "candidate_single": rank[-2:]}
    sel["all"] = sel["candidate_groups"] + sel["candidate_single"]
    montage = {}
    for c in sel["all"]:
        it = sp_tr.index[sp_tr.taxon_group == c]
        z = z_tr_all[it, :d_sel]
        resp = fits_by_d[d_sel][c]["G2_tied"].predict_proba(z)
        ids = sp_tr.individual.values[it]
        montage[c] = {f"component_{k}": [ids[i] for i in np.lexsort((ids, -resp[:, k]))[:4]] for k in (0, 1)}
        g1 = fits_by_d[d_sel][c]["G1_full"].score_samples(z)
        montage[c]["outliers_lowest_G1_density"] = [ids[i] for i in np.argsort(g1)[:2]]
    (a.out / "selection.json").write_text(json.dumps({**sel, "montage_specimens": montage}, indent=1))
    plots(a.out, classes, sel, m_sel.set_index("taxon_group").verdict.to_dict(), pairs, sp_tr, x_tr, sp_va, x_va,
          cam_tr, xc, fits_by_d[d_sel], z_tr_all[:, :d_sel], z_va_all[:, :d_sel], d_sel)
    pairs.sort_values("dprime_val").to_csv(a.out / "class_pairs.csv", index=False)
    nearest.to_csv(a.out / "nearest_class.csv", index=False)

    result = {
        "cohort_sha256": file_sha256(a.cohort), "encoder": meta["encoder"], "model_id": meta["model_id"],
        "transform": meta["transform"], "tf32": meta.get("tf32"), "seed": a.seed, "n_stab": a.n_stab,
        "unit": "specimen = mean of its camera-level means", "classes": len(classes),
        "classes_insufficient_support": insufficient,
        "specimens": {"train": len(sp_tr), "val": len(sp_va)},
        "specimens_one_camera": {s: int((sp_keys[sp_keys.fold0 == s].cameras == 1).sum()) for s in ("train", "val")},
        "train_specimens_per_class_min_median_max": [int(geo.n.min()), float(geo.n.median()), int(geo.n.max())],
        "val_specimens_per_class_min_median_max": [int(support.loc[classes, "val"].min()), float(support.loc[classes, "val"].median()),
                                                   int(support.loc[classes, "val"].max())],
        "pca_explained_variance": {str(d): float(pca.explained_variance_ratio_[:d].sum()) for d in a.pcs},
        "pca_2_explained_variance": float(pca2.explained_variance_ratio_.sum()),
        "within_class_native": {k: [float(geo[k].min()), float(geo[k].median()), float(geo[k].max())]
                                for k in ("rms_radius", "top1_share", "top5_share", "effective_rank", "participation_ratio",
                                          "effective_rank_n30", "participation_ratio_n30", "top1_share_n30", "camera_reid",
                                          "frame_rms_within_specimen_median")},
        "within_class_note": "[min, median, max] over classes; *_n30 = first 30 train specimens in sha1(seed:individual) order",
        "between_classes": {
            "pairs": len(pairs), "dprime_val_below_1": int((pairs.dprime_val < 1).sum()),
            "dprime_val_below_2": int((pairs.dprime_val < 2).sum()),
            "radius_sep_below_1": int((pairs.radius_sep < 1).sum()), "radius_sep_pca2d_below_1": int((pairs.radius_sep_pca2d < 1).sum()),
            "pca2d_above_native": int((pairs.radius_sep_pca2d > pairs.radius_sep).sum()),
            "spearman_radius_native_vs_l2norm": float(spearmanr(pairs.radius_sep, pairs.radius_sep_l2norm)[0]),
            "spearman_radius_native_vs_pca2d": float(spearmanr(pairs.radius_sep, pairs.radius_sep_pca2d)[0]),
            "spearman_dprime_val_vs_radius_native": float(spearmanr(pairs.dprime_val, pairs.radius_sep)[0]),
            "closest_pairs": json.loads(pairs.nsmallest(10, "dprime_val").to_json(orient="records")),
        },
        "modality": {"reg_covar": REG, "eligibility": "free parameters <= n_train * d / 2", "by_pcs": summaries},
        "selection_train_bic": sel, "wall_seconds": round(time.time() - t0, 1),
    }
    (a.out / "results.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "between_classes"}, indent=1, default=str))


if __name__ == "__main__":
    main()
