"""Class-macro summaries of per-class held-out log-density differences (model minus baseline).

Each class contributes its mean paired difference over val specimens and a vector of bootstrap means (resampled val
specimens, independent across classes). The macro mean is the plain mean over the included classes; its interval is
the 2.5 / 97.5 % quantiles of the mean over classes of the per-class bootstrap draws (stratified bootstrap over val
specimens, conditional on the fitted PCA and models and on the included classes; pointwise, no multiplicity control).

Reliability: a comparison of a class is *reliable* when both the model and the baseline have status ``ok`` (eligible,
converged, every component's responsibility mass >= d + 1). Primary summaries use reliable comparisons only; summaries
over every fitted comparison are kept as diagnostics. ``common_reliable`` lists the classes in which a whole set of
models is reliable, so several models can be compared on the same classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ci95(a: np.ndarray) -> list[float]:
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def macro_difference(table: pd.DataFrame, boots: dict, key: str, classes: list[str]) -> dict:
    """``table`` indexed by class with columns ``key`` and ``key_ci95``; ``boots[class][key]`` the bootstrap means."""
    if not classes:
        return {"classes": 0}
    t = table.loc[classes]
    return {"classes": len(classes), "mean": float(t[key].mean()),
            "ci95": ci95(np.mean([boots[c][key] for c in classes], axis=0)),
            "classes_better": int((t[key] > 0).sum()),
            "classes_ci_above_0": int(sum(v[0] > 0 for v in t[f"{key}_ci95"])),
            "classes_ci_below_0": int(sum(v[1] < 0 for v in t[f"{key}_ci95"]))}


def reliable(table: pd.DataFrame, classes: list[str], models: list[str]) -> list[str]:
    """Classes (in the given order) in which every model in ``models`` has status ``ok``."""
    ok = np.ones(len(classes), dtype=bool)
    for m in models:
        ok &= (table.loc[classes, f"{m}_status"] == "ok").to_numpy() if f"{m}_status" in table else False
    return [c for c, k in zip(classes, ok) if k]


def compare(table: pd.DataFrame, boots: dict, model: str, base: str, classes: list[str]) -> dict:
    """Primary (reliable model and baseline) summary of ``model - base``, the excluded classes, and the summary over
    every fitted comparison as a diagnostic."""
    key = f"{model}-{base}"
    have = [c for c in classes if key in boots[c]]
    rel = reliable(table, have, [model, base])
    return {**macro_difference(table, boots, key, rel), "classes_fitted": len(have),
            "classes_excluded_unreliable": sorted(set(have) - set(rel)),
            "all_fits_diagnostic": macro_difference(table, boots, key, have)}
