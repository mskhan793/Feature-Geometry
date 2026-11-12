"""Class-balanced and taxonomy-informed covariance pooling — OUR EXPERIMENTAL PROPOSAL (task.md f54f534), not a method
from Ledoit & Wolf (2004) or Raninen & Ollila (2021). Every choice below was fixed before any result was seen.

Pools are made of unbiased within-class covariances S_j (each around its own class mean; never a covariance of
several classes around a shared mean, which would mix between-class centroid shifts into the pool):
- specimen-weighted: sum_j (n_j / N) S_j (the CRSCM pooled SCM); balanced: (1/J) sum_j S_j (every class once).

Donor selection for a target class c (leave-target-out: c's own specimens never enter its donor pool). Candidates =
the fold's classes with >= MIN_TRAIN train specimens (train partition only). Levels by the target's rank in the
authors' table: species -> genus, then family; genus -> family; family or unknown -> none. At a level the donors are
the other candidates with the target's name at that level (each class once, so genus mates are not double-counted
within the family). A level is accepted when the donor degrees of freedom sum_j (n_j - 1) >= d (the pool can be full
rank in d dimensions); otherwise the next level; otherwise 'global' = all other candidates.
Non-taxonomic control ('nearest'): the k candidates whose train class means are closest (Euclidean, same space) to the
target's train mean, k = the number of taxonomic donors, extended in order of distance until the same df rule holds;
defined only where a taxonomic level was accepted.

Estimator — one fixed diagnostic rule for every pool P_c (``coupled_covariance``, Eq. 4 form):
    T = beta_c S_c + (1 - beta_c) P_c,   Sigma_c = alpha T + (1 - alpha) (tr T / d) I,
with beta_c = the authors' CRSCM beta of class c (tuned for their own pool, the specimen-weighted SCM over the
eligible classes; not optimal for other pools) and alpha = 1 - alpha_LW,c (the class's Ledoit-Wolf identity
shrinkage). Pooling weight, identity shrinkage and the identity scale (tr T / d) are kept separate and reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS_FROM = {"species": ["genus", "family"], "genus": ["family"], "family": [], None: []}


def pool(scm: dict, n: dict, classes: list, weighting: str) -> np.ndarray:
    """Specimen-weighted or balanced mean of the classes' unbiased covariances."""
    if weighting == "specimen":
        total = sum(n[c] for c in classes)
        return sum(n[c] / total * scm[c] for c in classes)
    if weighting == "balanced":
        return sum(scm[c] for c in classes) / len(classes)
    raise ValueError(weighting)


def donor_df(n: dict, donors: list) -> int:
    return int(sum(n[c] - 1 for c in donors))


def taxonomy_donors(target, taxonomy: pd.DataFrame, candidates: list, n: dict, d: int) -> tuple[str, list]:
    """(level, donors): the first accepted taxonomic level for ``target``, else ('global', all other candidates)."""
    others = [c for c in candidates if c != target]
    if target in taxonomy.index:
        t = taxonomy.loc[target]
        for level in LEVELS_FROM[t["rank"]]:
            if pd.isna(t[level]):
                continue
            donors = [c for c in others if c in taxonomy.index and taxonomy.loc[c, level] == t[level]]
            if donors and donor_df(n, donors) >= d:
                return level, donors
    return "global", others


def nearest_donors(target, means: dict, candidates: list, k: int, n: dict, d: int) -> list:
    """The k nearest other candidates by train-mean distance, extended until the df rule holds."""
    others = sorted((c for c in candidates if c != target), key=lambda c: float(np.linalg.norm(means[c] - means[target])))
    chosen = others[:k]
    for c in others[k:]:
        if donor_df(n, chosen) >= d:
            break
        chosen.append(c)
    return chosen
