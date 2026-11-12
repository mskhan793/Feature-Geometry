"""Coupled regularized sample covariance matrix (CRSCM) estimator of Raninen & Ollila, *Coupled Regularized Sample
Covariance Matrix Estimator for Multiple Classes*, IEEE TSP 69:5681-5692, 2021 (doi 10.1109/TSP.2021.3118546), run
with the authors' original code (``third_party/coupled_rscm/multiclass_rscm.py``, commit bb09e12, file unmodified;
source note ``docs/coupled_rscm_source_note.md``).

Estimator (paper Eq. 4; code ``rscmpool``), class k with its own (alpha_k, beta_k) in [0, 1]^2:
    T_k = beta_k S_k + (1 - beta_k) S_pool,    Sigmahat_k = alpha_k T_k + (1 - alpha_k) (tr T_k / p) I,
S_k = unbiased class sample covariance (1/(n_k - 1)), S_pool = sum_k (n_k/N) S_k over the K classes passed in.
alpha_k is the weight KEPT on the data matrix T_k; 1 - alpha_k is the shrinkage toward the scaled identity (the
quantity scikit-learn's / our Ledoit-Wolf alpha denotes). (alpha_k, beta_k) minimise a plug-in estimate of the
Frobenius MSE E||Sigmahat_k - Sigma_k||_F^2 (Theorem 1): grid {0, 0.05, ..., 1}^2, then alternating closed-form
updates (7)/(8) clipped to [0, 1] (Sect. V-A). Plug-in parameters (Sect. IV; code ``estimate_parameters``), per class:
scale eta_k = tr(S_k)/p; elliptical kurtosis kappa_k = mean marginal excess kurtosis / 3 (biased moments), floored at
-2/(p + 2); sphericity gamma_k from the spatial sign covariance around the spatial median, clipped to [1, p] (as in
the paper, Sect. IV-B); <Sigma_i, Sigma_j>_F (i != j) = p^2 eta_i eta_j <SSCM_i, SSCM_j>_F. The tuning assumes
elliptical classes with finite fourth moments.

Our adaptations (applied in memory when the source is loaded; the file is never edited; each asserted to match once):
1. essential: the grid evaluation in ``rscmpool``, ``msefunction(...).reshape(4)``, hard-codes K = 4 classes and becomes
   ``.reshape(K)``;
2. output only (methods review 2026-11-12): after the alternating search, ``rscmpool`` also stores its MSE-polynomial
   coefficients and iteration count in ``params`` (keys ``_rscmpool_C``, ``_rscmpool_iterations``), so convergence can
   be audited: the authors' stopping rule adds the alpha and beta changes before the norm (they can cancel) and its
   iteration-limit warning is unreachable. ``convergence_residuals`` applies the authors' updates (7)/(8) once more to
   the returned (alpha, beta); nothing in the estimate changes.
Everything else runs as published. ``original_module`` loads the unmodified source (usable only with K = 4) so the two
can be compared.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[2] / "third_party" / "coupled_rscm" / "multiclass_rscm.py"
SOURCE_SHA256 = "a67e443d18613b2671f6997059cf900a3a51853d5584f7251e6a290b780aa326"
PATCHES = [("C).reshape(4)", "C).reshape(K)"),
           ("        print('rscmpool: slow convergence.')\n",
            "        print('rscmpool: slow convergence.')\n"
            "    params['_rscmpool_C'] = C\n    params['_rscmpool_iterations'] = iteration + 1\n")]


def _load(patched: bool) -> ModuleType:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError(f"{SOURCE} differs from the pinned authors' file")
    text = raw.decode()
    if patched:
        for find, replace in PATCHES:
            if text.count(find) != 1:
                raise RuntimeError(f"patched text not found exactly once: {find!r}")
            text = text.replace(find, replace)
    module = ModuleType("multiclass_rscm_k" if patched else "multiclass_rscm")
    module.__file__ = str(SOURCE)
    exec(compile(text, str(SOURCE), "exec"), module.__dict__)
    return module


@functools.cache
def original_module() -> ModuleType:
    return _load(patched=False)


@functools.cache
def adapted_module() -> ModuleType:
    return _load(patched=True)


@dataclass
class CoupledFit:
    classes: list
    n: np.ndarray            # train samples per class
    alpha: np.ndarray        # weight kept on T_k (1 - alpha = shrinkage toward the scaled identity)
    beta: np.ndarray         # weight on the class's own S_k inside T_k
    eta: np.ndarray
    gamma: np.ndarray
    kappa: np.ndarray
    scm: list[np.ndarray]    # authors' unbiased S_k
    pooled: np.ndarray       # authors' S_pool
    sigmas: list[np.ndarray]
    residual_alpha: np.ndarray  # |update (7) at the returned beta - alpha| (NaN for the unmodified module)
    residual_beta: np.ndarray   # |update (8) at the returned alpha - beta|
    iterations: int | None


def convergence_residuals(c: dict, alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One more application of the authors' clipped updates (7) and (8) (``opt_al``, ``opt_be`` in ``rscmpool``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.clip(-0.5 * (beta * c["11"] + c["10"]) / (beta ** 2 * c["22"] + beta * c["21"] + c["20"]), 0, 1)
        b = np.clip(-0.5 * (alpha ** 2 * c["21"] + alpha * c["11"] + c["01"]) / (alpha ** 2 * c["22"] + c["02"]), 0, 1)
    return np.abs(a - alpha), np.abs(b - beta)


def coupled_covariance(s_k: np.ndarray, s_pool: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Paper Eq. (4) for given (alpha, beta); used for the special cases and to check the authors' output."""
    t = beta * s_k + (1 - beta) * s_pool
    return alpha * t + (1 - alpha) * np.trace(t) / t.shape[0] * np.eye(t.shape[0])


def fit_coupled(x: np.ndarray, y: np.ndarray, classes: list, module: ModuleType | None = None) -> CoupledFit:
    """Run the authors' ``estimate_parameters`` + ``rscmpool`` (class-specific parameters, no averaging) on the rows
    of ``x`` whose label is in ``classes`` (classes coded 0..K-1 in the given order, as the code requires)."""
    m = module or adapted_module()
    code = {c: k for k, c in enumerate(classes)}
    keep = np.isin(y, classes)
    data = pd.DataFrame(np.asarray(x, dtype=np.float64)[keep], columns=[f"v{j}" for j in range(x.shape[1])])
    data["class"] = [code[c] for c in np.asarray(y)[keep]]
    params = m.estimate_parameters(data)
    out = m.rscmpool(params, averaging=False, compute_inverse=False)
    nan = np.full(len(classes), np.nan)
    res = convergence_residuals(params["_rscmpool_C"], out["alpha"], out["beta"]) if "_rscmpool_C" in params else (nan, nan)
    return CoupledFit(classes=list(classes), n=params["n"], alpha=out["alpha"], beta=out["beta"], eta=params["eta"],
                      gamma=params["gamma"], kappa=params["kappa"], scm=[np.asarray(s) for s in params["SCM"]],
                      pooled=np.asarray(params["Sp"]), sigmas=[np.asarray(s) for s in out["Sigmas"]],
                      residual_alpha=res[0], residual_beta=res[1], iterations=params.get("_rscmpool_iterations"))
