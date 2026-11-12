# Coupled RSCM — authors' original code (unmodified)

- Source: https://github.com/EliasRaninen/CoupledRSCM, commit `bb09e1230b5d9b0e03dbb7a5d9850ee9e18399a3`
  (archive `codeload.github.com/EliasRaninen/CoupledRSCM/tar.gz/<commit>`, sha256
  `84d001474fdeb4b9ce3481a0d06c7701d07cc0c517cbb23caed7484813a5f079`), file `Python/multiclass_rscm.py`
  ("This version: March 2022").
- Licence: MIT (`LICENSE`, copied unchanged).
- Paper: E. Raninen and E. Ollila, "Coupled Regularized Sample Covariance Matrix Estimator for Multiple Classes",
  IEEE Trans. Signal Processing 69:5681-5692, 2021, doi 10.1109/TSP.2021.3118546.

| File | sha256 |
|---|---|
| `multiclass_rscm.py` | `a67e443d18613b2671f6997059cf900a3a51853d5584f7251e6a290b780aa326` |
| `LICENSE` | `aa6f84cb42f2db3bab99e2dd641ea84ad2d1df8fda7e7d5885ffda9cf81f26c3` |

The files are byte-identical to the commit (checked in `tests/test_coupled_rscm.py`). They are not edited. The one
adaptation we need (the grid search in `rscmpool` reshapes to a hard-coded `4` classes) is applied at load time by
`src/methods/coupled_rscm.py` and is described there; the unmodified module is also loaded and compared with the
adapted one on 4-class inputs.
