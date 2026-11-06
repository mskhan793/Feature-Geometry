# 03 — Gaussian synthetic feature replay

**Status:** Starter implementation of reusable Gaussian sampling and tests; empirical study not yet run.

**Research question:** When sampling numerical embeddings from a Gaussian fitted to frozen features, how accurately do the empirical mean and covariance match the generating distribution?

**First experiment:** 2D Gaussian with known mean/covariance, repeated n = 10, 100, 1,000, 10,000; compute mean error and covariance Frobenius error, vary seed.

**Next:** Evaluate effect of covariance estimation from limited *real* features, then train a lightweight classifier on synthetic features.

**Primary reference:** Cao et al. (ICML 2026), §4.1 and Appendix B.

**Caution:** Matching two moments does not prove synthetic samples reproduce multimodal real embeddings.
