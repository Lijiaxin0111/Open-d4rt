"""A3 detection metrics: does a residual (or the native confidence) find large errors?"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr


def _clean(score: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    ok = np.isfinite(score) & np.isfinite(error)
    return score[ok], error[ok]


def spearman(score: np.ndarray, error: np.ndarray) -> float:
    """Spearman rho between a score (higher = more suspicious) and the error."""
    s, e = _clean(score, error)
    if s.size < 3:
        return float("nan")
    return float(spearmanr(s, e).statistic)


def auroc(score: np.ndarray, error: np.ndarray, top_frac: float = 0.2) -> float:
    """AUROC of ``score`` for positives = errors at or above the ``1 - top_frac`` quantile.

    Mann-Whitney rank formula with average ranks, so tied scores count one half.
    """
    s, e = _clean(score, error)
    if s.size < 2:
        return float("nan")
    pos = e >= np.quantile(e, 1.0 - top_frac)
    n_pos = int(pos.sum())
    n_neg = int(s.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_coverage(score: np.ndarray, error: np.ndarray) -> dict[str, np.ndarray | float]:
    """Risk-coverage curve: keep queries in ascending ``score`` order.

    ``risk[k]`` = mean error of the ``k + 1`` lowest-score queries, ``coverage[k] = (k+1)/n``.
    ``aurc`` = mean risk over coverages; ``aurc_oracle`` sorts by the error itself and
    ``eaurc = aurc - aurc_oracle``. Ties in ``score`` are kept in a stable order.
    """
    s, e = _clean(score, error)
    n = s.size
    if n == 0:
        nan = float("nan")
        return {"coverage": np.zeros(0), "risk": np.zeros(0), "aurc": nan, "aurc_oracle": nan, "eaurc": nan}
    order = np.argsort(s, kind="stable")
    k = np.arange(1, n + 1, dtype=np.float64)
    risk = np.cumsum(e[order]) / k
    oracle = np.cumsum(np.sort(e)) / k
    aurc = float(risk.mean())
    aurc_o = float(oracle.mean())
    return {"coverage": k / n, "risk": risk, "aurc": aurc, "aurc_oracle": aurc_o, "eaurc": aurc - aurc_o}


def detection_metrics(score: np.ndarray, error: np.ndarray, top_frac: float = 0.2) -> dict[str, float]:
    """Spearman rho, AUROC (top ``top_frac`` errors) and AURC / E-AURC of one score."""
    rc = risk_coverage(score, error)
    return {
        "spearman": spearman(score, error),
        "auroc": auroc(score, error, top_frac),
        "aurc": float(rc["aurc"]),
        "eaurc": float(rc["eaurc"]),
        "n": int(np.isfinite(np.asarray(score, float)).sum()),
    }


def confidence_score(conf_logit: np.ndarray) -> np.ndarray:
    """Suspicion score from the native confidence logit: ``-sigmoid(conf)``."""
    return -1.0 / (1.0 + np.exp(-np.asarray(conf_logit, dtype=np.float64)))


def compare_with_confidence(
    residual: np.ndarray,
    error: np.ndarray,
    conf_logit: np.ndarray,
    top_frac: float = 0.2,
) -> dict[str, Any]:
    """Detection metrics of a check residual next to the native confidence, on the same rows."""
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    conf = confidence_score(conf_logit).reshape(-1)
    ok = np.isfinite(residual) & np.isfinite(error) & np.isfinite(conf)
    return {
        "residual": detection_metrics(residual[ok], error[ok], top_frac),
        "confidence": detection_metrics(conf[ok], error[ok], top_frac),
        "n": int(ok.sum()),
    }
