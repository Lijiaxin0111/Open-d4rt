from __future__ import annotations

import itertools

import numpy as np
import pytest

from l3d import detection as det


def test_auroc_perfect_random_and_ties() -> None:
    rng = np.random.default_rng(0)
    err = rng.exponential(size=5000)
    assert det.auroc(err, err) == pytest.approx(1.0)
    assert det.auroc(-err, err) == pytest.approx(0.0)
    assert det.auroc(rng.normal(size=20000), rng.exponential(size=20000)) == pytest.approx(0.5, abs=0.02)
    assert det.auroc(np.ones(100), rng.normal(size=100)) == pytest.approx(0.5)


def test_auroc_matches_pair_counting_with_ties() -> None:
    rng = np.random.default_rng(1)
    err = rng.normal(size=40)
    score = rng.integers(0, 4, size=40).astype(float)  # many ties
    pos = err >= np.quantile(err, 0.8)
    wins = 0.0
    for i, j in itertools.product(np.flatnonzero(pos), np.flatnonzero(~pos)):
        wins += 1.0 if score[i] > score[j] else (0.5 if score[i] == score[j] else 0.0)
    assert det.auroc(score, err) == pytest.approx(wins / (pos.sum() * (~pos).sum()))


def test_spearman_and_risk_coverage() -> None:
    rng = np.random.default_rng(2)
    err = rng.exponential(size=1000)
    assert det.spearman(np.log(err), err) == pytest.approx(1.0)
    rc = det.risk_coverage(err, err)
    assert rc["eaurc"] == pytest.approx(0.0, abs=1e-12)
    assert rc["risk"][-1] == pytest.approx(err.mean())
    rnd = det.risk_coverage(rng.normal(size=1000), err)
    assert rnd["aurc"] > rc["aurc"]
    assert rnd["eaurc"] > 0


def test_compare_with_confidence_sign() -> None:
    rng = np.random.default_rng(3)
    err = rng.exponential(size=2000)
    conf_logit = -3.0 * err  # confident where the error is small
    out = det.compare_with_confidence(rng.normal(size=2000), err, conf_logit)
    assert out["confidence"]["auroc"] == pytest.approx(1.0)
    assert out["residual"]["auroc"] == pytest.approx(0.5, abs=0.05)
    assert out["n"] == 2000
