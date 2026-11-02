"""GAUGE baseline (l3d.gauge) on exact synthetic distortions."""

from __future__ import annotations

import numpy as np

from l3d import gauge
from l3d import geometry as geo

T, N_STATIC, N_DYN = 10, 20, 40


def _scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(N_STATIC, int), np.ones(N_DYN // 2, int), np.full(N_DYN // 2, 2)]
    base = rng.uniform([-1, -1, 2], [1, 1, 5], size=(labels.size, 3))
    drift = np.where(labels[:, None] > 0, rng.normal(size=(labels.size, 3)) * 0.02, 0.0)
    Q = base[None] + np.arange(T)[:, None, None] * drift[None]
    centers = np.cumsum(rng.normal(size=(T, 3)) * 0.02, axis=0)
    centers[0] = 0.0
    return Q, labels, centers


def test_gauge_undoes_group_ray_scales_and_static_scale() -> None:
    Q, labels, centers = _scene()
    P = Q.copy()
    for g, alpha in ((1, 1.3), (2, 0.8)):  # constant per-group ray scale about each camera center
        m = labels == g
        P[:, m] = centers[:, None] + (Q[:, m] - centers[:, None]) / alpha
    P[:, labels == 0] = Q[:, labels == 0] / 1.2  # static group: scale about the origin
    known = np.ones(labels.shape, bool)[None].repeat(T, 0)
    anchors = np.zeros(labels.size, bool)
    anchors[[0, 1, 2, N_STATIC, N_STATIC + 1, N_STATIC + N_DYN // 2]] = True
    out = gauge.gauge_correct(P, Q, known, labels, anchors, centers)
    np.testing.assert_allclose(out, Q, atol=1e-9)
    m = gauge.apd_epe(out, Q, np.ones((T, labels.size), bool))
    assert m["apd"] == 1.0 and m["epe"] < 1e-9


def test_ungrouped_and_unanchored_points_keep_the_baseline() -> None:
    Q, labels, centers = _scene(1)
    labels = labels.copy()
    labels[:3] = -1
    P = Q * 1.1
    known = np.ones((T, labels.size), bool)
    anchors = labels == 1  # only group 1 is anchored
    out = gauge.gauge_correct(P, Q, known, labels, anchors, centers)
    keep = (labels == -1) | (labels == 0) | (labels == 2)
    np.testing.assert_array_equal(out[:, keep], P[:, keep])


def test_sim3_per_group_recovers_a_group_similarity() -> None:
    Q, labels, _ = _scene(2)
    R = geo.rotvec_to_matrix(np.array([0.1, -0.2, 0.05]))
    P = Q.copy()
    m = labels == 1
    P[:, m] = geo.apply_sim3(Q[:, m], 0.7, R, np.array([0.1, 0.0, -0.2]))
    anchors = m & (np.arange(labels.size) % 3 == 0)
    out = gauge.sim3_per_group(P, Q, np.ones((T, labels.size), bool), labels, anchors)
    np.testing.assert_allclose(out[:, m], Q[:, m], atol=1e-9)


def test_allocate_anchors_budget_and_caps() -> None:
    labels = np.r_[np.zeros(50, int), np.ones(30, int), np.full(20, 2), np.full(5, -1)]
    motion = np.r_[np.zeros(50), np.full(30, 0.5), np.full(20, 1.0), np.zeros(5)]
    cand = np.ones(labels.size, bool)
    cand[80:100] = False  # group 2 has no candidates
    a = gauge.allocate_anchors(labels, motion, cand, 0.1, np.random.default_rng(0))
    assert a.sum() <= 10 and not a[80:].any() and a[labels >= 0].sum() == a.sum()
    assert a[50:80].sum() >= 3  # moving group 1 gets more than its size share
