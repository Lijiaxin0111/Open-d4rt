from __future__ import annotations

import numpy as np
import pytest

from l3d import sensitivity as sens
from l3d import tsc


def _centered(rng: np.random.Generator, shape: tuple[int, ...], axes: tuple[int, ...], scale: float) -> np.ndarray:
    x = rng.normal(size=shape) * scale
    for ax in axes:
        x = x - x.mean(axis=ax, keepdims=True)
    return x


def test_two_way_split_recovers_planted_parts_exactly() -> None:
    rng = np.random.default_rng(0)
    n, t, S, C, d = 10, 4, 6, 6, 3
    mu = rng.normal(size=(n, t, 1, 1, d)) * 0.5
    a = _centered(rng, (n, t, S, 1, d), (2,), 1.0)
    b = _centered(rng, (n, t, 1, C, d), (3,), 0.7)
    i = _centered(rng, (n, t, S, C, d), (2, 3), 0.3)
    out = tsc.two_way_split(mu + a + b + i)
    np.testing.assert_allclose(out["T"], (mu**2).sum(-1)[..., 0, 0])
    np.testing.assert_allclose(out["S"], (a**2).sum(-1).mean(axis=(2, 3)))
    np.testing.assert_allclose(out["C"], (b**2).sum(-1).mean(axis=(2, 3)))
    np.testing.assert_allclose(out["SC"], (i**2).sum(-1).mean(axis=(2, 3)))
    planted = {
        "T": (mu**2).sum(), "S": (a**2).sum() / S, "C": (b**2).sum() / C, "SC": (i**2).sum() / (S * C),
    }
    tot = sum(planted.values())
    for k, v in planted.items():
        assert out["shares"][k] == pytest.approx(v / tot, rel=1e-10)


def test_two_way_identity_and_incomplete_cells() -> None:
    rng = np.random.default_rng(1)
    E = rng.normal(size=(7, 5, 6, 4)) + 0.4  # scalar component
    mask = np.ones(E.shape, dtype=bool)
    mask[0, 0, 2, 1] = False
    out = tsc.two_way_split(E, mask)
    parts = out["T"] + out["S"] + out["C"] + out["SC"]
    ok = np.isfinite(out["total"])
    np.testing.assert_allclose(parts[ok], out["total"][ok], rtol=1e-12)
    assert np.isnan(out["total"][0, 0]) and out["n_cells"] == 7 * 5 - 1
    assert sum(out["shares"].values()) == pytest.approx(1.0, abs=1e-12)
    # A cell with only 4 of 6 source slots used is still a complete (4 x C) grid.
    mask2 = np.ones(E.shape, dtype=bool)
    mask2[1, 1, 4:] = False
    out2 = tsc.two_way_split(E, mask2)
    ref = tsc.two_way_split(E[1:2, 1:2, :4])
    assert out2["n_cells"] == 7 * 5
    for k in ("T", "S", "C", "SC", "total"):
        assert out2[k][1, 1] == pytest.approx(ref[k][0, 0], rel=1e-12)


def test_two_way_statistical_recovery_of_variances() -> None:
    # Independent draws: expected parts T = sT^2 + ..., S ~ sS^2 (1 - 1/S) + sSC^2 (1 - 1/S) / C, etc.
    rng = np.random.default_rng(2)
    n, t, S, C = 400, 8, 6, 6
    sT, sS, sC = 1.0, 0.8, 0.6
    E = (rng.normal(size=(n, t, 1, 1)) * sT + rng.normal(size=(n, t, S, 1)) * sS
         + rng.normal(size=(n, t, 1, C)) * sC)
    out = tsc.two_way_split(E)
    assert np.nanmean(out["S"]) == pytest.approx(sS**2 * (1 - 1 / S), rel=0.05)
    assert np.nanmean(out["C"]) == pytest.approx(sC**2 * (1 - 1 / C), rel=0.05)
    assert np.nanmean(out["T"]) == pytest.approx(sT**2 + sS**2 / S + sC**2 / C, rel=0.05)
    assert np.nanmean(out["SC"]) < 1e-20


def test_one_way_split_and_gap_bins() -> None:
    rng = np.random.default_rng(3)
    E = rng.normal(size=(30, 4, 6, 2)) + 1.0
    mask = rng.random(E.shape[:3]) > 0.2
    out = tsc.one_way_split(E, mask)
    ok = np.isfinite(out["total"])
    np.testing.assert_allclose((out["T"] + out["S"])[ok], out["total"][ok], rtol=1e-12)

    E2 = rng.normal(size=(30, 4, 6, 5, 3))
    s_frames = rng.integers(0, 24, size=(30, 4, 6))
    t_frames = np.array([0, 8, 16, 23])
    res = tsc.s_share_vs_gap(E2, None, s_frames, t_frames)
    full = tsc.two_way_split(E2)
    # Bins aggregate exactly to the overall S share (balanced design).
    w_bins = np.nansum(res["share"] * np.array([
        np.nansum(((E2**2).sum(-1).mean(-1))[np.digitize(np.abs(s_frames - t_frames[None, :, None]), res["bin_edges"]) - 1 == i])
        for i in range(res["share"].size)
    ]))
    assert w_bins / ((E2**2).sum(-1).mean(-1).sum()) == pytest.approx(full["shares"]["S"], rel=1e-10)


def test_tensor_from_table_sees_planted_dependence() -> None:
    scene = sens.make_scene(2, T=12, n_static=120, n_objects=2, n_per_object=40)
    shares = {}
    for dep in ("T", "S", "C"):
        dec = sens.VirtualDecoder(scene, sens.Injection("image_plane", dep, 0.05), seed=5)
        d = sens.synthetic_table(scene, dec, max_tracks=40)
        ten = tsc.tensor_from_table(d, component="xyz")
        assert ten["E"].shape[2:4] == (6, 6)
        shares[dep] = tsc.two_way_split(ten["E"], ten["mask"])["shares"]
    assert shares["T"]["T"] > 0.95
    assert shares["S"]["S"] > 0.7
    assert shares["C"]["C"] + shares["C"]["T"] > 0.95 and shares["C"]["S"] < 1e-6
