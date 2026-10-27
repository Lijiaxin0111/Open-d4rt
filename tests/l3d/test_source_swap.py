"""Design 5 (second decode pass) on synthetic tables: schema, swap row triples, footnote a."""

from __future__ import annotations

import numpy as np
import pytest

from l3d import checks, table
from l3d import sensitivity as sens

SMALL = dict(T=12, n_static=120, n_objects=2, n_per_object=40, occlusion_prob=0.2)
MAX_TRACKS = 32


@pytest.fixture(scope="module")
def scene() -> sens.Scene:
    return sens.make_scene(0, **SMALL)


def _swap(d: dict) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows = table.source_swap_rows(d)
    P = d["p_xyz"].astype(np.float64)
    return rows, checks.source_swap(P[rows["a"]], P[rows["b"]]), checks.source_swap(P[rows["a"]], P[rows["g"]])


def test_swap_rows_cover_design5(scene: sens.Scene) -> None:
    d = sens.synthetic_table(scene, sens.VirtualDecoder(scene), max_tracks=MAX_TRACKS)
    assert table.validate_table(d) == []
    rows = table.source_swap_rows(d)
    b = np.flatnonzero(d["design"] == 5)
    assert b.size > 0
    np.testing.assert_array_equal(rows["b"], b)  # every design-5 row has both design-1 partners
    for key, src in (("a", "t1"), ("g", "t2")):
        assert (d["design"][rows[key]] == 1).all()
        np.testing.assert_array_equal(d["q_ts"][rows[key]], rows[src])
        np.testing.assert_array_equal(d["q_tt"][rows[key]], rows["t3"])
        np.testing.assert_array_equal(d["q_tc"][rows[key]], rows["c"])
        np.testing.assert_array_equal(d["q_track"][rows[key]], rows["n"])
    assert (rows["t1"] != rows["t2"]).all()
    assert (rows["t1"] != rows["t3"]).any()  # occluded targets anchor at the nearest source
    # The source pixel of a design-5 row is its parent's predicted pixel.
    np.testing.assert_allclose(d["q_u"][b], d["p_uv"][rows["parent"], 0])
    np.testing.assert_allclose(d["q_v"][b], d["p_uv"][rows["parent"], 1])


def test_footnote_a_model_u2_leaks_image_plane_error(scene: sens.Scene) -> None:
    _, model, gt = _swap(sens.synthetic_table(scene, sens.VirtualDecoder(scene), max_tracks=MAX_TRACKS))
    assert model.max() < 1e-9 and gt.max() < 1e-9

    inj = sens.Injection("image_plane", "T", 0.05)
    _, model, gt = _swap(sens.synthetic_table(scene, sens.VirtualDecoder(scene, inj, seed=7), max_tracks=MAX_TRACKS))
    assert gt.max() < 1e-9  # a T-type error is shared by both sources
    assert model.mean() > 0.01  # the model's own u2 carries its 2D error into the swap

    flat = sens.VirtualDecoder(scene, inj, seed=7, uv_follows_xyz=False)
    _, model, _ = _swap(sens.synthetic_table(scene, flat, max_tracks=MAX_TRACKS))
    assert model.max() < 1e-9  # a 2D head that returns GT pixels leaks nothing


def test_validate_rejects_bad_parent(scene: sens.Scene) -> None:
    d = sens.synthetic_table(scene, sens.VirtualDecoder(scene), max_tracks=MAX_TRACKS)
    b = np.flatnonzero(d["design"] == 5)[0]
    bad = dict(d, q_parent=d["q_parent"].copy())
    bad["q_parent"][b] = 0  # row 0 is a design-0 query
    assert any("parent" in p for p in table.validate_table(bad, strict=False))
    bad = dict(d, q_parent=d["q_parent"].copy())
    bad["q_parent"][0] = b
    assert table.validate_table(bad, strict=False) == ["'q_parent' should be -1 outside design 5"]
    with pytest.raises(ValueError, match="full table"):
        table.source_swap_rows(table.select(d, design="swap"))
