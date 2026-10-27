from __future__ import annotations

import numpy as np
import pytest

from l3d import geometry as geo
from l3d import sensitivity as sens

SMALL = dict(T=12, n_static=120, n_objects=2, n_per_object=40)
QUERY = dict(max_tracks=48, max_static=64)


@pytest.fixture(scope="module")
def setup():
    scene = sens.make_scene(0, **SMALL)
    q = sens.build_queries(scene, seed=1, **QUERY)
    clean = sens.compute_residuals(scene, sens.VirtualDecoder(scene), q)
    return scene, q, clean


def _response(setup, kind: str, dep: str, **kw) -> dict[str, float]:
    scene, q, clean = setup
    res = sens.compute_residuals(scene, sens.VirtualDecoder(scene, sens.Injection(kind, dep, 0.05), seed=7, **kw), q)
    return {k: (res[k] - clean[k]) / 0.05 for k in res}


def test_scene_is_valid() -> None:
    scene = sens.make_scene(3, **SMALL)
    assert np.isfinite(scene.uv_px).all()
    assert (scene.uv_px >= 0).all() and (scene.uv_px <= 255).all()
    assert scene.X_cam[..., 2].min() > 1.5
    assert scene.is_static.sum() == 120 and scene.group.max() == 2
    rot = sens.make_scene(3, camera_translates=False, **SMALL)
    assert np.abs(rot.T_c2w[:, :3, 3]).max() < 1e-12


def test_clean_decoder_gives_zero_residuals(setup) -> None:
    _, _, clean = setup
    for k, v in clean.items():
        assert abs(v) < 1e-9, k


def test_self_frame_errors_propagate_by_gt_pose() -> None:
    scene = sens.make_scene(1, **SMALL)
    for kind in ("depth_jitter", "image_plane"):
        dec = sens.VirtualDecoder(scene, sens.Injection(kind, "S"), seed=2)
        n, s, t, c = np.arange(30), 3, 5, 9
        self_out = dec.decode(n, s, t, t)
        np.testing.assert_allclose(dec.decode(n, s, t, c), geo.apply_transform(scene.T_rel[t, c], self_out), atol=1e-12)
    # Camera-frame errors leave self-frame outputs untouched.
    dec = sens.VirtualDecoder(scene, sens.Injection("gauge", "C"), seed=2)
    np.testing.assert_allclose(dec.decode(np.arange(30), 2, 4, 4), scene.X_cam[4, :30], atol=1e-12)
    assert sens.VirtualDecoder(scene, uv_follows_xyz=False).uv_hat(np.arange(5), 0, 3).shape == (5, 2)


def test_key_responses(setup) -> None:
    r = _response(setup, "image_plane", "T")
    assert r["reprojection"] > 0.5 and r["ray_center_se3_spread"] > 0.5
    assert r["source_swap"] > 0.5 and abs(r["source_swap_gt_u2"]) < 1e-6  # footnote a leak
    assert abs(r["camera_swap_se3"]) < 1e-6
    r = _response(setup, "image_plane", "T", uv_follows_xyz=False)
    assert abs(r["source_swap"]) < 1e-6
    r = _response(setup, "gauge", "C")
    assert max(abs(r[k]) for k in ("camera_swap_se3", "ray_center_se3_spread", "ray_center_se3_center",
                                   "source_swap", "reprojection", "static_rigidity_excl_tc")) < 1e-6
    assert r["static_rigidity"] > 0.01 and r["ray_center_gt_center"] > 0.1
    r = _response(setup, "scale", "C")
    assert r["camera_swap_se3"] > 0.1 and abs(r["camera_swap_sim3"]) < 1e-6
    r = _response(setup, "nonrigid_radial", "C")
    assert r["camera_swap_se3"] > 0.1 and abs(r["ray_center_gt_spread"]) < 1e-6


def test_predicted_class_footnotes() -> None:
    ctx = {"uv_follows_xyz": True, "camera_translates": True}
    assert sens.predicted_class("image_plane", "T", "source_swap", ctx) == "Yes"
    assert sens.predicted_class("image_plane", "T", "source_swap_gt_u2", ctx) == "No"
    assert sens.predicted_class("image_plane", "T", "source_swap", {**ctx, "uv_follows_xyz": False}) == "No"
    assert sens.predicted_class("depth_offset", "T", "static_rigidity", ctx) == "Yes"
    assert sens.predicted_class("depth_offset", "T", "static_rigidity", {**ctx, "camera_translates": False}) == "No"
    assert sens.predicted_class("gauge", "C", "static_rigidity", ctx) == "Yes"
    assert sens.predicted_class("gauge", "C", "static_rigidity_excl_tc", ctx) == "No"
    assert sens.predicted_class("scale", "C", "camera_swap_se3", ctx) == "Yes"
    assert sens.predicted_class("scale", "C", "camera_swap_sim3", ctx) == "No"
    assert sens.predicted_class("nonrigid", "C", "ray_center_se3", ctx) == "Partly"
    assert sens.predicted_class("nonrigid", "C", "ray_center_gt", ctx) is None
    assert sens.predicted_class("gauge", "T", "camera_swap_se3", ctx) is None
    assert sens.classify(0.2) == "Yes" and sens.classify(0.05) == "Partly" and sens.classify(1e-4) == "No"


def test_run_sensitivity_smoke() -> None:
    out = sens.run_sensitivity(seeds=(0,), kinds=("depth_jitter",), deps=("T",), scene_kwargs=SMALL, query_kwargs=QUERY)
    cols = out["cells"]["depth_jitter/T"]["columns"]
    assert cols["static_rigidity"]["observed"] == "Yes" and cols["camera_swap_se3"]["observed"] == "No"
    assert sens.disagreements(out) == []
