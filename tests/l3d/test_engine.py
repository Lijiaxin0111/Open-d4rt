"""CPU tests for l3d.engine, l3d.worldtrack_clip and scripts/l3d/dump_queries.py (tiny random model)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from eval_track3d_in_worldtrack import load_worldtrack_sequence
from infer_track_3d import _infer_tracks, _resize_video
from l3d import table as l3d_table
from l3d.engine import QueryEngine
from l3d.worldtrack_clip import load_clip

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_YAML = REPO_ROOT / "checkpoints" / "OpenD4RT_48CLIP_9Mix_NoCropAUG" / "model.yaml"
T_CLIP = 8
H_NATIVE, W_NATIVE = 72, 96
ATOL = 1e-5


def _load_dump_module():
    spec = importlib.util.spec_from_file_location("l3d_dump_queries", REPO_ROOT / "scripts" / "l3d" / "dump_queries.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def make_synthetic_worldtrack_npz(
    path: Path,
    num_frames: int = 12,
    num_tracks: int = 32,
    with_extrinsics: bool = True,
    seed: int = 0,
) -> Path:
    """WorldTrack-format npz: static + moving points; moving camera with extrinsics, or a fixed camera without."""
    rng = np.random.default_rng(seed)
    fx = fy = 0.9 * W_NATIVE
    cx, cy = (W_NATIVE - 1) / 2.0, (H_NATIVE - 1) / 2.0
    c2w = np.tile(np.eye(4), (num_frames, 1, 1))
    if with_extrinsics:
        for t in range(num_frames):
            c2w[t, :3, :3] = _rot_y(0.15 + 0.03 * t)
            c2w[t, :3, 3] = [0.4 + 0.06 * t, -0.2 - 0.02 * t, 0.3]
    w2c = np.linalg.inv(c2w)

    # Points seen by camera 0: random pixels and depths, unprojected to world.
    px = rng.uniform(4, W_NATIVE - 5, num_tracks)
    py = rng.uniform(4, H_NATIVE - 5, num_tracks)
    depth = rng.uniform(2.0, 5.0, num_tracks)
    cam0 = np.stack([(px - cx) / fx * depth, (py - cy) / fy * depth, depth], axis=-1)
    world0 = (c2w[0, :3, :3] @ cam0.T).T + c2w[0, :3, 3]
    num_dynamic = num_tracks // 3
    velocity = np.zeros((num_tracks, 3))
    velocity[:num_dynamic] = rng.normal(0.0, 0.08, (num_dynamic, 3)) + np.asarray([0.08, 0.0, 0.0])
    world = world0[None] + np.arange(num_frames)[:, None, None] * velocity[None]

    tracks_xyz = np.einsum("tij,tnj->tni", w2c[:, :3, :3], world) + w2c[:, None, :3, 3]
    u = tracks_xyz[..., 0] / tracks_xyz[..., 2] * fx + cx
    v = tracks_xyz[..., 1] / tracks_xyz[..., 2] * fy + cy
    inside = (tracks_xyz[..., 2] > 0) & (u >= 0) & (u <= W_NATIVE - 1) & (v >= 0) & (v <= H_NATIVE - 1)
    visibility = inside & (rng.random((num_frames, num_tracks)) > 0.15)
    visibility[0, : num_tracks - 2] = inside[0, : num_tracks - 2]  # most tracks visible at frame 0
    visibility[0, num_tracks - 1] = False  # one track not visible at frame 0
    tracks_xyz[min(3, num_frames - 1), num_tracks - 2] = np.nan  # one invalid GT sample

    images = np.empty((num_frames,), dtype=object)
    base = cv2.GaussianBlur(rng.integers(0, 256, (H_NATIVE, W_NATIVE, 3), dtype=np.uint8), (5, 5), 0)
    for t in range(num_frames):
        frame = np.roll(base, shift=2 * t, axis=1)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok
        images[t] = buf.tobytes()

    payload = {
        "images_jpeg_bytes": images,
        "tracks_XYZ": tracks_xyz.astype(np.float32),
        "fx_fy_cx_cy": np.asarray([fx, fy, cx, cy], dtype=np.float32),
        "visibility": visibility,
    }
    if with_extrinsics:
        payload["extrinsics_w2c"] = w2c.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("worldtrack_release")
    make_synthetic_worldtrack_npz(root / "po_mini" / "synth_pose.npz", with_extrinsics=True, seed=1)
    make_synthetic_worldtrack_npz(root / "pstudio_mini" / "synth_nopose.npz", with_extrinsics=False, seed=2)
    return root


@pytest.fixture(scope="module")
def engine() -> QueryEngine:
    return QueryEngine.tiny_random(MODEL_YAML, clip_frames=T_CLIP, device="cpu", chunk_size=512, seed=0)


def _random_queries(rng: np.random.Generator, num: int, num_frames: int) -> tuple[np.ndarray, ...]:
    return (
        rng.random(num).astype(np.float32),
        rng.random(num).astype(np.float32),
        rng.integers(0, num_frames, num),
        rng.integers(0, num_frames, num),
        rng.integers(0, num_frames, num),
    )


def test_tiny_memory_shape(engine: QueryEngine) -> None:
    video = (np.random.default_rng(0).random((T_CLIP, 256, 256, 3)) * 255).astype(np.uint8)
    memory = engine.encode(video, aspect_wh=1.0)
    assert tuple(memory.shape) == (1, 1025, 64)


def test_query_independence(engine: QueryEngine, data_root: Path) -> None:
    clip = load_clip(data_root / "po_mini" / "synth_pose.npz", clip_start=0, clip_len=T_CLIP)
    engine.encode(clip.video, aspect_wh=clip.aspect_wh)
    rng = np.random.default_rng(3)
    u, v, ts, tt, tc = _random_queries(rng, 48, clip.T)

    batched = engine.decode(u, v, ts, tt, tc, chunk_size=4096)
    single = [engine.decode(u[i : i + 1], v[i : i + 1], ts[i : i + 1], tt[i : i + 1], tc[i : i + 1]) for i in range(u.shape[0])]
    perm = rng.permutation(u.shape[0])
    shuffled = engine.decode(u[perm], v[perm], ts[perm], tt[perm], tc[perm], chunk_size=7)
    inv = np.argsort(perm)
    for key, value in batched.items():
        one_by_one = np.concatenate([item[key] for item in single], axis=0)
        np.testing.assert_allclose(one_by_one, value, atol=ATOL, rtol=0, err_msg=f"{key}: single vs batch")
        np.testing.assert_allclose(shuffled[key][inv], value, atol=ATOL, rtol=0, err_msg=f"{key}: shuffled vs batch")


def test_decode_range_errors(engine: QueryEngine) -> None:
    video = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    engine.encode(video)
    with pytest.raises(ValueError, match="out of range"):
        engine.decode([0.5], [0.5], [0], [4], [0])
    with pytest.raises(ValueError, match="clip_frames"):
        engine.encode(np.zeros((T_CLIP + 1, 32, 32, 3), dtype=np.uint8))


def test_load_clip_matches_upstream_and_rebases(data_root: Path) -> None:
    path = data_root / "po_mini" / "synth_pose.npz"
    clip = load_clip(path, clip_start=0, clip_len=T_CLIP)
    ref = load_worldtrack_sequence(path, num_frames=T_CLIP)
    np.testing.assert_array_equal(clip.video, ref["video_rgb"])
    np.testing.assert_allclose(clip.T_w2c_gt, ref["extrinsics_w2c"], atol=1e-9)
    np.testing.assert_allclose(clip.gt_world, ref["tracks_xyz_world"], atol=1e-6, equal_nan=True)
    uv_px = clip.gt_uv.astype(np.float64) * np.asarray([W_NATIVE - 1, H_NATIVE - 1])
    np.testing.assert_allclose(uv_px, ref["tracks_uv"], atol=1e-3, equal_nan=True)
    assert clip.has_pose_gt and clip.pose_gt_valid

    # Re-based clip starting at frame 3: frame 0 is the world, static points are
    # consistent across cameras, moving points are not static.
    late = load_clip(path, clip_start=3, clip_len=T_CLIP)
    np.testing.assert_allclose(late.T_w2c_gt[0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(late.gt_xyz_cam, clip_full_xyz(path)[3 : 3 + T_CLIP], atol=1e-6, equal_nan=True)
    num_tracks = late.N
    static = np.flatnonzero(late.is_static)
    assert static.size > 0 and not late.is_static[: num_tracks // 3].any()
    n = np.repeat(static, late.T * late.T)
    t = np.tile(np.repeat(np.arange(late.T), late.T), static.size)
    c = np.tile(np.tile(np.arange(late.T), late.T), static.size)
    pred = late.gt_point(n, t, c)
    expect = late.gt_xyz_cam[c, n].astype(np.float64)
    ok = np.isfinite(pred).all(-1) & np.isfinite(expect).all(-1)
    np.testing.assert_allclose(pred[ok], expect[ok], atol=1e-4)
    assert np.all(np.isnan(late.gt_point(np.asarray([-1]), np.asarray([0]), np.asarray([1]))))


def clip_full_xyz(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)["tracks_XYZ"]


def test_design0_matches_upstream_full_clip(engine: QueryEngine, data_root: Path) -> None:
    dump = _load_dump_module()
    path = data_root / "po_mini" / "synth_pose.npz"
    clip = load_clip(path, clip_start=0, clip_len=T_CLIP)
    queries = dump.build_track_design(clip)
    engine.encode(clip.video, aspect_wh=clip.aspect_wh)
    pred = engine.decode(queries["q_u"], queries["q_v"], queries["q_ts"], queries["q_tt"], queries["q_tc"])

    # Upstream query construction (eval_track3d_in_worldtrack.main) on the same sequence.
    sample = load_worldtrack_sequence(path, num_frames=T_CLIP)
    visible = np.asarray(sample["visibility"][0], dtype=bool)
    query_uv = np.asarray(sample["tracks_uv"][0, visible], dtype=np.float64)
    depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible, 2], dtype=np.float64)
    finite = np.isfinite(query_uv).all(axis=-1) & np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    track_ids = np.flatnonzero(visible)[finite]
    query_uv_norm = query_uv[finite].astype(np.float32)
    query_uv_norm[:, 0] /= float(W_NATIVE - 1)
    query_uv_norm[:, 1] /= float(H_NATIVE - 1)
    query_uv_norm = np.clip(query_uv_norm, 0.0, 1.0)
    np.testing.assert_array_equal(np.unique(queries["q_track"]), track_ids)

    ours_uv = np.stack([queries["q_u"], queries["q_v"]], -1).reshape(track_ids.size, 2 * T_CLIP - 1, 2)[:, 0]
    np.testing.assert_allclose(ours_uv, query_uv_norm, atol=1e-6)
    upstream = _infer_tracks(
        model=engine.model,
        video_model_rgb=_resize_video(sample["video_rgb"], image_hw=engine.image_hw),
        native_aspect_ratio=float(W_NATIVE) / float(H_NATIVE),
        query_uv_norm=ours_uv.astype(np.float32),
        query_chunk_size=256,
    )
    row = {(int(n), int(t), int(c)): i for i, (n, t, c) in enumerate(zip(queries["q_track"], queries["q_tt"], queries["q_tc"]))}
    local_idx = np.asarray([[row[(n, t, t)] for t in range(T_CLIP)] for n in track_ids])
    ref_idx = np.asarray([[row[(n, t, 0)] for t in range(T_CLIP)] for n in track_ids])
    np.testing.assert_allclose(pred["xyz_3d"][local_idx], upstream["tracks_xyz_local"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(pred["xyz_3d"][ref_idx], upstream["tracks_xyz_ref0"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(pred["uv_2d"][local_idx], upstream["tracks_uv_norm"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(pred["visibility"][local_idx], upstream["tracks_visibility_logits"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(pred["confidence"][local_idx], upstream["tracks_confidence"], atol=ATOL, rtol=0)


def test_memory_roundtrip(engine: QueryEngine, data_root: Path, tmp_path: Path) -> None:
    clip = load_clip(data_root / "po_mini" / "synth_pose.npz", clip_start=0, clip_len=T_CLIP)
    engine.encode(clip.video, aspect_wh=clip.aspect_wh, save_memory_path=tmp_path / "mem.pt")
    u, v, ts, tt, tc = _random_queries(np.random.default_rng(5), 32, clip.T)
    before = engine.decode(u, v, ts, tt, tc)
    engine.clear()
    engine.load_memory(tmp_path / "mem.pt")
    after = engine.decode(u, v, ts, tt, tc)
    for key in before:
        np.testing.assert_allclose(after[key], before[key], atol=2e-2, rtol=0)


@pytest.mark.parametrize("mmap", [False, True])
def test_from_checkpoint_roundtrip(engine: QueryEngine, tmp_path: Path, mmap: bool) -> None:
    import torch
    import yaml

    from l3d.engine import tiny_model_section

    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": tiny_model_section(MODEL_YAML, clip_frames=T_CLIP).to_dict()}))
    ckpt_path = tmp_path / "tiny.ckpt"
    state = engine.model.state_dict()
    torch.save({"model": state, "optimizer": {"state": {}, "param_groups": []}, "step": 7}, ckpt_path)

    loaded = QueryEngine.from_checkpoint(cfg_path, ckpt_path, device="cpu", mmap=mmap)
    assert loaded.load_info["missing_keys"] == 0 and loaded.load_info["unexpected_keys"] == 0
    video = (np.random.default_rng(1).random((T_CLIP, 40, 56, 3)) * 255).astype(np.uint8)
    u, v, ts, tt, tc = _random_queries(np.random.default_rng(2), 16, T_CLIP)
    engine.encode(video)
    loaded.encode(video)
    a = engine.decode(u, v, ts, tt, tc)
    b = loaded.decode(u, v, ts, tt, tc)
    for key in a:
        np.testing.assert_allclose(b[key], a[key], atol=ATOL, rtol=0)


def test_dump_queries_end_to_end(data_root: Path, tmp_path: Path) -> None:
    dump = _load_dump_module()
    out_root = tmp_path / "dumps"
    rc = dump.main(
        [
            "--fake-model-tiny",
            "--model-config", str(MODEL_YAML),
            "--data-root", str(data_root),
            "--subsets", "po_mini,pstudio_mini",
            "--clip-len", str(T_CLIP),
            "--designs", "0,1,2,3,4,5",
            "--out-root", str(out_root),
            "--run-name", "smoke",
            "--device", "cpu",
            "--chunk-size", "1024",
            "--save-memory",
        ]
    )
    assert rc == 0
    run_dir = out_root / "smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [c["status"] for c in manifest["clips"]] == ["ok", "ok"]
    assert all(c["swap_dropped_nonfinite"] == 0 and c["decode2_s"] > 0 for c in manifest["clips"])

    for subset, seq, has_pose in (("po_mini", "synth_pose", True), ("pstudio_mini", "synth_nopose", False)):
        path = run_dir / subset / f"{seq}__c0.npz"
        with np.load(path, allow_pickle=False) as npz:
            table = {key: npz[key] for key in npz.files}
        dump.validate_table(table)
        assert l3d_table.validate_table(l3d_table.load_table(path)) == []
        assert set(dump.REQUIRED_KEYS) <= set(table) and "memory_path" in table
        for key, dtype in dump.REQUIRED_KEYS.items():
            assert np.issubdtype(table[key].dtype, dtype), key
        assert str(table["subset"]) == subset and str(table["sequence"]) == seq
        assert int(table["T"]) == T_CLIP and int(table["clip_start"]) == 0
        assert tuple(table["hw_native"]) == (H_NATIVE, W_NATIVE)
        assert bool(table["has_pose_gt"]) == has_pose
        num_tracks = table["gt_xyz_cam"].shape[1]
        num_queries = table["design"].shape[0]
        assert table["gt_uv"].shape == (T_CLIP, num_tracks, 2) and table["is_static"].shape == (num_tracks,)
        for key, shape in (("p_xyz", (num_queries, 3)), ("p_uv", (num_queries, 2)), ("g_xyz", (num_queries, 3))):
            assert table[key].shape == shape, key
        assert set(np.unique(table["design"]).tolist()) == {0, 1, 2, 3, 4, 5}
        assert table["q_ts"].min() >= 0 and table["q_tc"].max() < T_CLIP

        # Design 0 self-frame rows: g_xyz == gt_xyz_cam at (t, n); ref rows are world (= camera 0) points.
        d0 = table["design"] == 0
        self_rows = d0 & (table["q_tc"] == table["q_tt"])
        n, t = table["q_track"][self_rows].astype(np.int64), table["q_tt"][self_rows].astype(np.int64)
        np.testing.assert_array_equal(table["g_xyz"][self_rows], table["gt_xyz_cam"][t, n])
        assert np.count_nonzero(d0) == np.unique(table["q_track"][d0]).size * (2 * T_CLIP - 1)
        # Grid rows have no GT track.
        d2 = table["design"] == 2
        assert np.all(table["q_track"][d2] == -1) and np.all(np.isnan(table["g_xyz"][d2]))
        assert np.all(table["q_ts"][d2] == table["q_tt"][d2])
        # Static rows: a static point seen from camera c is gt_xyz_cam[c, n] whatever t_t is.
        d3 = (table["design"] == 3) & np.isfinite(table["g_xyz"]).all(-1)
        assert np.any(d3)
        n3, c3 = table["q_track"][d3].astype(np.int64), table["q_tc"][d3].astype(np.int64)
        expect = table["gt_xyz_cam"][c3, n3]
        ok = np.isfinite(expect).all(-1)
        np.testing.assert_allclose(table["g_xyz"][d3][ok], expect[ok], atol=1e-4)
        # Design 1: full crossed grid, cameras include t_t and 0, sources are GT-visible.
        d1 = table["design"] == 1
        assert np.all(table["gt_vis"][table["q_ts"][d1], table["q_track"][d1]])
        groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for nn, ss, tt, cc in zip(table["q_track"][d1], table["q_ts"][d1], table["q_tt"][d1], table["q_tc"][d1]):
            groups.setdefault((int(nn), int(tt)), set()).add((int(ss), int(cc)))
        for (nn, tt), pairs in groups.items():
            sources = {p[0] for p in pairs}
            cameras = {p[1] for p in pairs}
            assert len(pairs) == len(sources) * len(cameras) and len(cameras) == 6 and {tt, 0} <= cameras
            assert len(sources) <= 6
        # Design 4: t_t == t_c, anchored at a design-1 source with GT pixel.
        d4 = table["design"] == 4
        assert np.all(table["q_tt"][d4] == table["q_tc"][d4])
        # Design 5: source pixel = clipped parent p_uv, and every row has its design-1 swap partners.
        d5 = np.flatnonzero(table["design"] == 5)
        parent = table["q_parent"][d5]
        assert np.all(table["q_parent"][table["design"] != 5] == -1)
        np.testing.assert_array_equal(table["q_u"][d5], np.clip(table["p_uv"][parent, 0], 0.0, 1.0))
        rows = l3d_table.source_swap_rows(l3d_table.load_table(path))
        np.testing.assert_array_equal(rows["b"], d5)


def test_design5_second_pass(engine: QueryEngine, data_root: Path) -> None:
    dump = _load_dump_module()
    clip = load_clip(data_root / "po_mini" / "synth_pose.npz", clip_start=0, clip_len=T_CLIP)
    engine.encode(clip.video, aspect_wh=clip.aspect_wh)
    queries, pred, info = dump.decode_designs(engine, clip, [1, 4, 5], seed=0)
    assert np.all(np.diff(queries["design"]) >= 0)  # rows in design order

    # One row per (group, source t2 != t1, camera); the anchor t1 is always a source.
    groups = dump.build_cross_groups(clip, dump._clip_rng(0, clip, 1))
    expected = sum((len(src) - 1) * len(cams) for _, _, src, cams in groups if src)
    d5 = np.flatnonzero(queries["design"] == 5)
    assert d5.size == expected > 0 and info["swap_dropped_nonfinite"] == 0
    parent = queries["q_parent"][d5]
    assert np.all(queries["design"][parent] == 4)
    assert np.all(queries["q_tt"][parent] == queries["q_ts"][d5]) and np.all(queries["q_track"][parent] == queries["q_track"][d5])
    np.testing.assert_array_equal(queries["q_u"][d5], np.clip(pred["uv_2d"][parent, 0], 0.0, 1.0))
    np.testing.assert_array_equal(queries["q_v"][d5], np.clip(pred["uv_2d"][parent, 1], 0.0, 1.0))
    assert info["swap_clipped"] == int(np.count_nonzero(((pred["uv_2d"][parent] < 0) | (pred["uv_2d"][parent] > 1)).any(-1)))

    # The second pass decodes like any other query against the same memory.
    direct = engine.decode(queries["q_u"][d5], queries["q_v"][d5], queries["q_ts"][d5], queries["q_tt"][d5], queries["q_tc"][d5])
    for key, value in direct.items():
        np.testing.assert_allclose(pred[key][d5], value, atol=ATOL, rtol=0, err_msg=key)

    # Rows whose parent prediction is non-finite are dropped and counted.
    stage1 = dump.build_queries(clip, [1, 4], seed=0, groups=groups)
    p_uv = np.full((stage1["design"].shape[0], 2), 0.5, dtype=np.float32)
    used = int(parent[0])
    p_uv[used] = np.nan
    swap, counts = dump.build_swap_design(stage1, p_uv, groups)
    assert counts["dropped_nonfinite"] == int(np.count_nonzero(parent == used)) > 0
    assert swap["design"].shape[0] == expected - counts["dropped_nonfinite"]

    with pytest.raises(ValueError, match="Design 5 needs"):
        dump.decode_designs(engine, clip, [0, 5], seed=0)
