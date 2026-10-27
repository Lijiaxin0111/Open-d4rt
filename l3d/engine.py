"""Arbitrary-query decoding against one cached D4RT encoder pass.

`QueryEngine` wraps a D4RT model: `encode()` runs the video encoder once on a
clip (T <= model clip_frames) and caches the memory; `decode()` then answers any
number of queries `(u, v, t_src, t_tgt, t_cam)` (clip-local frame indices) in
chunks, reusing upstream `_run_model_for_queries`. Model loading follows
`eval_track3d_in_worldtrack.main()`.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from infer_track_3d import _resize_video, _unwrap_state_dict
from src.core import load_checkpoint, load_yaml_config
from src.eval.tasks import _encode_model_memory, _model_clip_frames, _run_model_for_queries
from src.model import build_model

_LOGGER = logging.getLogger(__name__)

OUTPUT_KEYS: tuple[str, ...] = ("xyz_3d", "uv_2d", "visibility", "displacement", "normal", "confidence")
_AMP_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(raw: str | torch.device) -> torch.device:
    """Like upstream `_resolve_device`, plus `mps`."""
    if isinstance(raw, torch.device):
        return raw
    key = str(raw).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is not available.")
        return torch.device(key)
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS but it is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def tiny_model_section(model_yaml: str | Path, clip_frames: int = 8) -> Any:
    """`model` section of `model_yaml` shrunk to a tiny random-init model for CPU tests."""
    cfg = load_yaml_config(model_yaml)
    model_cfg = cfg["model"]
    model_cfg["encoder"].update(variant="vit-b", num_layers=2, hidden_dim=64, num_heads=4)
    model_cfg["decoder"].update(num_layers=2, hidden_dim=64, num_heads=4)
    model_cfg["input"]["clip_frames"] = int(clip_frames)
    return model_cfg


class QueryEngine:
    """Encode a clip once, decode arbitrary independent queries against the cached memory."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "auto",
        image_hw: tuple[int, int] = (256, 256),
        amp: str | None = None,
        chunk_size: int = 4096,
        model_config: str = "",
        ckpt_path: str = "",
        load_info: dict[str, Any] | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.clip_frames = int(_model_clip_frames(self.model))
        self.chunk_size = max(1, int(chunk_size))
        self.model_config = str(model_config)
        self.ckpt_path = str(ckpt_path)
        self.load_info = dict(load_info or {})
        amp_key = None if amp in (None, "", "none", "fp32") else str(amp).lower()
        if amp_key is not None and amp_key not in _AMP_DTYPES:
            raise ValueError(f"Unsupported amp={amp!r}; expected None, 'bf16' or 'fp16'.")
        if amp_key is not None and self.device.type != "cuda":
            _LOGGER.warning("amp=%s requested on %s; autocast is only used on CUDA, running fp32.", amp_key, self.device)
            amp_key = None
        self.amp = amp_key
        self._video: torch.Tensor | None = None  # [1, T, 3, h, w] float in [0, 1], model resolution
        self._aspect: torch.Tensor | None = None  # [1, 1]
        self._memory: torch.Tensor | None = None  # [1, tokens, C]
        self.last_encode_seconds = float("nan")
        self.last_decode_seconds = float("nan")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls,
        model_yaml: str | Path,
        ckpt_path: str | Path,
        device: str | torch.device = "auto",
        amp: str | None = None,
        chunk_size: int = 4096,
        mmap: bool = False,
        logger: logging.Logger | None = None,
    ) -> "QueryEngine":
        """Load like upstream eval: build_model, load_checkpoint on CPU, unwrap, load_state_dict(strict=False).

        `mmap=True` memory-maps the checkpoint instead (lower peak RAM; the 14 GB
        file likely carries optimizer state that is never touched).
        """
        log = logger or _LOGGER
        cfg = load_yaml_config(model_yaml)
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        start = time.perf_counter()
        model = build_model(cfg["model"]).eval()
        if mmap:
            payload = torch.load(ckpt_path, map_location="cpu", mmap=True)
        else:
            payload = load_checkpoint(ckpt_path, map_location="cpu")
        top_keys = sorted(str(k) for k in payload.keys()) if isinstance(payload, dict) else []
        state_dict = _unwrap_state_dict(payload)
        if not state_dict:
            raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
        load_result = model.load_state_dict(state_dict, strict=False)
        num_state_tensors = len(state_dict)
        # Free the raw checkpoint (weights are copied into the model; optimizer state is dropped).
        del state_dict, payload
        gc.collect()
        load_seconds = time.perf_counter() - start

        load_info = {
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
            "missing_examples": list(load_result.missing_keys[:10]),
            "unexpected_examples": list(load_result.unexpected_keys[:10]),
            "checkpoint_top_keys": top_keys[:20],
            "num_state_tensors": int(num_state_tensors),
            "load_seconds": float(load_seconds),
        }
        log.info("Loaded checkpoint %s in %.1fs (top-level keys: %s)", ckpt_path, load_seconds, top_keys[:20])
        log.info("Missing keys: %d  Unexpected keys: %d", load_info["missing_keys"], load_info["unexpected_keys"])
        image_size = cfg.get_path("model.input.image_size", [256, 256])
        return cls(
            model,
            device=device,
            image_hw=(int(image_size[0]), int(image_size[1])),
            amp=amp,
            chunk_size=chunk_size,
            model_config=str(model_yaml),
            ckpt_path=str(ckpt_path),
            load_info=load_info,
        )

    @classmethod
    def tiny_random(
        cls,
        model_yaml: str | Path,
        clip_frames: int = 8,
        device: str | torch.device = "cpu",
        chunk_size: int = 4096,
        seed: int = 0,
    ) -> "QueryEngine":
        """Tiny random-init model (no checkpoint) for CPU smoke tests."""
        model_cfg = tiny_model_section(model_yaml, clip_frames=clip_frames)
        torch.manual_seed(int(seed))
        model = build_model(model_cfg).eval()
        image_size = model_cfg.get_path("input.image_size", [256, 256])
        return cls(
            model,
            device=device,
            image_hw=(int(image_size[0]), int(image_size[1])),
            chunk_size=chunk_size,
            model_config=f"{model_yaml}#tiny_random(clip_frames={int(clip_frames)},seed={int(seed)})",
            ckpt_path="",
            load_info={"tiny_random": True},
        )

    # ------------------------------------------------------------------ helpers
    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.amp is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=_AMP_DTYPES[self.amp])

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def num_frames(self) -> int:
        if self._video is None:
            raise RuntimeError("No clip encoded yet; call encode() or load_memory() first.")
        return int(self._video.shape[1])

    @property
    def memory(self) -> torch.Tensor:
        if self._memory is None:
            raise RuntimeError("No clip encoded yet; call encode() or load_memory() first.")
        return self._memory

    @property
    def video_model(self) -> torch.Tensor:
        if self._video is None:
            raise RuntimeError("No clip encoded yet; call encode() or load_memory() first.")
        return self._video

    def clear(self) -> None:
        self._video = None
        self._aspect = None
        self._memory = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ encode / memory io
    def encode(
        self,
        video_uint8: np.ndarray,
        aspect_wh: float | None = None,
        save_memory_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Resize `video_uint8 [T, H, W, 3]` to the model size, run the encoder once, cache the memory.

        `aspect_wh` defaults to the native `W / H` of `video_uint8`.
        """
        video_uint8 = np.asarray(video_uint8)
        if video_uint8.ndim != 4 or video_uint8.shape[-1] != 3:
            raise ValueError(f"Expected video [T, H, W, 3], got {video_uint8.shape}")
        if video_uint8.dtype != np.uint8:
            raise ValueError(f"Expected uint8 video, got {video_uint8.dtype}")
        num_frames = int(video_uint8.shape[0])
        if num_frames <= 0:
            raise ValueError("Empty video.")
        if num_frames > self.clip_frames:
            raise ValueError(
                f"Clip has T={num_frames} frames but the model supports at most clip_frames={self.clip_frames}; "
                "use a shorter clip (anchor-clip re-encoding is only used by the upstream G0 eval)."
            )
        if aspect_wh is None:
            aspect_wh = float(video_uint8.shape[2]) / float(max(video_uint8.shape[1], 1))

        video_model_rgb = _resize_video(video_uint8, image_hw=self.image_hw)
        self._set_clip(video_model_rgb, float(aspect_wh))
        start = time.perf_counter()
        with torch.no_grad(), self._autocast():
            memory = _encode_model_memory(model=self.model, video_b=self._video, aspect_b=self._aspect)
        if memory is None:
            raise RuntimeError("Model has no encode_video(); cannot cache memory.")
        self._sync()
        self.last_encode_seconds = time.perf_counter() - start
        self._memory = memory.float()
        if save_memory_path is not None:
            self.save_memory(save_memory_path)
        return self._memory

    def _set_clip(self, video_model_rgb: np.ndarray, aspect_wh: float) -> None:
        self._video = (
            torch.from_numpy(np.ascontiguousarray(video_model_rgb))
            .to(device=self.device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            / 255.0
        )
        self._aspect = torch.tensor([[float(aspect_wh)]], dtype=torch.float32, device=self.device)

    def save_memory(self, path: str | Path) -> Path:
        """Save the cached memory (fp16) plus the model-resolution clip needed for local RGB patches."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        video_u8 = (self.video_model[0].permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        torch.save(
            {
                "memory_fp16": self.memory.detach().to(dtype=torch.float16).cpu(),
                "video_model_uint8": video_u8,  # [T, h, w, 3]
                "aspect_wh": float(self._aspect.item()) if self._aspect is not None else 1.0,
                "num_frames": int(self.num_frames),
                "image_hw": list(self.image_hw),
                "model_config": self.model_config,
                "ckpt_path": self.ckpt_path,
            },
            path,
        )
        return path

    def load_memory(self, path: str | Path) -> torch.Tensor:
        """Restore a clip saved by `save_memory` (memory is upcast from fp16 to fp32)."""
        payload = torch.load(Path(path), map_location="cpu")
        video_u8 = payload["video_model_uint8"].numpy()
        if int(video_u8.shape[0]) > self.clip_frames:
            raise ValueError(f"Saved clip has T={video_u8.shape[0]} > model clip_frames={self.clip_frames}")
        self._set_clip(video_u8, float(payload.get("aspect_wh", 1.0)))
        self._memory = payload["memory_fp16"].to(device=self.device, dtype=torch.float32)
        return self._memory

    # ------------------------------------------------------------------ decode
    def decode(
        self,
        u: np.ndarray,
        v: np.ndarray,
        t_src: np.ndarray,
        t_tgt: np.ndarray,
        t_cam: np.ndarray,
        chunk_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode `M` independent queries against the cached memory.

        `u, v` are normalized pixels (`x / (W - 1)`, `y / (H - 1)`); `t_*` are
        clip-local frame indices in `[0, T)`. Returns float32 numpy arrays:
        xyz_3d [M, 3], uv_2d [M, 2], visibility [M] (logit), displacement [M, 3],
        normal [M, 3], confidence [M] (logit).
        """
        num_frames = self.num_frames
        u_arr = np.asarray(u, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(v, dtype=np.float32).reshape(-1)
        t_arrays = {
            "t_src": np.asarray(t_src).reshape(-1),
            "t_tgt": np.asarray(t_tgt).reshape(-1),
            "t_cam": np.asarray(t_cam).reshape(-1),
        }
        num_queries = int(u_arr.shape[0])
        if v_arr.shape[0] != num_queries or any(arr.shape[0] != num_queries for arr in t_arrays.values()):
            raise ValueError(
                "u, v, t_src, t_tgt, t_cam must have the same length; got "
                f"{[u_arr.shape[0], v_arr.shape[0]] + [arr.shape[0] for arr in t_arrays.values()]}"
            )
        for name, arr in t_arrays.items():
            if num_queries == 0:
                break
            if not np.issubdtype(arr.dtype, np.integer):
                if not np.all(np.equal(np.mod(arr, 1), 0)):
                    raise ValueError(f"{name} must hold integer frame indices.")
            if int(arr.min()) < 0 or int(arr.max()) >= num_frames:
                raise ValueError(
                    f"{name} out of range: min={int(arr.min())} max={int(arr.max())}, clip length T={num_frames} "
                    "(indices are clip-local, must be in [0, T))."
                )
        if num_queries > 0 and not (np.isfinite(u_arr).all() and np.isfinite(v_arr).all()):
            raise ValueError("u, v must be finite.")

        query = {
            "u": torch.from_numpy(u_arr).to(device=self.device, dtype=torch.float32),
            "v": torch.from_numpy(v_arr).to(device=self.device, dtype=torch.float32),
        }
        for name, arr in t_arrays.items():
            query[name] = torch.from_numpy(arr.astype(np.int64)).to(device=self.device, dtype=torch.long)

        start = time.perf_counter()
        with torch.no_grad(), self._autocast():
            pred = _run_model_for_queries(
                model=self.model,
                video_b=self.video_model,
                aspect_b=self._aspect,
                query=query,
                chunk_size=int(chunk_size or self.chunk_size),
                memory_b=self.memory,
            )
        self._sync()
        self.last_decode_seconds = time.perf_counter() - start
        return {key: value.float().numpy() for key, value in pred.items()}
