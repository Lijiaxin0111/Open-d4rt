"""Report helpers shared by the Part A runners (l3d.report)."""

from __future__ import annotations

import json
from pathlib import Path

from l3d.report import model_label


def test_model_label_from_manifest_name_or_override(tmp_path: Path) -> None:
    for run_name, want in (("OpenD4RT_48CLIP_9Mix_NoCropAUG_T48", "OpenD4RT 48CLIP"), ("flow3r_T48", "flow3r"),
                           ("some_new_model", "some_new_model")):
        dump = tmp_path / run_name
        dump.mkdir()
        (dump / "manifest.json").write_text(json.dumps({"run_name": run_name}))
        assert model_label(dump) == want
    assert model_label(tmp_path / "flow3r_merged") == "flow3r"  # no manifest: the directory name
    assert model_label(tmp_path / "flow3r_T48", "ARROW") == "ARROW"
