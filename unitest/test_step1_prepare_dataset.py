"""Tests for the murat dataset preparation script."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class FakeRaw:
    """Minimal stub that mimics the MNE Raw ``save`` API."""

    def save(self, path: str | Path, overwrite: bool = False) -> None:  # noqa: FBT001, FBT002
        Path(path).write_text("fif")


def _install_fake_pyblinker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a lightweight ``pyblinker.utils.evaluation.mat_data`` stub."""

    mat_data = SimpleNamespace()

    def load_raw_from_mat(*_args, **_kwargs) -> FakeRaw:
        return FakeRaw()

    def pick_channels(raw: FakeRaw, _channels) -> FakeRaw:
        return raw

    def parse_channel_spec(spec: str) -> list[str]:
        return [part.strip() for part in spec.split(",") if part.strip()]

    mat_data.load_raw_from_mat = load_raw_from_mat
    mat_data.pick_channels = pick_channels
    mat_data.parse_channel_spec = parse_channel_spec

    evaluation_module = ModuleType("pyblinker.utils.evaluation")
    evaluation_module.mat_data = mat_data
    utils_module = ModuleType("pyblinker.utils")
    utils_module.evaluation = evaluation_module
    pyblinker_module = ModuleType("pyblinker")
    pyblinker_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "pyblinker", pyblinker_module)
    monkeypatch.setitem(sys.modules, "pyblinker.utils", utils_module)
    monkeypatch.setitem(sys.modules, "pyblinker.utils.evaluation", evaluation_module)


def test_convert_all_writes_fif_and_edf(tmp_path, monkeypatch):
    _install_fake_pyblinker(monkeypatch)
    sys.modules.pop("murat_sequence.step1_prepare_dataset", None)
    step1 = importlib.import_module("murat_sequence.step1_prepare_dataset")

    dataset_root = tmp_path / "dataset"
    rec_dir = dataset_root / "recording01"
    rec_dir.mkdir(parents=True)

    mat_path = rec_dir / "recording01.mat"
    mat_path.write_text("dummy")

    export_called: dict[str, bool] = {"value": False}

    def fake_export_raw(raw: FakeRaw, path: str | Path, fmt: str, overwrite: bool = False) -> None:  # noqa: FBT001, FBT002
        assert isinstance(raw, FakeRaw)
        assert fmt == "edf"
        Path(path).write_text("edf")
        export_called["value"] = True

    monkeypatch.setattr(step1, "export_raw", fake_export_raw)

    results = step1.convert_all(dataset_root, force=True)

    assert len(results) == 1
    result = results[0]
    assert result.fif_path.exists()
    assert result.edf_path.exists()
    assert result.fif_path.read_text() == "fif"
    assert result.edf_path.read_text() == "edf"
    assert export_called["value"]
