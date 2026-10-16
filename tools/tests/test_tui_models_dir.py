"""The models directory the TUI reads: the checkout's, or `OMPH_MODELS_DIR` (#534).

Everything model-shaped in the TUI comes from one constant, so this is one file
with two jobs: the default must stay the checkout's `models/`, and the override
must move the picker *and* the family defaults (the AppImage has no `models/`
next to its sources, so both have to follow the variable).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from tui import profiles

QWEN = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph"
DFLASH = "Qwen3.8-27B-DFlash2-Q4_K_M.omph"
ORNITH = "Ornith-1.5-35B-A3B-Q4_K_M.omph"


@pytest.fixture(autouse=True)
def _pristine() -> None:
    """Leave the module as the rest of the suite found it: `importlib.reload`
    patches it in place, and this fixture's teardown runs before monkeypatch's
    undo, so the variable is dropped here too."""
    yield
    os.environ.pop("OMPH_MODELS_DIR", None)
    importlib.reload(profiles)


def load(models: Path | None) -> object:
    """profiles as a fresh process with (or without) the variable set."""
    if models is None:
        os.environ.pop("OMPH_MODELS_DIR", None)
    else:
        os.environ["OMPH_MODELS_DIR"] = str(models)
    return importlib.reload(profiles)


def test_the_default_is_the_checkout() -> None:
    mod = load(None)
    assert mod.REPO_MODELS == Path(mod.__file__).resolve().parents[2] / "models"
    assert mod.MODELS["qwen3.8"]["model"] == mod.REPO_MODELS / QWEN
    assert mod.MODELS["ornith"]["model"] == mod.REPO_MODELS / ORNITH


def test_the_env_moves_the_picker_and_the_defaults(tmp_path) -> None:
    models = tmp_path / "somewhere" / "models"
    models.mkdir(parents=True)
    for name in (QWEN, DFLASH, ORNITH):
        (models / name).write_bytes(b"")

    mod = load(models)
    assert mod.REPO_MODELS == models
    assert mod.MODELS["qwen3.8"]["model"] == models / QWEN
    # the picker: the two loadable models, the drafter left out (#532)
    assert [Path(p).name for p in mod.available_models()] == [ORNITH, QWEN]
    # the form's defaults: this family's files that are there, "" for the ones
    # that are not (the encoder; and Ornith has neither drafter nor encoder)
    assert mod.default_files("qwen3.8") == {"model": str(models / QWEN),
                                            "dflash": str(models / DFLASH), "mmproj": ""}
    assert mod.default_files("ornith") == {"model": str(models / ORNITH), "dflash": "", "mmproj": ""}
    # `--model qwen3.8` is that directory's file too
    assert mod.resolve_model_arg("qwen3.8") == str(models / QWEN)


def test_nothing_to_find_is_not_an_error(tmp_path) -> None:
    """A directory with no models, or one that does not exist, leaves the form
    empty for the preflight to complain about instead of crashing."""
    mod = load(tmp_path)
    assert mod.available_models() == []
    assert mod.default_files("qwen3.8") == {"model": "", "dflash": "", "mmproj": ""}

    mod = load(tmp_path / "gone")
    assert mod.REPO_MODELS == tmp_path / "gone"
    assert mod.available_models() == []
