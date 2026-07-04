"""Tests for pipeline-level robustness: graceful skip on optional module errors."""

import os
import pytest

from modules.module_base import Module
from pipeline.pipeline import Pipeline


class _AlwaysFailing(Module):
    def __init__(self, key):
        super().__init__(key)

    def get_preconditions(self):
        return ["input"]

    def process(self, data, config):
        raise RuntimeError("boom")


class _AlwaysWorking(Module):
    def __init__(self, key, output="ok"):
        super().__init__(key)
        self._output = output

    def get_preconditions(self):
        return ["input"]

    def process(self, data, config):
        return self._output


@pytest.fixture(autouse=True)
def _bypass_env_check(monkeypatch):
    """The default _setup_environment requires TATR/denoise model files that
    are not present in the test environment. Stub it out."""
    monkeypatch.setattr(Pipeline, "_setup_environment", lambda self: True)


def test_non_critical_failure_does_not_abort_pipeline():
    pipeline = Pipeline()
    pipeline.add_stage(_AlwaysWorking("input", output="seed"))
    pipeline.add_stage(_AlwaysFailing("htr-vt-recognizer"))  # not critical
    pipeline.add_stage(_AlwaysWorking("trocr", output="rescued"))

    result = pipeline.run(input_data="x")
    assert result == "rescued"


def test_critical_failure_propagates():
    pipeline = Pipeline()
    pipeline.add_stage(_AlwaysFailing("pdf-converter"))  # critical
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run(input_data="x")


def test_returns_last_successful_output():
    pipeline = Pipeline()
    pipeline.add_stage(_AlwaysWorking("input", output="seed"))
    pipeline.add_stage(_AlwaysWorking("htr-vt-recognizer", output="species"))
    pipeline.add_stage(_AlwaysFailing("trocr"))

    result = pipeline.run(input_data="x")
    # trocr failed -> fall back to the latest successful stage.
    assert result == "species"


def test_missing_precondition_skips_non_critical():
    """A non-critical module whose preconditions are missing should be skipped, not raise."""
    pipeline = Pipeline()

    class _NeedsMissing(Module):
        def __init__(self):
            super().__init__("htr-vt-recognizer")

        def get_preconditions(self):
            return ["never-produced"]

        def process(self, data, config):
            return "should not run"

    pipeline.add_stage(_AlwaysWorking("input", output="seed"))
    pipeline.add_stage(_NeedsMissing())

    # Should not raise; htr-vt-recognizer is non-critical.
    result = pipeline.run(input_data="x")
    assert result == "seed"
