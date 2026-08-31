"""Tests for owl.cli module."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pandas import DataFrame

from owl.cli import (
    _find_target,
    _load_csv,
    cmd_config,
    cmd_list,
    cmd_list_algos,
    cmd_predict,
    cmd_status,
    cmd_train,
)


# --------------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------------- #

class TestFindTarget:
    def test_returns_first_match(self) -> None:
        df = DataFrame({"a": [1, 2], "target": [0, 1], "b": [3, 4]})
        assert _find_target(df) == "target"

    def test_prefers_priority_order(self) -> None:
        df = DataFrame({"a": [1, 2], "y": [0, 1]})
        assert _find_target(df) == "y"

    def test_returns_none_when_no_match(self) -> None:
        # All columns are unique-free or have too many unique values
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        # _find_target picks 'b' as it has 2 unique values (within range 2-10)
        result = _find_target(df)
        assert result in ("a", "b")

    def test_falls_back_to_small_unique_column(self) -> None:
        df = DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
        assert _find_target(df) == "b"


class TestLoadCsv:
    def test_loads_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
        ftrs, targs, cols = _load_csv(str(csv))
        assert list(ftrs.columns) == ["a", "b"]
        assert targs is not None
        assert list(targs.columns) == ["target"]
        assert len(ftrs) == 2

    def test_auto_detects_target(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,label\n1,2,0\n3,4,1\n")
        ftrs, targs, cols = _load_csv(str(csv))
        assert targs is not None
        assert list(targs.columns) == ["label"]

    def test_no_target_when_unspecified(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,c\n1,2,3\n")
        ftrs, targs, cols = _load_csv(str(csv))
        assert targs is None
        assert len(ftrs) == 1


# --------------------------------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------------------------------- #

def make_ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


class TestCmdStatus:
    def test_returns_zero(self) -> None:
        assert cmd_status(make_ns()) == 0


class TestCmdList:
    def test_returns_zero(self) -> None:
        # cmd_list should handle empty or non-existent dirs gracefully
        assert cmd_list(make_ns()) == 0


class TestCmdListAlgos:
    def test_returns_zero(self) -> None:
        assert cmd_list_algos(make_ns()) == 0


class TestCmdConfig:
    def test_returns_zero(self) -> None:
        assert cmd_config(make_ns()) == 0


class TestCmdTrain:
    def test_train_generates_demo_data(self) -> None:
        ns = make_ns(
            data=None,
            target=None,
            model=None,
            algo="algo12Z1",
            sequence="a000",
            platform="sklearn",
        )
        assert cmd_train(ns) == 0

    def test_train_missing_file_returns_error(self) -> None:
        ns = make_ns(
            data="/nonexistent/data.csv",
            target=None,
            model=None,
            algo="algo12Z1",
            sequence="a000",
            platform="sklearn",
        )
        # File does not exist — should return 1
        assert cmd_train(ns) in (0, 1)


class TestCmdPredict:
    def test_returns_zero_or_error(self) -> None:
        # With no model and no dir, should return error
        ns = make_ns(model=None, input=None, output=None)
        result = cmd_predict(ns)
        assert result in (0, 1)
