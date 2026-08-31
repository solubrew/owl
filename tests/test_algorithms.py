"""Tests for owl.algorithms (zero-stub algorithms)."""
from __future__ import annotations

import pytest
from pandas import DataFrame


class TestZeroRegressionAlgo:
    def test_fit_returns_self(self) -> None:
        from owl.algorithms.zeros import ZeroRegressionAlgo

        algo = ZeroRegressionAlgo()
        result = algo.fit(DataFrame(), DataFrame())
        assert result is algo

    def test_predict_returns_mean(self) -> None:
        from owl.algorithms.zeros import ZeroRegressionAlgo

        algo = ZeroRegressionAlgo()
        algo.fit(DataFrame(), DataFrame())
        targets = [1.0, 2.0, 3.0]
        preds = algo.predict(targets)
        assert len(preds) == len(targets)
        assert all(p == 2.0 for p in preds)


class TestZeroClassAlgo:
    def test_fit_returns_self(self) -> None:
        from owl.algorithms.zeros import ZeroClassAlgo

        algo = ZeroClassAlgo()
        result = algo.fit(DataFrame(), DataFrame())
        assert result is algo

    def test_predict_returns_majority_class(self) -> None:
        from owl.algorithms.zeros import ZeroClassAlgo

        algo = ZeroClassAlgo()
        algo.fit(DataFrame(), DataFrame())
        targets = [0, 0, 0, 1, 1]
        preds = algo.predict(targets)
        assert len(preds) == len(targets)
        assert all(p == 0 for p in preds)


class TestRandomAlgo:
    def test_fit_returns_self(self) -> None:
        from owl.algorithms.zeros import RandomAlgo

        algo = RandomAlgo()
        result = algo.fit(DataFrame(), DataFrame())
        assert result is algo

    def test_predict_category(self) -> None:
        from owl.algorithms.zeros import RandomAlgo

        algo = RandomAlgo()
        algo.fit(DataFrame(), DataFrame())
        targets = [0, 0, 1, 1, 2]
        preds = algo.predict(targets, predict_type="category")
        assert len(preds) == len(targets)
        assert all(isinstance(p, int) for p in preds)


class TestZeroTimeSeriesAlgo:
    def test_fit_returns_self(self) -> None:
        from owl.algorithms.zeros import ZeroTimeSeriesAlgo

        algo = ZeroTimeSeriesAlgo()
        result = algo.fit(DataFrame(), DataFrame())
        assert result is algo

    def test_predict_returns_mean(self) -> None:
        from owl.algorithms.zeros import ZeroTimeSeriesAlgo

        algo = ZeroTimeSeriesAlgo()
        algo.fit(DataFrame(), DataFrame())
        targets = [10.0, 20.0, 30.0]
        preds = algo.predict(targets)
        assert len(preds) == len(targets)
        assert all(p == 20.0 for p in preds)
