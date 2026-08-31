"""Tests for owl.evaluations module."""
from __future__ import annotations

import pytest
from pandas import DataFrame


class TestEvaluation:
    def test_init_default(self) -> None:
        from owl.evaluations import Evaluation

        e = Evaluation()
        assert e.cycle == ""
        assert e.data is None

    def test_init_with_dict(self) -> None:
        from owl.evaluations import Evaluation

        e = Evaluation({"key": "value"})
        # config may be Instruct object or dict depending on kahndor availability
        assert e.config is not None

    def test_initAnalyzer(self) -> None:
        from owl.evaluations import Evaluation

        e = Evaluation()
        df = DataFrame({"a": [1, 2, 3]})
        result = e.initAnalyzer(df)
        assert result is e
        assert e.data is not None

    def test_accuracy_details(self) -> None:
        from owl.evaluations import Evaluation

        e = Evaluation()
        # Set determinent manually since it's normally computed by accuracy_map
        e.determinent = [1, 0, 1, 0, 1]
        result = e.accuracy_details()
        assert result is e
        assert e.num_correct == 3
        assert e.num_total == 5
        assert e.corrperc == 60.0

    def test_test_returns_dict(self) -> None:
        from owl.evaluations import Evaluation

        e = Evaluation()
        result = e.test([0, 1])
        assert isinstance(result, dict)
        assert "AUC" in result
        assert "LogLoss" in result


class TestFeatureAnalysis:
    def test_init(self) -> None:
        from owl.evaluations import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 1.5, 2.5, 3.5]})
        fa = FeatureAnalysis(df, n=2)
        assert fa.output_features == 2

    def test_fit(self) -> None:
        from owl.evaluations import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 1.5, 2.5, 3.5]})
        fa = FeatureAnalysis(df, n=2)
        fa.fit()
        assert fa.results is not None

    def test_analyze(self) -> None:
        from owl.evaluations import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 1.5, 2.5, 3.5]})
        fa = FeatureAnalysis(df, n=2)
        fa.fit()  # must call fit() before analyze()
        result = fa.analyze()
        assert isinstance(result, DataFrame)

    def test_pca_stub(self) -> None:
        from owl.evaluations import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0]})
        fa = FeatureAnalysis(df, n=2)
        fa.PCA()  # should not raise
