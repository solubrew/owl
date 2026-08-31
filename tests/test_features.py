"""Tests for owl.features module."""
from __future__ import annotations

import pytest
from pandas import DataFrame


class TestFeatureAnalysis:
    def test_init(self) -> None:
        from owl.features import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 1.5, 2.5, 3.5]})
        fa = FeatureAnalysis(df, n=2)
        assert fa.output_features == 2
        assert fa.pca is None

    def test_pca_sets_attribute(self) -> None:
        from owl.features import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 1.5, 2.5, 3.5]})
        fa = FeatureAnalysis(df, n=2)
        fa.PCA()
        assert fa.pca is not None

    def test_fit_transforms_data(self) -> None:
        from owl.features import FeatureAnalysis

        df = DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [0.5, 1.5, 2.5, 3.5, 4.5],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        fa = FeatureAnalysis(df, n=2)
        fa.PCA()
        fa.fit()
        assert fa.results is not None

    def test_analyze_returns_dataframe(self) -> None:
        from owl.features import FeatureAnalysis

        df = DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [0.5, 1.5, 2.5, 3.5],
        })
        fa = FeatureAnalysis(df, n=2)
        fa.PCA()  # must init PCA before fit()
        fa.fit()
        result = fa.analyze()
        assert isinstance(result, DataFrame)
        assert "feature_name" in result.columns
        assert "explained_variance_ratio" in result.columns

    def test_explain_without_results(self) -> None:
        from owl.features import FeatureAnalysis

        df = DataFrame({"a": [1.0, 2.0, 3.0]})
        fa = FeatureAnalysis(df, n=2)
        # No PCA() or fit() called — explain should return "no results" message
        result = fa.explain()
        assert "No analysis results" in result

    def test_explain_after_analysis(self) -> None:
        from owl.features import FeatureAnalysis

        df = DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [0.5, 1.5, 2.5, 3.5],
        })
        fa = FeatureAnalysis(df, n=2)
        fa.PCA()
        fa.fit()
        result = fa.explain()
        assert "FeatureAnalysis complete" in result
