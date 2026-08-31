# -*- coding: utf-8 -*-
"""
Owl + Metrologic Integration

Provides ML evaluation metrics backed by sklearn and optional Metrologic
integration for statistical measurement and experiment tracking.

Usage::

    # Core metrics (no extra install needed)
    from owl.metrics import OwlMetrics, ModelMetrics

    om = OwlMetrics()
    om.accuracy(y_true, y_pred)

    # Metrologic integration (requires: pip install owl[metrics])
    from owl.metrics import OwlEvaluation
    eval = OwlEvaluation()
    eval.add_datatable(df)

Install::

    pip install owl[metrics]   # installs metrologic as a dependency
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

# ==============================================================================
# Core metrics — always available (sklearn only)
# ==============================================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import cross_val_score

__all__ = [
    "OwlMetrics",
    "ModelMetrics",
    "OwlEvaluation",
]


class OwlMetrics:
    """ML evaluation metrics using sklearn.

    All methods are lightweight wrappers around sklearn.metrics.
    No Metrologic dependency required.
    """

    # ── Classification ────────────────────────────────────────────────────────

    def accuracy(self, y_true: Any, y_pred: Any) -> float:
        return accuracy_score(y_true, y_pred)

    def precision(
        self, y_true: Any, y_pred: Any, average: str = "binary"
    ) -> float:
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))  # type: ignore[arg-type]

    def recall(
        self, y_true: Any, y_pred: Any, average: str = "binary"
    ) -> float:
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))  # type: ignore[arg-type]

    def f1(
        self, y_true: Any, y_pred: Any, average: str = "binary"
    ) -> float:
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))  # type: ignore[arg-type]

    def confusion_matrix(self, y_true: Any, y_pred: Any) -> Any:
        return confusion_matrix(y_true, y_pred)

    def classification_report(self, y_true: Any, y_pred: Any) -> str:
        result = classification_report(y_true, y_pred, zero_division=0)  # type: ignore[arg-type]
        if isinstance(result, dict):
            return str(result)
        return result

    # ── Regression ─────────────────────────────────────────────────────────────

    def mse(self, y_true: Any, y_pred: Any) -> float:
        return cast("float", mean_squared_error(y_true, y_pred))

    def rmse(self, y_true: Any, y_pred: Any) -> float:
        return float(mean_squared_error(y_true, y_pred) ** 0.5)

    def mae(self, y_true: Any, y_pred: Any) -> float:
        return mean_absolute_error(y_true, y_pred)

    def r2(self, y_true: Any, y_pred: Any) -> float:
        return r2_score(y_true, y_pred)

    # ── Ensemble / validation ──────────────────────────────────────────────────

    def cross_validate(
        self,
        model: Any,
        X: Any,
        y: Any,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Any:
        return cross_val_score(model, X, y, cv=cv, scoring=scoring)

    def roc_auc(self, y_true: Any, y_pred_proba: Any) -> float:
        return roc_auc_score(y_true, y_pred_proba)


class ModelMetrics:
    """High-level model evaluation combining multiple OwlMetrics."""

    def __init__(self) -> None:
        self._metrics = OwlMetrics()

    def evaluate_classification(
        self,
        y_true: Any,
        y_pred: Any,
        y_pred_proba: Any | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "accuracy": self._metrics.accuracy(y_true, y_pred),
            "precision": self._metrics.precision(y_true, y_pred),
            "recall": self._metrics.recall(y_true, y_pred),
            "f1": self._metrics.f1(y_true, y_pred),
            "confusion_matrix": self._metrics.confusion_matrix(y_true, y_pred),
        }
        if y_pred_proba is not None:
            results["roc_auc"] = self._metrics.roc_auc(y_true, y_pred_proba)
        return results

    def evaluate_regression(
        self, y_true: Any, y_pred: Any
    ) -> dict[str, float]:
        return {
            "mse": self._metrics.mse(y_true, y_pred),
            "rmse": self._metrics.rmse(y_true, y_pred),
            "mae": self._metrics.mae(y_true, y_pred),
            "r2": self._metrics.r2(y_true, y_pred),
        }

    def cross_validate_model(
        self,
        model: Any,
        X: Any,
        y: Any,
        cv: int = 5,
        task: str = "classification",
    ) -> dict[str, Any]:
        scoring = "accuracy" if task == "classification" else "r2"
        scores = self._metrics.cross_validate(model, X, y, cv=cv, scoring=scoring)
        return {
            "scores": scores.tolist() if hasattr(scores, "tolist") else list(scores),
            "mean": float(scores.mean() if hasattr(scores, "mean") else sum(scores) / len(scores)),
            "std": float(
                scores.std() if hasattr(scores, "std") else
                (sum((s - (sum(scores) / len(scores))) ** 2 for s in scores) / len(scores)) ** 0.5
            ),
        }


# ==============================================================================
# Metrologic integration — only if metrologic is installed
# ==============================================================================

if TYPE_CHECKING:
    from pandas import DataFrame


class OwlEvaluation:
    """Adapter: Owl evaluation API backed by Metrologic's DataTableEvaluation.

    Requires metrologic to be installed (``pip install owl[metrics]``).

    Provides Owl's familiar Evaluation interface (accuracy_score, auc_score,
    logloss, etc.) backed by Metrologic's richer DataTableEvaluation when
    available, while remaining usable without it.
    """

    _data_table_eval: Any = None
    _metrologic: Any = None

    def __new__(cls) -> OwlEvaluation:
        """Lazy-init Metrologic only when first used."""
        if cls._data_table_eval is None:
            try:
                from metrologic.analytics.evaluation import DataTableEvaluation

                cls._data_table_eval = DataTableEvaluation
                cls._metrologic = cls._load_metrologic()
            except ImportError:
                cls._data_table_eval = None
                cls._metrologic = None
        return super().__new__(cls)

    @staticmethod
    def _load_metrologic() -> Any:
        """Load Metrologic or None if not installed."""
        try:
            from metrologic.metrologic import Metrologic

            return Metrologic()
        except ImportError:
            return None

    def add_datatable(self, df: Any) -> OwlEvaluation:
        """Add a DataFrame to the evaluation context.

        Args:
            df: A pandas DataFrame with labeled columns.

        Returns:
            self, for chaining.
        """
        if self._data_table_eval is None:
            raise ImportError(
                "metrologic is not installed. Install with: pip install owl[metrics]"
            )
        self._dt_eval = self._data_table_eval()
        self._dt_eval.add_datatable(df)
        return self

    def accuracy_score(self, y_true: Any, y_pred: Any) -> float:
        """Compute accuracy score (sklearn path, always available)."""
        return OwlMetrics().accuracy(y_true, y_pred)

    def auc_score(self, y_true: Any, y_pred: Any) -> float | None:
        """Compute ROC AUC score (sklearn path, always available)."""
        return OwlMetrics().roc_auc(y_true, y_pred)

    def logloss(self, y_true: Any, y_pred: Any) -> float | None:
        """Compute log loss (sklearn path, always available)."""
        try:
            from sklearn.metrics import log_loss

            return log_loss(y_true, y_pred)
        except Exception:
            return None

    def measure_predictions(
        self, y_true: Any, y_pred: Any
    ) -> Any:
        """Return prediction errors as a Metrologic Measurement.

        Requires metrologic installed.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            metrologic.Measurement of absolute errors, or None if metrologic
            is not available.
        """
        if self._metrologic is None:
            return None
        import numpy as np

        errors = np.abs(np.array(y_true) - np.array(y_pred))
        return self._metrologic.Measurement(errors)

    def measure_confidence_interval(
        self, scores: Any, confidence: float = 0.95
    ) -> tuple[float, float] | None:
        """Compute a confidence interval for a list of scores.

        Args:
            scores: Iterable of numeric scores.
            confidence: Confidence level (default 0.95).

        Returns:
            (lower, upper) tuple, or None if scipy is not available.
        """
        try:
            import numpy as np
            from scipy import stats

            arr = np.array(scores)
            n = len(arr)
            mu = arr.mean()
            se = stats.sem(arr)
            h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
            return float(mu - h), float(mu + h)
        except Exception:
            return None
