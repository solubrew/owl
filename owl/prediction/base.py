"""Prediction engine base class for Owl.

Provides the interface for train/predict/evaluate pipelines.
Subclass this to add new model types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from kahndor.logma import Logma
from owl.metrics import OwlMetrics

logma = Logma(__name__)


class PredictionEngine(ABC):
    """Abstract base for all Owl prediction engines.

    Subclass this and implement train(), predict(), and optionally evaluate().
    The engine tracks its own metrics instance for consistent measurement.
    """

    def __init__(self, name: str = "engine") -> None:
        self.name = name
        self.metrics = OwlMetrics()
        self._trained = False
        logma.info("PredictionEngine(%s) initialized", name)

    @abstractmethod
    def train(self, X, y, **kwargs) -> None:
        """Train the model on X (features) and y (labels)."""

    @abstractmethod
    def predict(self, X, **kwargs) -> np.ndarray:
        """Return predictions for X. Must work after train() is called."""

    def evaluate(self, y_true, y_pred, **kwargs) -> dict[str, float]:
        """Compute metrics for y_true vs y_pred using OwlMetrics."""
        auc = float(self.metrics.roc_auc(y_true, y_pred))
        mse = float(self.metrics.mse(y_true, y_pred))
        mae = float(self.metrics.mae(y_true, y_pred))
        return {"auc": round(auc, 6), "mse": round(mse, 6), "mae": round(mae, 6)}
