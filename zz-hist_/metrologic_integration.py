# -*- coding: utf-8 -*-
"""
Owl + Metrologic Integration

Provides statistical calculations and measurements using Metrologic.
"""

import sys
import os

from metrologic.metrologic import Metrologic
from kahndor.logma import Logma

logma = Logma(__name__)

__all__ = [
    'Metrologic',
    'OwlMetrics',
    'ModelMetrics',
    'ExperimentTracker',
]

class OwlMetrics:
    """
    Statistical metrics calculation using Metrologic.
    
    Provides common ML metrics (accuracy, precision, recall, F1, etc.)
    using Metrologic's measurement system.
    """
    
    def __init__(self):
        logma.info("OwlMetrics initialized (zz-hist_)")
        self._metrologic = Metrologic()
        
    # Classification Metrics
    
    def accuracy(self, y_true, y_pred):
        """Calculate accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    
    def precision(self, y_true, y_pred, average='binary'):
        """Calculate precision score."""
        from sklearn.metrics import precision_score
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    def recall(self, y_true, y_pred, average='binary'):
        """Calculate recall score."""
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    
    def f1(self, y_true, y_pred, average='binary'):
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
    
    def classification_report(self, y_true, y_pred):
        """Generate classification report."""
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred)
    
    # Regression Metrics
    
    def mse(self, y_true, y_pred):
        """Mean squared error."""
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)
    
    def rmse(self, y_true, y_pred):
        """Root mean squared error."""
        import numpy as np
        return np.sqrt(self.mse(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        """Mean absolute error."""
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_true, y_pred)
    
    def r2(self, y_true, y_pred):
        """R² score."""
        from sklearn.metrics import r2_score
        return r2_score(y_true, y_pred)
    
    # Ensemble & Model Metrics
    
    def cross_validate(self, model, X, y, cv=5, scoring='accuracy'):
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        return cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    def roc_auc(self, y_true, y_pred_proba):
        """ROC AUC score."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred_proba)
    
    # Statistical measurements via Metrologic
    
    def measure_predictions(self, y_true, y_pred):
        """
        Use Metrologic to measure prediction quality.
        
        Returns a Measurement object with statistical analysis.
        """
        import numpy as np
        errors = np.abs(np.array(y_true) - np.array(y_pred))
        return self._metrologic.Measurement(errors)
    
    def confidence_interval(self, scores, confidence=0.95):
        """
        Calculate confidence interval for scores.
        
        Uses Metrologic's statistical functions.
        """
        import numpy as np
        from scipy import stats
        
        n = len(scores)
        mean = np.mean(scores)
        se = stats.sem(scores)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h


class ModelMetrics:
    """
    High-level model evaluation combining multiple metrics.
    """
    
    def __init__(self):
        self._metrics = OwlMetrics()
        
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate a classification model.
        
        Returns dict with all relevant metrics.
        """
        results = {
            'accuracy': self._metrics.accuracy(y_true, y_pred),
            'precision': self._metrics.precision(y_true, y_pred),
            'recall': self._metrics.recall(y_true, y_pred),
            'f1': self._metrics.f1(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            results['roc_auc'] = self._metrics.roc_auc(y_true, y_pred_proba)
        
        # Add confusion matrix
        results['confusion_matrix'] = self._metrics.confusion_matrix(y_true, y_pred)
        
        return results
    
    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate a regression model.
        
        Returns dict with all relevant metrics.
        """
        return {
            'mse': self._metrics.mse(y_true, y_pred),
            'rmse': self._metrics.rmse(y_true, y_pred),
            'mae': self._metrics.mae(y_true, y_pred),
            'r2': self._metrics.r2(y_true, y_pred),
        }
    
    def cross_validate_model(self, model, X, y, cv=5, task='classification'):
        """
        Perform cross-validation and return scores.
        """
        scoring = 'accuracy' if task == 'classification' else 'r2'
        scores = self._metrics.cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std(),
            'ci_95': self._metrics.confidence_interval(scores),
        }


class ExperimentTracker:
    """
    Track ML experiments using Metrologic's analytics.
    
    Integrates with Metrologic for logging and analyzing
    experiments, hyperparameters, and results.
    """
    
    def __init__(self, experiment_name='owl_experiment'):
        self.experiment_name = experiment_name
        self._metrologic = Metrologic()
        self._runs = []
        
    def log_run(self, run_id, params, metrics, model=None):
        """
        Log a single experiment run.
        
        Args:
            run_id: Unique identifier for this run
            params: Hyperparameters dict
            metrics: Evaluation metrics dict
            model: Optional trained model object
        """
        run = {
            'run_id': run_id,
            'params': params,
            'metrics': metrics,
            'model': model,
        }
        self._runs.append(run)
        
    def get_best_run(self, metric='accuracy', higher_is_better=True):
        """
        Get the best run based on a metric.
        """
        if not self._runs:
            return None
        
        key = lambda r: r['metrics'].get(metric, 0)
        reverse = higher_is_better
        
        return max(self._runs, key=key) if reverse else min(self._runs, key=key)
    
    def compare_runs(self, runs=None):
        """
        Compare multiple runs using Metrologic analytics.
        """
        runs = runs or self._runs
        if not runs:
            return {}
        
        # Extract metrics for comparison
        metrics_names = set()
        for run in runs:
            metrics_names.update(run['metrics'].keys())
        
        comparison = {}
        for metric in metrics_names:
            values = [run['metrics'].get(metric, None) for run in runs if run['metrics'].get(metric) is not None]
            if values:
                measurement = self._metrologic.Measurement(values)
                comparison[metric] = {
                    'mean': measurement.mean,
                    'std': measurement.std,
                    'min': measurement.min,
                    'max': measurement.max,
                }
        
        return comparison
