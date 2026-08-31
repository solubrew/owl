# ======================================Standard Library Modules======================================================||
from __future__ import annotations

import logging
from os.path import dirname, join
from typing import Any

logger = logging.getLogger(__name__)

# ======================================3rd Party Library Modules======================================================||
try:
    pass
except Exception as e:
    logger.debug("pandas_profiling not available: %s", e)


try:
    from sklearn import metrics
except Exception as e:
    logger.debug("sklearn not available: %s", e)

try:
    pass
except Exception as e:
    logger.debug("tsfresh not available: %s", e)

try:
    pass
except Exception:
    logger.debug("memory_profiler not available")

# ======================================Solutions Brewer Library Modules==============================================||
try:
    from kahndor import Instruct as _kahndor_class
except Exception:
    _kahndor_class = None  # type: ignore

kahndor = _kahndor_class  # backwards compat alias

try:
    from kahndor.logma import Logma
except Exception:
    Logma = None  # type: ignore

# ====================================================================================================================||
here = join(dirname(__file__), '')
if Logma:
    logma = Logma(__name__)
else:
    logma = logger

# ====================================================================================================================||
pxcfg = join(here, '_data_', 'evaluations.yaml')


class Evaluation:
    """Analyze input data using available and/or identified methods."""
    
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        logma.info("Evaluation initialized")
        if kahndor:
            self.config = kahndor(pxcfg).override(cfg or {})
        else:
            self.config = cfg or {}
        self.cycle: str = ''
        self.data: Any = None
        self.auc: float | None = None
        self.accuracy: float | None = None
        self.dikt: dict[str, Any] = {}

    def initAnalyzer(self, data: Any) -> Evaluation:
        self.data = data
        return self

    def accuracy_score(self, probabilities: Any) -> float:
        """Calculate accuracy from probabilities."""
        if not metrics:
            return 0.0
        self.accuracy = metrics.accuracy_score(self.data, probabilities)
        return self.accuracy

    def accuracy_details(self) -> Evaluation:
        """Calculate and return details of correct prediction scores."""
        self.num_correct = 0
        for i in self.determinent:
            self.num_correct += i
        self.num_total = len(self.determinent)
        self.corrperc = self.num_correct / self.num_total * 100.0
        return self

    def accuracy_map(self, obsv: Any, knowns: Any, probabilities: Any,
                     actvtr: float) -> Evaluation:
        """Calculate Accuracy of knowns to predictions."""
        self.determinent = []
        for i in range(len(obsv)):
            if knowns[i] == 1 and obsv[i] > actvtr or knowns[i] == 0 and obsv[i] < actvtr:
                self.determinent.append(1)
            else:
                self.determinent.append(0)
        self.accuracy_score(probabilities)
        return self

    def auc_score(self, probabilities: Any) -> float | None:
        """Score the Area Under the Curve for the given Probabilities."""
        if not metrics:
            return None
        try:
            self.auc = metrics.roc_auc_score(self.data, probabilities)
        except Exception as e:
            self.auc = None
            logger.warning("AUC analysis failed: %s", e)
        return self.auc

    def logloss(self, probabilities: Any) -> float | None:
        """Measure accuracy of predictions."""
        if not metrics:
            return None
        try:
            return metrics.log_loss(self.data, probabilities)
        except Exception as e:
            logger.warning("LogLoss analysis failed: %s", e)
            return None

    def test(self, predicts: Any, how: str = 'full') -> dict[str, Any]:
        """Run full evaluation suite."""
        auc = self.auc_score(predicts)
        ll = self.logloss(predicts)
        return {
            'AUC': auc,
            'LogLoss': ll,
            'Mean': 0,
            'Median': 0,
            'StdDev': 0,
            'Length': 0,
            'Width': 0,
        }


# owl.features re-exports FeatureAnalysis (deduplicated: single implementation lives in owl/features.py)
from owl.features import FeatureAnalysis  # noqa: F401


# ====================================================================================================================||
"""
https://en.wikipedia.org/wiki/Coefficient_of_variation
"""
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||