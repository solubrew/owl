# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
	docid:
	name:
	description: >
	version: 0.0.0.0.0.0
	authority: filesystem
	security: seclvl2
	<(WT)>: -32
"""
# -*- coding: utf-8 -*
# ======================================Standard Library Modules======================================================||
from __future__ import annotations

import logging
from os.path import dirname, join
import secrets
from typing import Any

# ======================================Solutions Brewer Library Modules==============================================||
try:
    from kahndor import Instruct as _kahndor_class
except Exception:
    _kahndor_class = None

if _kahndor_class:
    kahndor = _kahndor_class
else:
    kahndor = None

try:
    from kahndor.logma import Logma
except Exception:
    Logma = None

# ====================================================================================================================||
logger = logging.getLogger(__name__)
here = join(dirname(__file__), '')
pxcfg = join(here, '_data_', '.yaml')

if Logma:
    logma = Logma(__name__)
else:
    logma = logger


class RandomAlgo:
    """Zero-stub random prediction algorithm."""

    def __init__(self) -> None:
        if Logma:
            logma.info("RandomAlgo initialized")
        self.config = None

    def fit(self, ftrs: Any, targs: Any) -> 'RandomAlgo':
        return self

    def predict(self, targets: list, predict_type: str = 'category') -> list:
        if predict_type == 'category':
            vals = list(set(targets))
            return [vals[secrets.randbelow(len(vals))] for _ in targets]
        return [secrets.randbelow(max(targets) - min(targets)) + min(targets) for _ in targets]


class ZeroClassAlgo:
    """Zero-stub majority-class prediction algorithm."""

    def __init__(self) -> None:
        if Logma:
            logma.info("ZeroClassAlgo initialized")
        self.config = None

    def fit(self, ftrs: Any, targs: Any) -> 'ZeroClassAlgo':
        return self

    def predict(self, targets: list) -> list:
        majority = max(set(targets), key=list(targets).count)
        return [majority for _ in targets]


class ZeroRegressionAlgo:
    """Zero-stub mean-regression prediction algorithm."""

    def __init__(self) -> None:
        if Logma:
            logma.info("ZeroRegressionAlgo initialized")
        self.config = None

    def fit(self, ftrs: Any, targs: Any) -> 'ZeroRegressionAlgo':
        return self

    def predict(self, targets: list) -> list:
        mean = sum(targets) / len(targets)
        return [mean for _ in targets]


class ZeroTimeSeriesAlgo:
    """Zero-stub time-series prediction algorithm."""

    def __init__(self) -> None:
        if Logma:
            logma.info("ZeroTimeSeriesAlgo initialized")
        self.config = None

    def fit(self, ftrs: Any, targs: Any) -> 'ZeroTimeSeriesAlgo':
        return self

    def predict(self, targets: list) -> list:
        mean = sum(targets) / len(targets)
        return [mean for _ in targets]
