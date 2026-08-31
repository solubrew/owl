# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
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
# -*- coding: utf-8 -*-
# ======================================Standard Library Modules======================================================||
from __future__ import annotations

import logging
from os.path import dirname, join
from random import randrange

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


class RandomAlgo:
    """Zero-stub random prediction algorithm."""

    def __init__(self) -> None:
        self.config = None

    def fit(self, ftrs, targs):
        return self

    def predict(self, targets, predict_type: str = 'category'):
        if predict_type == 'category':
            vals = list(set(targets))
            return [vals[randrange(len(vals))] for _ in targets]
        return [randrange(min(targets), max(targets)) for _ in targets]


class ZeroClassAlgo:
    """Zero-stub majority-class prediction algorithm."""

    def __init__(self) -> None:
        self.config = None

    def fit(self, ftrs, targs):
        return self

    def predict(self, targets):
        majority = max(set(targets), key=list(targets).count)
        return [majority for _ in targets]


class ZeroRegressionAlgo:
    """Zero-stub mean-regression prediction algorithm."""

    def __init__(self) -> None:
        self.config = None

    def fit(self, ftrs, targs):
        return self

    def predict(self, targets):
        mean = sum(targets) / len(targets)
        return [mean for _ in targets]


class ZeroTimeSeriesAlgo:
    """Zero-stub time-series prediction algorithm."""

    def __init__(self) -> None:
        self.config = None

    def fit(self, ftrs, targs):
        return self

    def predict(self, targets):
        mean = sum(targets) / len(targets)
        return [mean for _ in targets]
