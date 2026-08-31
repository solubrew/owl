# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
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
import secrets
from os.path import dirname, join

# ======================================3rd Party Library Modules======================================================||
# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import kahndor
from kahndor.logma import Logma

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '../_data_', '.yaml')


class RandomAlgo:
    """Random prediction algorithm."""

    def __init__(self):
        logma.info("RandomAlgo initialized")
        self.config = kahndor.instruct(pxcfg).select('RandomAlgo')

    def predict(self, targets) -> list:
        """Predict randomly from available classes."""
        classes = list(set(targets))
        predictions = [classes[secrets.randbelow(len(classes))] for x in targets]
        return predictions


class ZeroClassAlgo:
    """Majority-class prediction algorithm."""

    def __init__(self):
        logma.info("ZeroClassAlgo initialized")
        self.config = kahndor.instruct(pxcfg).select('ZeroAlgo')

    def predict(self, targets) -> list:
        """Predict the majority class."""
        classes = list(set(targets))
        predictions = [max(classes, key=classes.count) for i in targets]
        return predictions


class ZeroRegressionAlgo:
    """Mean-regression prediction algorithm."""

    def __init__(self):
        logma.info("ZeroRegressionAlgo initialized")
        self.config = kahndor.instruct(pxcfg).select('ZeroRegressionAlgo')

    def predict(self, targets) -> list:
        """Predict the mean of targets."""
        prediction = sum(targets) / float(len(targets))
        predictions = [prediction for i in targets]
        return predictions


class ZeroTimeSeriesAlgo:
    """Time-series mean prediction algorithm."""

    def __init__(self):
        logma.info("ZeroTimeSeriesAlgo initialized")
        self.config = kahndor.instruct(pxcfg).select('ZeroTimeSeriesAlgo')

    def predict(self, targets) -> list:
        """Predict the mean of targets."""
        prediction = sum(targets) / float(len(targets))
        predictions = [prediction for i in targets]
        return predictions
