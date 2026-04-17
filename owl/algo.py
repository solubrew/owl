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
# -*- coding: utf-8 -*
# ======================================Standard Library Modules======================================================||
from os.path import abspath, dirname, join

# ======================================3rd Party Library Modules=====================================================||

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import kahndor
from kahndor.logma import Logma

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')
class RandomAlgo():
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('RandomAlgo')

	def predict(self, targets):
		""""""
		classes = list(set(targets))
		predictions = [classes[randrange(len(classes))] for x in targets]
		return predictions

class ZeroClassAlgo():
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('ZeroAlgo')

	def predict(self, targets):
		""""""
		classes = list(set(targets))
		predictions = [max(classes, key=classes.count) for i in targets]
		return predictions


class ZeroRegressionAlgo():
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('ZeroRegressionAlgo')

	def predict(self, targets):
		""""""
		prediction = sum(targets) / float(len(targets))
		predictions = [prediction for i in targets]
		return predictions

class ZeroTimeSeriesAlgo():
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('ZeroTimeSeriesAlgo')

# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||