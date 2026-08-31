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
from random import randrange

# ======================================3rd Party Library Modules=====================================================||

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import kahndor
from kahndor.logma import Logma

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')
class RandomAlgo(object):
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('RandomAlgo')

	def fit(self, FTRs, TARGs):
		""""""
		return self

	def predict(self, targets, predict_type='category'):
		""""""
		if predict_type == 'category':
			predictions = [list(set(targets))[randrange(len(list(set(targets))))] for x in targets]
		else:
			predictions = [randrange(min(targets), max(targets)) for x in targets]
		return predictions

class ZeroClassAlgo(object):
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('ZeroAlgo')

	def fit(self, FTRs, TARGs):
		""""""
		return self

	def predict(self, targets):
		""""""
		return [max(list(set(targets)), key=list(set(targets)).count) for i in targets]


class ZeroRegressionAlgo(object):
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('ZeroRegressionAlgo')

	def fit(self, FTRs, TARGs):
		""""""
		return self

	def predict(self, targets):
		""""""
		return [sum(targets) / float(len(targets)) for i in targets]

class ZeroTimeSeriesAlgo(object):
	""""""
	def __init__(self):
		""""""
		self.config = kahndor.instruct(pxcfg).select('ZeroTimeSeriesAlgo')

	def fit(self, FTRs, TARGs):
		""""""
		return self

	def predict(self, targets):
		""""""
		return [sum(targets) / float(len(targets)) for i in targets]

# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||