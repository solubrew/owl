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
from os.path import dirname, join
from typing import Any

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor.logma import Logma

# ======================================3rd Party Library Modules=====================================================||
from tsfresh import extract_features

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')

class engine:
	def __init__(self, dataseries: Any) -> None:
		"""Initialize tsfresh engine."""
		logma.info("tsfresh.engine initialized")
		self.dataseries = dataseries

	def Features(self, id: Any, sort: Any) -> 'engine':
		"""Extract features from time series data."""
		self.features = extract_features(self.dataseries, column_id=id, column_sort=sort)
		return self

# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
