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
from __future__ import annotations

from importlib import import_module
import gc
import logging as _log
from os.path import dirname, join
from typing import Any, Dict, Optional

_log.basicConfig(level=_log.INFO)
_log.getLogger('PIL').setLevel(_log.WARNING)

# ======================================3rd Party Library Modules=====================================================||
from pandas import DataFrame

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import Instruct as kahndor
from kahndor.logma import Logma
#from falcon.profilers.memory_profilers import memProfiler
#from subtrix import thing

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
log = True
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', 'functions.yaml')
algos = join(here, '_data_', 'algos.yaml')
nsmbls = join(here, '_data_', 'nsmbls.yaml')
actvs = join(here, '_data_', 'activations.yaml')

def _instantiate_from_path(model_path: str, args: dict) -> object:
	"""Instantiate a class from a dotted module path.
	
	Args:
		model_path: Dotted path like 'sklearn.linear_model.LinearRegression'
		args: Keyword arguments to pass to the constructor
		
	Returns:
		Instantiated model object
	"""
	try:
		parts = model_path.rsplit('.', 1)
		if len(parts) == 2:
			module_path, class_name = parts
			module = import_module(module_path)
			cls = getattr(module, class_name)
		else:
			cls = import_module(model_path)
		# Filter out None values
		clean_args = {k: v for k, v in (args or {}).items() if v is not None}
		return cls(**clean_args)
	except Exception as e:
		logma.info(f"Failed to instantiate {model_path}: {e}")
		# Fallback: try direct import
		module = import_module(model_path)
		return module


def code(self, tag, algo, seq, plat, targ):
	""" """
	algon = f'{algo}_{seq}'
	plat = plat
	return f'{tag}_{algo}_{seq}_{plat}_{targ}'  # ||

#@memProfiler(stream=fp)
def predict(model: Any, features: Any) -> Any:
	"""Make predictions using a trained model.
	
	Args:
		model: Trained model with predict() or predict_proba() method
		features: Feature data
		
	Returns:
		DataFrame with 'probability' column
	"""
	r = DataFrame()
	try:
		r = DataFrame(data={'probability': model.predict_proba(features)[:, 0]})
		if log: logma.info(f'Model {model} prediction successful using predict_proba')
	except Exception:
		if log: logma.info(f'Model {model} prediction failed')
		try:
			r = DataFrame(data={'probability': [float(i) for i in model.predict(features)]})
			if log: logma.info(f"Model {model} prediction successful using predict")
		except Exception:
			if log: logma.info(f'Model {model} prediction failed')
	return r

def save(writer, res, name, code, model, scores={}, start=None):  # ||
	"""Save Predictions and Model Object along with a id of the DataSet"""  # ||
	predictCOLs = ['targetid', 'target', 'predicts', 'foldcycl', 'algoname',  # ||
	               'platform', 'runtype', 'targetname']  # ||

	# change code to standard table
	table = 'predictions'  # ||
	# add column for fold cycl, algo_name, platform, outputtype, targetname
	#
	# add memoryt usage to metrics
	#
	try:
		wrdata = {code: {'records': res, 'columns': predictCOLs}}  # ||
		writer.write(wrdata)  # ||
	except Exception as e:
		print('Prediction Write Failed', e, 'Save Code', wrdata.keys())  # ||
	del wrdata
	try:  # ||
		cfs = model.coef_  # ||
	except Exception:
		cfs = None
	logger.debug('Save Model: %s', model)
	try:  # ||
		params = model.get_params()  # ||
	except Exception:
		print('Save Model Params Failed for ', model)  # ||
		params = None
	if scores == None:
		scores = {}
	scores['time'] = calcts.readable(next(thing.when().gen('object')) - start)  # ||
	data = [code, name, model.get_params(), cfs, scores]  # ||
	serialize = [encoder.engine(data).serialize('list').out]  # ||
	seriCols = ['algocode', 'algoname', 'coefs', 'args', 'scores']  # ||
	try:  # ||
		wrdata = {'models': {'records': serialize, 'columns': seriCols}}  # ||
		writer.write(wrdata)  # ||
	except Exception as e:  # ||
		print('Model Score Write Failed', e)  # ||

#	cfg = self.config.dikt[algoN][seqN]['platform'][platN]#					||
def train(algo: str, version: str, platform: str,
          FTRs: DataFrame, TARGs: DataFrame,
          how: str = 'full', cfg: Optional[Dict[str, Any]] = None) -> Any:
	"""Train Model using the giveb algorythm

		Modulize model and fit piecewise if possible and if not fit fully

	"""
	config = kahndor(algos).override(cfg)
	cfg = config.dikt[algo][version]['platform'][platform]
	if cfg['valid'] == 0:
		return None
	if not isinstance(TARGs, DataFrame):
		TARGs = DataFrame(TARGs)
	model = cfg['model']
	args = cfg['dargs']
	done = False
	logma.info(f'FTRs {FTRs.shape} TARGs {TARGs.shape}')
	if FTRs.shape[0] != TARGs.shape[0]:
		return done
	if isinstance(model, str):
		model = _instantiate_from_path(model, args)
	if done == False and how == 'partial':
		try:
			model.partialfit(FTRs, TARGs.values.ravel())
			done = True
			logma.info('Partial Fit Ravel Attempt Successful')
		except Exception as e:
			if "object has no attribute 'partialfit'" in e.__str__():
				how = 'full'
			logma.info(f'Fit Ravel Attempt Failed {e}')
	if done == False and how == 'partial':
		# T-NEW-001: infer target dtype from TARGs column to auto-select partialfit strategy
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]  # ||
			model.partialfit(FTRs, TARGfloat)
			done = True
			logma.info('Partial Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			logma.info('Fit Float Attempt Failed', e)
	if done == False and how == 'partial':
		try:
			TARGint = [int(y) for y in TARGs.values.ravel()]  # ||
			model.partialfit(FTRs, TARGint)
			done = True
			logma.info('Partial Fit Int Attempt Successful')
			del TARGint
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			logma.info(f'Partial Fit Int Attempt Failed {e}')
	if done == False and how == 'partial':
		try:
			model.partialfit(FTRs, TARGs)
			done = True
			logma.info('Partial Fit Failed')
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			logma.info(f'Partial Fit Failed {e}')
	if done == False and how == 'full':
		try:
			logma.info(f"features {FTRs.head()}")
			logma.info(f"Targets {TARGs.head()}")
			model.fit(FTRs, TARGs.values.ravel())
			done = True
			logma.info('Full Fit Ravel Attempt Successful')
		except Exception as e:
			logma.info(f'Fit Ravel Attempt Failed {e}')
	if done == False and how == 'full':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]  # ||
			model.fit(FTRs, TARGfloat)
			done = True
			logma.info('Full Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			logma.info(f'Fit Float Attempt Failed {e}')
	if done == False and how == 'full':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]  # ||
			FTRsfloat = []
			for row in FTRs:
				for x in row:
					FTRsfloat.append(float(x))
			model.fit(FTRsfloat, TARGfloat)
			done = True
			logma.info('Full Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			logma.info(f'Fit Float Attempt Failed {e}')
	if done == False and how == 'full':
		try:
			TARGint = [int(y) for y in TARGs.values.ravel()]  # ||
			model.fit(FTRs, TARGint)
			done = True
			logma.info('Full Fit Int Attempt Successful')
			del TARGint
		except Exception as e:
			logma.info(f'Full Fit Int Attempt Failed {e}')
	if done == False and how == 'full':
		try:
			model.fit(FTRs, TARGs)
			done = True
			logma.info('Full Fit Successful')
		except Exception as e:
			logma.info(f'Full Fit Failed {e}')
	del FTRs
	del TARGs
	gc.collect()
	if done == False:
		return done
	return model

	logma.info(f'Trained Model {model}')
	return model

# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
