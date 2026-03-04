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
import datetime as dt, gc

# ======================================3rd Party Library Modules=====================================================||
from pandas import DataFrame

# ======================================Solutions Brewer Library Modules==============================================||
from condor import condor
from ogma.logma import Logma
from ogma.profilers.memory_profilers import memProfiler
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

def code(self, tag, algo, seq, plat, targ):
	""""""
	algon = f'{algo}_{seq}'
	plat = plat
	return f'{tag}_{algo}_{seq}_{plat}_{targ}'  # ||

#@memProfiler(stream=fp)
def predict(model, features):
	""""""
	r = DataFrame()
	try:
		r = DataFrame(data={'probability': model.predict_proba(features)[:, 0]})
		if log: logma.info(f'Model {model} prediction successful using predict_proba')
	except Exception as e:
		if log: logma.info(f'Model {model} prediction failed {e}')
		try:
			r = DataFrame(data={'probability': [float(i) for i in model.predict(features)]})
			if log: logma.info(f"Model {model} prediction successful using predict")
		except Exception as e:
			if log: logma.info(f'Model {model} prediction failed {e}')
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
	except:
		cfs = None
	print('Save Model', model)
	try:  # ||
		params = model.get_params()  # ||
	except:
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
def train(algo, version, platform, FTRs, TARGs, how='full', cfg=None):
	"""Train Model using the giveb algorythm

		Modulize model and fit piecewise if possible and if not fit fully

	"""
	config = condor.instruct(algos).override(cfg)
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
		model = condor.instruct(model).modulize().obj(**args)
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
		# TODO: get datatypes from TARG dataframe
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
