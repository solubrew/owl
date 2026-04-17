# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
	docid:
	name:
	description: >
		Generic prediction algorythm trainer and excerciser.
		The Book class provides a recipe of steps for turning raw data into
		functional models  #													||
		The WorkBook class loads previously trained models and uses them to
		predict outcomes given raw data
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
from subtrix import thing

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')


class Predictor():
	"""Analyze input data using available and/or identified methods"""
	
	def __init__(self, cfg=None):
		self.config = kahndor.instruct(pxcfg).override(join(here, '_data_', 'algos.yaml'))
		self.config.override(join(here, '_data_', 'nsmbls.yaml')).override(cfg)

	def initPredictor(self, tag, writer):
		""" """
		self.cycle = thing.what().uuid().ruuid
		self.FTRs, self.model, self.tag = None, None, tag
		self.writer = writer
		self.start = next(thing.when().gen('object'))
		#self._rtr(kind)
		self.workername = f'WRKR{tag}'
		return self

	def makePrediction(self, model, x):
		""""""
		done = False
		try:
			if log: print('Predict From this Model', model)
			r = model.predict_proba(x)
			r = DataFrame(data={'probability': r[:, 0]})
			done = True
		except Exception as e0:
			if log: print('Model', model, 'Prediction Failed', e0)
			try:
				r = model.predict(x)
				r = DataFrame(data={'probability': [float(i) for i in r]})
				done = True
			except Exception as e1:
				if log: print('Model', model, 'Prediction Failed', e1)
		if done == False:
			return done
		return r


class Trainer():
	""""""
	def __init__(self):
		""""""

	def train(self, algoN, seqN, platN, ftrs, targs, ids, what=None, how='full'):  # ||
		"""Train Model using the give algorythm"""
		print('CONFIG', self.config.dikt)
		cfg = self.config.dikt[algoN][seqN]['platform'][platN]
		TAG = thing.what().uuid().ruuid
		targTAG, trainTAG = TAG[-5:], TAG[:5]
		#		print('CFG', cfg)
		if cfg['valid'] == 0:
			return None
		model = cfg['model']
		self.name = cfg['model']
		if self.model is not None:
			model = self.model
		self.code = self._code(self.tag, algoN, seqN, platN, targTAG)
		#		print('Train Code', self.code)
		print('Train', targs.head())
		print('FTRs', ftrs.head())
		if not isinstance(targs, DataFrame):
			targs = DataFrame(targs)
		done = False
		if done == False:
			#			try:#																||
			model = utils._runAlgo(ftrs, targs, model, cfg['dargs'], how)
		#				if model == False:
		#					return False
		#				done = True
		#			except Exception as e:#												||
		#				print('Non pooled run failed',e)
		if done == True:
			del ftrs
			del targs
			gc.collect()
		print('Trained Model', model)
		self.model = model
		return self
	
	
	# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||