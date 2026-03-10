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
import gc
import datetime as dt

# ======================================3rd Party Library Modules=====================================================||
from pandas import DataFrame

# ======================================Solutions Brewer Library Modules==============================================||
from condor import condor
from ogma.logma import Logma
from owl.functions import train, predict, save
from sentinel.sentinel import Sentinel
from squirl.orgnql import sonql

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
log = True
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', 'owl.yaml')
algos = join(here, '_data_', 'algos.yaml')
nsmbls = join(here, '_data_', 'nsmbls.yaml')
actvs = join(here, '_data_', 'activations.yaml')

class Owl(Sentinel):
	""""""
	def __init__(self, tag: str, cfg=None):
		""""""
		self.config = condor.instruct(pxcfg).override(algos).override(nsmbls).override(cfg)
		name = 'Owl'
		Sentinel.__init__(self, name, self.config)
		self.code = None
		self.cycle = thing.What().uuid().ruuid
		self.FTRs = None
		self.model = None
		self.tag = tag
		self.start = dt.datetime.now()
		self.workername = f'WRKR_{self.tag}'
		self.active_model = None


	def setActiveModel(self, algoN, seqN, platN):
		""""""
		self.active_model = {'algoN': algoN, 'seqN': seqN, 'platN': platN}


	def setDataLibrary(self, data_library):
		""""""
		self.data_library = data_library
		return self

	def setModels(self, models):
		""""""
		self.model = models
		return self

	def addModel(self, algoN, seqN, platform):
		""""""
		self.models.append([algoN, seqN, platform])
		return self

	def cross_validate(self):
		""""""

		return self

	def evaluate(self, train_targs, valid_targs, runs=None, vplats=None):
		"""Run full suite of training and predictions against provided data

			need to generalize

		"""
		with sonql.doc(db, 0) as dbo:
			if runs == None:
				runs = algos
			aplats = Predictor.loadPlats()
			if vplats == None:
				vplats = aplats

			cfg = {'tables': [f'{ftag}train'], 'page': page, 'split': [2, 3, 5, 315],
			       'targBIN': targBIN}  # ||
			params = {'algoN': algoNAME, 'seqN': algoSEQ, 'platN': algoPLAT,
			          'what': 'train', 'how': 'partial'}  # ||
			columns = {'ids': 1, 'some_targs': 3, 'pmax_targs': 4, 'count_targs': 5}
			cfg = {'tables': [train_targs, ], 'page': page}
			train_targets = selector.procDataExtract(dbn, cfg, columns)
			cfg = {'tables': [valid_targs, ], 'page': page}
			tourn_targets = selector.procDataExtract(dbn, cfg, columns)
			# try going with 200k and 3 processors
			# use threading to train model on two subsets? is this possible?
			# multiprocessing is provided in the next level up and would
			# processes each algorythm against the whole data seperately on a
			# in a different and/or machine
			# with worker integrated processing opening up the path to use multiple cores
			if plat in self.algos[algo][seq]['platform'].keys():  # ||
				valid = self.algos[algo][seq]['platform'][plat]['valid']  # ||
				if valid == 0:  # ||
					return None  # ||
				# this should be encapsolated for multiprocessing
				status = None
				status = self.process(dbo, self.train, cfg)  # ||train models
				if status != None and status != False:  # ||
					status = self.process(dbo, self.predict, cfg)  # ||predict test targets
				if status == True:  # ||
					status = self.process(dbo, self.predict, cfg)  # ||predict validation targets
				if status == True:  # ||
					status = self.process(dbo, self.predict, cfg)  # ||predict unknown targets
		return self

	def predict(self, cycl, ftrs, targs, ids, targn, what=None, model=None, limit=0.8):
		"""Create predictions from model and score those predictions if possible
			expand to allow this method to leverage multiple threads on local
			machine to engage multiple processors"""
		logma.info(f'Predict Model {model}')
		if model == None:
			model = self.model
		if model == None:
			return False
		logma.info('Predict Model', model)
		result = DataFrame()
		result['ids'] = ids
		result['BinaryTargets'] = targs
		result['Predicts'] = predict(model, ftrs)  # ||
		result['foldcycl'] = [self.tag for x in range(len(ids))]
		result['algoname'] = [self.algon for x in range(len(ids))]
		result['platform'] = [self.plat for x in range(len(ids))]
		result['runtype'] = [what for x in range(len(ids))]
		result['targetname'] = [targn for x in range(len(ids))]
		scores = self.score(targs, result['Predicts'])
		code = f'{self.code}_{what}_{targn}'
		save(self.writer, result, self.name, code, model, scores, self.start) #  save predictions to db
		if limit != None and scores != None and scores['LogLoss'] > limit:
			logma.info(scores['LogLoss'])

	def run_train(self):
		""""""
		for model in self.models:
			self.setActiveModel(model[0], model[1], model[2])
			self.spawnProcess(self.train, args=(self.features, self.targets))

	def score(self, targs, predicts):
		""" """
		self.initAnalyzer(targs)
		scores = self.test(result['Predicts'], 'full')  # ||
		return scores

	def train(self, ftrs, targs, how='full'):  # ||
		"""
		:param algoN: (str) Algorithm name.
		:param seqN: (str) Sequence name.
		:param platN: (str) Platform name.
		:param ftrs: (pandas.DataFrame) Features.
		:param targs: (pandas.DataFrame) Targets.
		:param ids: (pandas.DataFrame) IDS.
		:param what: (object) Object to determine tags.
		:param how: (str) Training method. Default is 'full'.
		:return: (object) Self.

		Trains the model based on the specified parameters.

		This method:
		- Retrieves the configuration for the specified algorithm, sequence, and platform.
		- Generates tags based on the provided object.
		- Checks if the configuration is valid.
		- Sets the model based on the configuration. If a custom model is provided, it is used instead.
		- Generates code for the training process.
		- Logs the target and feature data.
		- Converts the targets to a DataFrame if necessary.
		- Trains the model using the specified features, targets, model, and additional arguments.
		- Deletes the feature and target data if training is successful.
		- Logs the trained model.
		- Updates the self.model with the trained model.
		- Returns the self object.
		"""
		algoN = self.active_model['algoN']
		seqN = self.active_model['seqN']
		platN = self.active_model['platN']
		TAG = thing.What().uuid().ruuid  # ||
		targTAG, trainTAG = TAG[-5:], TAG[:5]  # ||
		self.code = self._code(self.tag, algoN, seqN, platN, targTAG)  # ||
		if not isinstance(targs, DataFrame):  # ||
			targs = DataFrame(targs)
		model = train(algoN, seqN, platN, ftrs, targs, how, None)  # ||
		del ftrs
		del targs
		gc.collect()
		self.model = model
		return self  # ||

	def add_bias(self, adjusters):  # need to allow various manipulations of weight values...
		""" """
		# between limits of 0 and 1
		# between limits of -1 and 1
		# between limits of -1, i and 1
		if isinstance(adjusters, float):
			df = df.DataFrame(self.data['trainfeats'])
			df.applymap(lambda x: adjusters * x)
			#			df.apply(adjust, axis=1, (adjusters,))
			self.data
		elif isinstance(adjusters, dict):
			pass
		elif isinstance(adjusters, list):
			pass
		return self

	def _code(self, tag, algo, seq, plat, targ):
		self.algon = f'{algo}_{seq}'
		self.plat = plat
		return f'{tag}_{algo}_{seq}_{plat}_{targ}'  # ||

	def _rtr(self, kind, dmap=None):  # ||=>Define method
		"""Route given inputs into the appropriate algorythm"""  # ||=>Describe method
		self.kind = kind
		if dmap == None:  # ||check decision map
			dmap = self.config.dikt['dmap']
		if isinstance(self.FTRs, str):  # ||
			self.dtype = 'STR'  # ||
			self.handler = anal_text  # ||
		elif isinstance(self.FTRs, int) or isinstance(self.FTRs, float):  # ||
			self.dtype = 'INT'  # ||
			self.handler = anal_num  # ||
		else:  # ||
			self.dtype = 'DATA'  # ||
			self.hanlder = ''  # anal_data#										||
		return self  # ||

# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||