#@@@@@@@@@@@@@@Pheonix.Molecules.Predictor.Predictor@@@@@@@@@@@@@@@@@@@@@||
'''
---
<(META)>:
	docid: 7b8f907f-3e26-4db9-8f2b-616d95d7e03d
	name: Molecules Level Predictor Module Python Document
	description: >
		Load and Process Algorythms Selectred through the algos.yaml
		configuration file
		addition configuration files for neuralnets, ensmbles, and activations
		will be included as needed
		Leverage an Array of Machine Learning Frameworks and Make it
		Possible for Infantum to Mulitplex Algorythms across an array of
		Hardware
		the predictor class employs workers in order to divide a dataset and
		run training and prediction in parallelized processes where possible
		this will be done via simple data fold seperation running subsets of
		data independantly and recombining the output to generate a model or
		prediction set
	version: 0.0.0.0.0.0
	expire: <^[expire]^>
	authority: document|this
	security: sec|lvl2
	<(WT)>: -32
'''
# -*- coding: utf-8 -*-
#===============================================================================||
from __future__ import annotations

import gc
import logging
from os.path import abspath, dirname, join
from typing import Any

from pandas import DataFrame

logger = logging.getLogger(__name__)
#===============================================================================||
from excalc import data as calcd  #											||
from fxsquirl import analyzer
from fxsquirl.orgnql import sonql  #											||
from kahndor import kahndor, thing  #										||
from kahndor.logma import Logma

from owl import utils

#===============================================================================||
here = join(dirname(__file__),'')#										||
page = 200000
log = True
logma = Logma(__name__)
#===============================================================================||
pxcfg = join(abspath(here), '_data_', 'predictor.yaml')#					||use default configuration

#uuid = thing.what().uuid().ruuid[-5:]
#fp = open('logicizier_engine{0}.memlog'.format(uuid), 'w+')
#@profile(stream=fp)
class Engine(analyzer.engine):#worker.engine):#								||=>Define class
	'''Analyze input data using available and/or identified methods'''#			||=>Describe class

	def __init__(self, cfg: Any = None) -> None:
		"""Initialize Engine."""
		logma.info("Engine initialized")
		self.config = kahndor.instruct(pxcfg)
		self.config.override(f'{here}_data_/algos.yaml')
		self.config.override(f'{here}_data_/nsmbls.yaml').override(cfg)
		analyzer.engine.__init__(self, self.config)

	def initPredictor(self, tag: str, writer: Any) -> 'Engine':
		"""Initialize predictor with tag and writer."""
		self.cycle = thing.what().uuid().ruuid
		self.FTRs, self.model, self.tag = None, None, tag
		self.writer = writer
		self.start = next(thing.when().gen('object'))
		self.workername = f'WRKR{tag}'
		return self

	def _rtr(self, kind: str, dmap: Any = None) -> 'Engine':
		"""Route given inputs into the appropriate algorithm."""
		self.kind = kind
		if dmap is None:
			dmap = self.config.dikt.get('dmap', {})
		if isinstance(self.FTRs, str):
			self.dtype = 'STR'
		elif isinstance(self.FTRs, (int, float)):
			self.dtype = 'INT'
		else:
			self.dtype = 'DATA'
		return self

	def activation(self, actvtr: float, optzrng: Any, acttype: str = 'binary',
	               cfg: Any = None) -> 'Engine':
		"""Apply activation function."""
		return self

	def ensemble(self) -> 'Engine':
		"""Combine various models for increased accuracy."""
		return self

	def predict(self, cycl: str, ftrs: Any, targs: Any, ids: list,
	           targn: str, what: str | None = None, model: Any = None,
	           limit: float = 0.8) -> bool | DataFrame:
		"""Create predictions from model and score those predictions."""
		if model is None:
			model = self.model
		if model is None:
			return False
		result = DataFrame()
		result['ids'] = ids
		result['BinaryTargets'] = targs
		result['Predicts'] = _predict(model, ftrs)
		result['foldcycl'] = [self.tag for _ in range(len(ids))]
		result['algoname'] = [self.algon for _ in range(len(ids))]
		result['platform'] = [self.plat for _ in range(len(ids))]
		result['runtype'] = [what for _ in range(len(ids))]
		result['targetname'] = [targn for _ in range(len(ids))]
		scores = self.score(targs, result['Predicts'])
		code = f'{self.code}_{what}_{targn}'
		utils._save(self.writer, result, self.name, code, model, scores, self.start)
		if limit is not None and scores is not None and scores.get('LogLoss', 0) > limit:
			return False
		return True

	def run(self, train_targs: Any, valid_targs: Any,
	         runs: Any = None, vplats: Any = None) -> 'Engine':
		"""Run full suite of training and predictions against provided data."""
		return self

	def score(self, targs, predicts):
		''' '''
		self.initAnalyzer(targs)
		scores = self.test(result['Predicts'],'full')#	||
		return scores

	def train(self, algoN, seqN, platN, ftrs,targs,ids,what=None, how='full'):#	||
		'Train Model using the give algorythm'
		logger.debug('CONFIG: %s', self.config.dikt)
		cfg = self.config.dikt[algoN][seqN]['platform'][platN]#					||
		TAG = thing.what().uuid().ruuid#										||
		targTAG, trainTAG = TAG[-5:], TAG[:5]#								||
		if cfg['valid'] == 0:
			return None#														||
		model = cfg['model']#													||
		self.name = cfg['model']
		if self.model is not None:#												||
			model = self.model
		self.code = self._code(self.tag, algoN, seqN, platN, targTAG)#		||
		logger.debug('Train targets: %s', targs.head())
		logger.debug('Features: %s', ftrs.head())
		if not isinstance(targs, DataFrame):#									||
			targs = DataFrame(targs)
		done = False
		if done == False:#														||
			model = utils._runAlgo(ftrs, targs, model, cfg['dargs'], how)#	||
		if done == True:
			del ftrs
			del targs
			gc.collect()
		logger.info('Trained Model: %s', model)
		self.model = model
		return self#															||

	def weighting(self, adjusters):#need to allow various manipulations of weight values...
		''' '''
		#between limits of 0 and 1
		#between limits of -1 and 1
		#between limits of -1, i and 1
		if isinstance(adjusters, float):
			df = df.DataFrame(self.data['trainfeats'])
			df.applymap(lambda x: adjusters * x)
			self.data
		elif isinstance(adjusters, dict) or isinstance(adjusters, list):
			pass
		return self

	def weightTable(self, layers, nodes, ws):
		wt = {}
		for l in layers:
			wt[l] = {}
			for n in nodes:
				wt[l][n] = w
		return self

	def _code(self, tag, algo, seq, plat, targ):
		self.algon = f'{algo}_{seq}'
		self.plat = plat
		return f'{tag}_{algo}_{seq}_{plat}_{targ}'#						||

def predictOutcomes(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
                   targs, columns):  # ||
	'''Predict outcomes using trained model against tournament data '''
	with sonql.doc(db, 0) as dbo:
		cfg = {'tables': ['numerai_tournament_data'], 'page': page,
		       'split': [1, 3, 4, 314], 'targBIN': targBIN}  # ||
		params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'predict',
		          'targn': list(columns.keys())[targBIN]}  # ||
		done = otils._procData(cfg, params, dbo, model.predict, targs)
		return done

def predictValidation(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
                      targs, columns):  # ||
	'''Predict target for tournament dataset with validation targets
		available'''
	with sonql.doc(db, 0) as dbo:
		cfg = {'tables': ['numerai_tournament_data'], 'page': page,
		       'split': [1, 3, 4, 314], 'targBIN': targBIN}  # ||
		params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'validate',
		          'targn': list(columns.keys())[targBIN]}  # ||
		done = otils._procData(cfg, params, dbo, model.predict, targs,
		                       'validation')  # ||
		return done

def trainModel(db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN, targets):  # ||
	'''Train algorithm against known targets'''
	setPath = 'REDACTED/DataWorkRepo/NumeraiSETs/'  # ||
	with sonql.doc(db, 0) as dbo, sonql.doc(db) as dbi:
		model = engine().initPredictor(ftag, dbi)
		cfg = {'tables': [f'{ftag}train'], 'page': page, 'split': [2, 3, 5, 315],
		       'targBIN': targBIN}  # ||
		params = {'algoN': algoNAME, 'seqN': algoSEQ, 'platN': algoPLAT,
		          'what': 'train', 'how': 'partial'}  # ||
		done = utils._procData(cfg, params, dbo, model.train, targets)
		return done, model

def testModel(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN, targets,
              limit, columns):  # ||
	'''Test model against holdout set'''
	with sonql.doc(db, 0) as dbo:  # ||
		cfg = {'tables': [f'{ftag}trainvalid'], 'page': page, 'split': [2, 3, 5, 315],
		       'targBIN': targBIN}  # ||
		params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'test',  # ||
		          'targn': list(columns.keys())[targBIN]}  # ||
		logger.info('Test Model: %s', model.predict)
		done = utils._procData(cfg, params, dbo, model.predict, targets, None, None, limit)  # ||
		return done  # ||

#uuid = thing.what().uuid().ruuid[-5:]#										||
#fp = open('logicizier__procData{0}.memlog'.format(uuid), 'w+')#			||
#@profile(stream=fp)#													||
def _predict(model, x):#													||
	done = False
	try:
		if log: logger.info('Predict From this Model: %s', model)
		r = model.predict_proba(x)
		r = DataFrame(data={'probability': r[:, 0]})
		done = True
	except Exception as e0:
		if log: logger.info('Model %s Prediction Failed: %s', model, e0)
		try:
			r = model.predict(x)
			r = DataFrame(data={'probability': [float(i) for i in r]})
			done = True
		except Exception as e1:
			if log: logger.info('Model %s Prediction Failed: %s', model, e1)
	if done == False:
		return done
	return r

def csvFromDB():
	'''replace with a usage of the generator engine.procDBFile'''
	t = cfg['tables'][0].lower()
	cfg['page'] = 10000000
	with sonql.doc(db) as viewDOC:
		viewRDR = viewDOC.read(cfg)
		cnt = 0
		pxcfg = kahndor.instruct(f'{here}fx.yaml').load().dikt  # ||
		while True:
			data = next(viewRDR, None)
			predicts = DataFrame()
			if data == None:
				logger.debug('Break on count: %s', cnt)
				break
			logger.debug('Data keys: %s', data.dikt.keys())
			df = DataFrame(data.dikt[t]['records'], columns=data.dikt[t]['columns'])
			for model in smodels.keys():
				f = f'{setPath}{When}/Submission_{smodels[model]}.csv'
				twrite = tblonql.doc(f)
				predicts['id'] = df['targetid']
				predicts['prediction_kazutsugi'] = df[smodels[model]]
				ndata = calcd.df2lists(predicts)
				if cnt == 0:
					ndata = [['id', 'prediction_kazutsugi']] + ndata
				twrite.write(ndata)
			cnt += 1
#===============================Code Source Examples============================||
'''
https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
http://fastml.com/what-you-wanted-to-know-about-auc/
'''
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
