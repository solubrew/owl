#@@@@@@@@@@@@@Pheonix.Molecules.Predictor.Predictor@@@@@@@@@@@@@@@@@@@@@||
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
from os.path import abspath, dirname, join
import datetime as dt, gc#, multiprocessing as mp#					||
#===============================================================================||
from memory_profiler import profile
from pandas import DataFrame
try:
	from joblib import jobdump, jobload
except Exception as e:
	pass
#===============================================================================||
from excalc import data as calcd, tree as calctr, exam#							||
from condor import condor, thing#										||
from fxsquirl.orgnql import fonql, sonql, yonql#								||
from fxsquirl import analyzer, collector, encoder
from owl import utils
#===============================================================================||
here = join(dirname(__file__),'')#												||
page = 200000
log = True
#===============================================================================||
pxcfg = join(abspath(here), '_data_', 'predictor.yaml')#						||use default configuration

#uuid = thing.what().uuid().ruuid[-5:]
#fp = open('logicizier_engine{0}.memlog'.format(uuid), 'w+')
#@profile(stream=fp)
class engine(analyzer.engine):#worker.engine):#													||=>Define class
	'''Analyze input data using available and/or identified methods'''#			||=>Describe class

	def __init__(self, cfg=None):#			||=>Initialize class instance
		self.config = condor.instruct(pxcfg)#						||load configuration file
		self.config.override(f'{here}_data_/algos.yaml')#											||load configuration file
		self.config.override(f'{here}_data_/nsmbls.yaml').override(cfg)
		analyzer.engine.__init__(self, self.config)

	def initPredictor(self, tag, writer):
		''' '''
		self.cycle = thing.what().uuid().ruuid#									||Collection Id
		self.FTRs, self.model, self.tag = None, None, tag#						||
		self.writer = writer
		self.start = next(thing.when().gen('object'))
		#self._rtr(kind)
		self.workername = f'WRKR{tag}'#								||define worker name
		return self

	def _rtr(self, kind, dmap=None):#											||=>Define method
		'''Route given inputs into the appropriate algorythm'''#					||=>Describe method
		self.kind = kind
		if dmap == None:#														||check decision map
			dmap = self.config.dikt['dmap']
		if isinstance(self.FTRs, str):#											||
			self.dtype = 'STR'#													||
			self.handler = anal_text#											||
		elif isinstance(self.FTRs,int) or isinstance(self.FTRs,float):#			||
			self.dtype = 'INT'#													||
			self.handler = anal_num#											||
		else:#																	||
			self.dtype = 'DATA'#												||
			self.hanlder = ''#anal_data#										||
		return self#															||

	def activation(self, actvtr, optzrng,  acttype='binary', cfg={}):#			||
		''' '''
		activations = {}#														||
		if acttype == 'binary':#												||
			activator.switch(fx)#												||
			ual = ''#upper activation limit
			lal = ''#lower activation limit
		if correct.corrperc > score['accuracy']:#								||
			score['accuracy'] = correct.corrperc#								||
			score['activation'] = actvtr#										||
		if actvtr < 0.503:#														||
			actvtr = self.iterate(correct.corrperc,actvtr,optzrng, score)#		||
		if corrperc < self.goal:#												||
			self.actvtr(wrkrng)#												||
		return self#															||

	def ensemble(self):
		'''Combine various models to achieve increased accuracy in trained
			models '''
		return self

	def predict(self, cycl, ftrs, targs, ids, targn, what=None, model=None, limit=0.8):
		'''Create predictions from model and score those predictions if possible
			expand to allow this method to leverage multiple threads on local
			machine to engage multiple processors'''
		print('Predict Model', model)
		if model == None:
			model = self.model
		if model == None:
			return False
		print('Predict Model',model)
		result = DataFrame()
		result['ids'] = ids
		result['BinaryTargets'] = targs
		result['Predicts'] = _predict(model, ftrs)#			||
		result['foldcycl'] = [self.tag for x in range(len(ids))]
		result['algoname'] = [self.algon for x in range(len(ids))]
		result['platform'] = [self.plat for x in range(len(ids))]
		result['runtype'] = [what for x in range(len(ids))]
		result['targetname'] = [targn for x in range(len(ids))]
		scores = self.score(targs, result['Predicts']))#						||
		code = f'{self.code}_{what}_{targn}'
		utils._save(self.writer, result, self.name, code, model, scores, self.start)#	||save predictions to db
		if limit != None and scores != None and scores['LogLoss'] > limit:
			print(scores['LogLoss'])
			return False
		return True

	def run(self, train_targs, valid_targs, runs=None, vplats=None):
		'''Run full suite of training and predictions against provided data

			need to generalize

		'''
		dbo = sonql.doc(db, 0)
		if runs == None:
			runs = algos
		aplats = Predictor.loadPlats()
		if vplats == None:
			vplats = aplats

		cfg = {'tables': [f'{ftag}train'], 'page': page, 'split': [2,3,5,315],
														'targBIN': targBIN}#	||
		params = {'algoN': algoNAME, 'seqN': algoSEQ, 'platN': algoPLAT,
											'what': 'train', 'how': 'partial'}#	||
		columns = {'ids': 1, 'some_targs': 3, 'pmax_targs': 4, 'count_targs': 5}
		cfg = {'tables': [train_targs, ], 'page': page}
		train_targets = selector.procDataExtract(dbn, cfg, columns)
		cfg = {'tables': [valid_targs, ], 'page': page}
		tourn_targets = selector.procDataExtract(dbn, cfg, columns)
		#try going with 200k and 3 processors
		#use threading to train model on two subsets? is this possible?
		#multiprocessing is provided in the next level up and would
		#processes each algorythm against the whole data seperately on a
		#in a different and/or machine
		#with worker integrated processing opening up the path to use multiple cores
		if plat in self.algos[algo][seq]['platform'].keys():#					||
			valid = self.algos[algo][seq]['platform'][plat]['valid']#			||
			if valid == 0:#														||
				return None#														||
			#this should be encapsolated for multiprocessing
			status = None
			status = self.process(dbo, self.train, cfg)#						||train models
			if status != None and status != False:#								||
				status = self.process(dbo, self.predict, cfg)#					||predict test targets
			if status == True:#													||
				status = self.process(dbo, self.predict, cfg)#					||predict validation targets
			if status == True:#													||
				status = self.process(dbo, self.predict, cfg)#					||predict unknown targets
		return self

	def score(self, targs, predicts):
		''' '''
		self.initAnalyzer(targs)
		scores = self.test(result['Predicts'],'full')#	||
		return scores

	def train(self, algoN, seqN, platN, ftrs,targs,ids,what=None, how='full'):#	||
		'Train Model using the give algorythm'
		print('CONFIG', self.config.dikt)
		cfg = self.config.dikt[algoN][seqN]['platform'][platN]#					||
		TAG = thing.what().uuid().ruuid#										||
		targTAG, trainTAG = TAG[-5:], TAG[:5]#									||
#		print('CFG', cfg)
		if cfg['valid'] == 0:
			return None#														||
		model = cfg['model']#													||
		self.name = cfg['model']
		if self.model is not None:#												||
			model = self.model
		self.code = self._code(self.tag, algoN, seqN, platN, targTAG)#			||
#		print('Train Code', self.code)
		print('Train', targs.head())
		print('FTRs', ftrs.head())
		if not isinstance(targs, DataFrame):#									||
			targs = DataFrame(targs)
		done = False
		if done == False:#														||
#			try:#																||
			model = utils._runAlgo(ftrs, targs, model, cfg['dargs'], how)#		||
#				if model == False:
#					return False
#				done = True#													||
#			except Exception as e:#												||
#				print('Non pooled run failed',e)#								||
		if done == True:
			del ftrs
			del targs
			gc.collect()
		print('Trained Model', model)
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
#			df.apply(adjust, axis=1, (adjusters,))
			self.data
		elif isinstance(adjusters, dict):
			pass
		elif isinstance(adjusters, list):
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
		return f'{tag}_{algo}_{seq}_{plat}_{targ}'#			||

def predictOutcomes(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
															targs, columns):#	||
	'''Predict outcomes using trained model against tournament data '''
	dbo = sonql.doc(db, 0)
#---------------Generate Predictions--------------------------------------------
	cfg = {'tables': ['numerai_tournament_data'], 'page': page,
									'split': [1,3,4,314], 'targBIN': targBIN}#	||
	params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'predict',
									'targn': list(columns.keys())[targBIN]}#	||
	done = otils._procData(cfg, params, dbo, model.predict, targs)
	return done

def predictValidation(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
															targs, columns):#	||
	'''Predict target for tournament dataset with validation targets
		available'''
	dbo = sonql.doc(db, 0)
	cfg = {'tables': ['numerai_tournament_data'], 'page': page,
									'split': [1,3,4,314], 'targBIN': targBIN}#	||
	params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'validate',
									'targn': list(columns.keys())[targBIN]}#	||
	done = otils._procData(cfg, params, dbo, model.predict, targs,
																'validation')#	||
	return done

def trainModel(db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN, targets):#		||
	'''Train algorithm against known targets'''
	setPath = '/home/solubrew/DataWorkRepo/NumeraiSETs/'#						||
	dbo = sonql.doc(db, 0)
	dbi = sonql.doc(db)
	model = engine().initPredictor(ftag, dbi)
	cfg = {'tables': [f'{ftag}train'], 'page': page, 'split': [2,3,5,315],
														'targBIN': targBIN}#	||
	params = {'algoN': algoNAME, 'seqN': algoSEQ, 'platN': algoPLAT,
											'what': 'train', 'how': 'partial'}#	||
	done = utils._procData(cfg, params, dbo, model.train, targets)
	return done, model


def testModel(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN, targets,
															limit, columns):#	||
	'''Test model against holdout set'''
	dbo = sonql.doc(db, 0)#														||
	cfg = {'tables': [f'{ftag}trainvalid'], 'page': page, 'split': [2,3,5,315],
														'targBIN': targBIN}#	||
	params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'test',#			||
									'targn': list(columns.keys())[targBIN]}#	||
	print('Test Model', model.predict)
	done = utils._procData(cfg, params, dbo, model.predict, targets, None, None, limit)#	||
	return done#																||

#uuid = thing.what().uuid().ruuid[-5:]#											||
#fp = open('logicizier__procData{0}.memlog'.format(uuid), 'w+')#					||
#@profile(stream=fp)#															||
def _predict(model, x):#														||
	done = False
	try:
		if log: print('Predict From this Model',model)
		r = model.predict_proba(x)
		r = DataFrame(data={'probability': r[:, 0]})
		done = True
	except Exception as e0:
		if log: print('Model',model,'Prediction Failed',e0)
		try:
			r = model.predict(x)
			r = DataFrame(data={'probability': [float(i) for i in r]})
			done = True
		except Exception as e1:
			if log: print('Model',model,'Prediction Failed',e1)
	if done == False:
		return done
	return r


#===============================Code Source Examples============================||
'''
https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
http://fastml.com/what-you-wanted-to-know-about-auc/
'''
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
