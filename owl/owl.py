#@@@@@@@@@@@@@@@@@@@@ Numerai Operation Command Line Interface @@@@@@@@@@@@@@@@@||
'''  #																			||
---  #																			||
<(META)>:  #																	||
	docid:   #								||
	name:  #														||
	description: >  #															||
		Generic prediction algorythm trainer and excerciser.
		The Book class provides a recipe of steps for turning raw data into
		functional models  #													||
		The WorkBook class loads previously trained models and uses them to
		predict outcomes given raw data
	expirary: <[expiration]>  #													||
	version: <[version]>  #														||
	path: <[LEXIvrs]>  #														||
	outline: <[outline]>  #														||
	authority: document|this  #													||
	security: sec|lvl2  #														||
	<(WT)>: -32  #																||
''' #																			||
# -*- coding: utf-8 -*-#														||
#=================================Core Modules==================================||
from os.path import abspath, dirname, join
from sys import argv, path
import psutil, pickle, dill

from memory_profiler import profile
import threading
import multiprocessing as mp
#===============================================================================||
# from fx import calcLimits, createViews, downloadData, extractTARGETs#	||
# from fx import storeTblFile2DB, foldDataSets, outputEnsemble, runTARGBinning
# from fx import testModel, trainModel, predictOutcomes, predictValidation
# from fx import uploadPredictions, ensemble, ensembleSearch, ensembleSelect, outputModel#		||
#===============================================================================||
from excalc import data as calcd, tree as calctr
from condor import condor, thing
from fxsquirl.orgnql import sonql
from owl import predictor, utils
#===============================================================================||
here = join(dirname(__file__),'')#							 		  			||
there = join(dirname(__file__))#												||
where = abspath(join(''))#													||set path at pheonix level
module_path = abspath(join('../../../'))#										||
version = '0.0.0.0.0.0'#														||
page = 200000#
#===============================================================================||
pxcfg = f'{here}_data_/owl.yaml'#								||use default configuration
class Book(predictor.engine):
	''' '''
	def __init__(self, cfg={}):
		''' '''
		self.config = config.instruct(pxcfg).override(cfg)#						||load configuration file
		predictor.engine.__init__(self, self.config)
	def download(self, wdir, name):
		''' '''
		#numerai.portal(wdir).getData(name)#								||
		self.setReader('worldbridger', 'api', 'numerai')
		self.setSink(wdir, 'dir').collect()
	def extract(self):
		''' '''
		self.setReader(path)
		self.collect()
	def fold(self, table, foldType='RandomSelection', sets=[], page=1000000):
		'''Read data from database and fold into sets as needed using the
			given or default page size in order to keep memory limits within
			usable range'''
		self.setReader({'table': table}, {'page': page})
		self.initSelector('fold', 'method', self.name, params)
		self.collect({'table':[table]})
	def binTargets(self):
		''' '''
		self.setReader()
		self.setModifier()
		self.collect()
	def train(self, save=True):
		''' '''
		self.setReader()
		self.process()
		self.collect()
		if save == True:
			self.save()
	def loadModel(self):
		self.setReader()
		self.process()
		self.collect()
	def ensemble(self):
		''' '''
		self.setReader()
		self.process()
		self.collect()
	def save(self):
		''' '''
		self.encodeModel()
		self.collect()
	def output(self):
		''' '''
		self.setReader()
		self.export()
	def upload(self):
		''' '''
		self.send()
class WorkBook(Book):
	''' '''
	def __init_(self):
		''' '''



def runModel(db, ftag, rte, columns, targ, algo, seq, plat, train_targets, tourn_targets):
	''' '''
	targn = int(targ[-1:])#							||
	cfg = [algo, seq, plat, targn]#				||
	d = calcd.stuff(cfg).list_2_str().it#			||
	cfg = [db,  ftag, algo, seq, plat, targn,
									train_targets]#	||
	p = rte['cfg']['srcpath']#						||
	p = f'{p}/{d}.pickle'#				||
	status, model = predictor.trainModel(*cfg)#				||
	try:
		with open(p, 'wb') as f:#						||
			dill.dump(model, f)#						||
	except Exception as e:
		pass
	if status != None and status != False:#			||
		cfg = [model, db, ftag, algo, seq, plat, targn,train_targets, 0.8,
																	columns]#	||
		status = predictor.testModel(*cfg)#					||
	if status == True:
		cfg = [model, db, ftag, algo, seq, plat,
					targn, tourn_targets, columns]
		status = predictValidation(*cfg)

	#model storage

	if status == True:
		cfg = [model, db, ftag, algo, seq, plat,
					targn, tourn_targets, columns]
		status = predictOutcomes(*cfg)

def predictOutcomes(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
															targs, columns):#	||
	'''Predict outcomes using trained model against tournament data '''
	dbo = sonql.doc(db, 0)
#---------------Generate Predictions--------------------------------------------
	cfg = {'tables': ['numerai_tournament_data'], 'page': page,
									'split': [1,3,4,314], 'targBIN': targBIN}#	||
	params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'predict',
									'targn': list(columns.keys())[targBIN]}#	||
	done = utils._procData(cfg, params, dbo, model.predict, targs)
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
	done = utils._procData(cfg, params, dbo, model.predict, targs,
																'validation')#	||
	return done
