# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""  #																			||
---  #																			||
<(META)>:  #																	||
	docid:   #								||
	name:  #														||
	description: >  #															||

	expirary: <[expiration]>  #													||
	version: <[version]>  #														||
	authority: document|this  #													||
	security: sec|lvl2  #														||
	<(WT)>: -32  #																||
"""  # ||
# -*- coding: utf-8 -*-#														||
# =================================Core Modules==================================||
from os.path import dirname, join
from sys import argv

import crow
crow.crowLoad('Owl', 'DELTA')

# ===============================================================================||
from ogma.logma import Logma
from _work.llm import Chat
from squirl.objnql import tblonql
from owl.functions import train, predict
from owl.owl import Owl

# ===============================================================================||
here = join(dirname(__file__), '')  # ||
log = True
logma = Logma(__name__)

# ===============================================================================||
pxcfg = join(here, '_data_', 'owl.yaml')

def predictOutcomes(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
                   targs, columns):  # ||
	"""Predict outcomes using trained model against tournament data """
	with sonql.doc(db, 0) as dbo:
		# ---------------Generate Predictions--------------------------------------------
		cfg = {'tables': ['numerai_tournament_data'], 'page': page,
		       'split': [1, 3, 4, 314], 'targBIN': targBIN}  # ||
		params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'predict',
		          'targn': list(columns.keys())[targBIN]}  # ||
		done = otils._procData(cfg, params, dbo, model.predict, targs)
		return done

def predictValidation(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN,
                      targs, columns):  # ||
	"""Predict target for tournament dataset with validation targets
		available"""
	with sonql.doc(db, 0) as dbo:
		cfg = {'tables': ['numerai_tournament_data'], 'page': page,
		       'split': [1, 3, 4, 314], 'targBIN': targBIN}  # ||
		params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'validate',
		          'targn': list(columns.keys())[targBIN]}  # ||
		done = otils._procData(cfg, params, dbo, model.predict, targs,
		                       'validation')  # ||
		return done


def trainModel(db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN, targets):  # ||
	"""Train algorithm against known targets"""
	setPath = 'REDACTED/DataWorkRepo/NumeraiSETs/'  # ||
	with sonql.doc(db, 0) as dbo:
		with sonql.doc(db) as dbi:
			model = engine().initPredictor(ftag, dbi)
			cfg = {'tables': [f'{ftag}train'], 'page': page, 'split': [2, 3, 5, 315],
			       'targBIN': targBIN}  # ||
			params = {'algoN': algoNAME, 'seqN': algoSEQ, 'platN': algoPLAT,
			          'what': 'train', 'how': 'partial'}  # ||
			done = utils._procData(cfg, params, dbo, model.train, targets)
			return done, model


def testModel(model, db, ftag, algoNAME, algoSEQ, algoPLAT, targBIN, targets,
              limit, columns):  # ||
	"""Test model against holdout set"""
	with sonql.doc(db, 0) as dbo:  # ||
		cfg = {'tables': [f'{ftag}trainvalid'], 'page': page, 'split': [2, 3, 5, 315],
		       'targBIN': targBIN}  # ||
		params = {'cycl': thing.what().uuid().ruuid[:5], 'what': 'test',  # ||
		          'targn': list(columns.keys())[targBIN]}  # ||
		print('Test Model', model.predict)
		done = utils._procData(cfg, params, dbo, model.predict, targets, None, None, limit)  # ||
		return done  #




def runr(args):
	""" """
	if args[1] == 'train':
		algo = 'algo12Z1'
		version = 'a000'
		platform = 'sklearn'
		features = ''
		targets = ''
		path = 'REDACTED/_work/DataRepo/housing.tab'
		df = next(tblonql.Doc(path).read())
		df = df.drop(df.index[[0, 1]])
		target_columns = ['MEDV']
		columns = [x for x in df.columns if x not in target_columns]
		features = df[columns]
		targets = df[target_columns]
		model = train(algo, version, platform, features, targets)
		logma.info(model)
		predictions = _predict(model, features)
		logma.info(predictions)

	elif args[1] == 'owl':
		tag = 'dev'
		owl = Owl(tag)
		algo = 'algo12Z1'
		version = 'a000'
		platform = 'sklearn'
		path = 'REDACTED/_work/DataRepo/housing.tab'
		df = next(tblonql.Doc(path).read())
		df = df.drop(df.index[[0, 1]])
		target_columns = ['MEDV']
		columns = [x for x in df.columns if x not in target_columns]
		features = df[columns]
		targets = df[target_columns]
		owl.train(algo, version, platform, features, targets)



if __name__ == '__main__':
	runr(argv)

# ==========================Source Materials=============================||
"""

"""
# =====


