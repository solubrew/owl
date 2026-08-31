# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
	docid:
	name: Module Python Document
	description: >
	version: 0.0.0.0.0.0
	path: <[LEXIvrs]>/panda/LEXI/LEXI.yaml
	outline:
	expire: <^[expire]^>
	authority: document|this
	security: sec|lvl2
	<(WT)>: -32
"""
# -*- coding: utf-8 -*-
# ===============================================================================||
from __future__ import annotations

import datetime as dt
import gc
import logging
import zipfile as zipf
from os.path import dirname, join

logger = logging.getLogger(__name__)

# ===============================================================================||
from pandas import DataFrame

# ===============================================================================||
try:
	from excalc import ts as calcts
except Exception:
	calcts = None  # type: ignore

try:
	from kahndor import Instruct as kahndor
except Exception:
	kahndor = None  # type: ignore

try:
	from subtrix import thing
except Exception:
	thing = None  # type: ignore
from fxsquirl import encoder

# ===============================================================================||
here = join(dirname(__file__), '')  # ||
log = True

# ===============================================================================||

def _procData(cfg, params, dbo, fx, targs, filtr=None, ftrs=None, limit=0.8):
	"""Process logicizer using provided data and configurations
		generalize to the processor module	"""
	logger.debug('ProcData Params: %s', params)
	if 'tables' in cfg.keys():
		tables = cfg['tables']
	elif 'views' in cfg.keys():
		tables = cfg['views']
	if len(tables) <= 1:
		table = tables[0].lower()
	#	print('Process Cfg', cfg)
	# clctr.setReader({'table': [table]})
	rdr = dbo.read({'table': [table]}, {}, {'page': 200000})
	# need to develop a selective read so that ids can be used to select
	done = None
	while True:
		dataSET = next(rdr, None)
		if dataSET == None or dataSET.dfs[table].empty:
			break
		dset = dataSET.dfs[table]
		params['ids'], CATs, params['ftrs'] = split(dset, dset.columns, cfg['split'])  # ||
		if filtr != None:
			#			print('Filter Features')
			vUUIDs = CATs.iloc[:, 0].isin([filtr])  # ||Filter Categories
			params['ftrs'] = params['ftrs'][vUUIDs].reset_index(drop=True)  # ||
			#			print('IDs', len(params['ids']))
			params['ids'] = params['ids'][vUUIDs].reset_index(drop=True)  # ||
			#			print('Filter', filtr, 'Filtered IDs', len(params['ids']))
			if len(params['ids']) == 0:
				continue
		if ftrs != None:
			pass
		# build in a feature filter for restricting and analyzing
		#		print('Targets', len(targs), 'IDs', len(params['ids']))
		params['targs'] = targs[targs.iloc[:, 0].isin(params['ids'])].iloc[:, cfg['targBIN']].reset_index(drop=True)
		#		print('Filtered Targets', len(params['targs']))
		#		print('Run Function', fx)
		done = fx(**params)  # ||Run Function
		if done == None:
			return done
	return done


def _runAlgo(FTRs, TARGs, model, args, how='full'):
	"""Modulize model and fit piecewise if possible and if not fit fully"""
	done = False
	logger.debug('FTRs %s x TARGs %s', FTRs.shape, TARGs.shape)
	if FTRs.shape[0] != TARGs.shape[0]:
		return done
	if isinstance(model, str):
		model = kahndor.instruct(model).modulize().obj(**args)
	if done == False and how == 'partial':
		try:
			model.partialfit(FTRs, TARGs.values.ravel())
			done = True
			logger.debug('Partial Fit Ravel Attempt Successful')
		except Exception as e:
			if "object has no attribute 'partialfit'" in e.__str__():
				how = 'full'
			logger.debug('Fit Ravel Attempt Failed: %s', e)
	if done == False and how == 'partial':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]  # ||
			model.partialfit(FTRs, TARGfloat)
			done = True
			logger.debug('Partial Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			logger.debug('Full Fit Float Attempt Failed: %s', e)
	if done == False and how == 'partial':
		try:
			TARGint = [int(y) for y in TARGs.values.ravel()]  # ||
			model.partialfit(FTRs, TARGint)
			done = True
			logger.debug('Partial Fit Int Attempt Successful')
			del TARGint
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			logger.debug('Partial Fit Int Attempt Failed: %s', e)
	if done == False and how == 'partial':
		try:
			model.partialfit(FTRs, TARGs)
			done = True
			logger.debug('Partial Fit Attempt Successful')
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			logger.debug('Partial Fit Attempt Failed: %s', e)
	if done == False and how == 'full':
		try:
			model.fit(FTRs, TARGs.values.ravel())
			done = True
			logger.debug('Full Fit Ravel Attempt Successful')
		except Exception as e:
			logger.debug('Fit Ravel Attempt Failed: %s', e)
	if done == False and how == 'full':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]  # ||
			model.fit(FTRs, TARGfloat)
			done = True
			logger.debug('Full Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			logger.debug('Full Fit Float Attempt Failed: %s', e)
	if done == False and how == 'full':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]  # ||
			FTRsfloat = []
			for row in FTRs:
				for x in row:
					FTRsfloat.append(float(x))
			model.fit(FTRsfloat, TARGfloat)
			done = True
			logger.debug('Full Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			logger.debug('Full Fit Float Attempt Failed: %s', e)
	if done == False and how == 'full':
		try:
			TARGint = [int(y) for y in TARGs.values.ravel()]  # ||
			model.fit(FTRs, TARGint)
			done = True
			logger.debug('Full Fit Int Attempt Successful')
			del TARGint
		except Exception as e:
			logger.debug('Full Fit Int Attempt Failed: %s', e)
	if done == False and how == 'full':
		try:
			model.fit(FTRs, TARGs)
			done = True
			logger.debug('Full Fit Successful')
		except Exception as e:
			logger.debug('Full Fit Failed: %s', e)
	del FTRs
	del TARGs
	gc.collect()
	if done == False:
		return done
	return model


def _save(writer, res, name, code, model, scores={}, start=None):  # ||
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
		logger.warning('Prediction Write Failed: %s (keys: %s)', e, wrdata.keys())
	del wrdata
	try:  # ||
		cfs = model.coef_  # ||
	except Exception:
		cfs = None
	logger.debug('Save Model: %s', model)
	try:  # ||
		params = model.get_params()  # ||
	except Exception:
		logger.warning('Save Model Params Failed for: %s', model)
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
			logger.warning('Model Score Write Failed: %s', e)


# def loadAlgos():
# 	"""Load configured algorythms from module configuration files"""
# 	algos = f'{here}z-data_/algos.yaml'#								||
# 	algos = kahndor.instruct(algos).load()
# 	nsmbls = f'{here}z-data_/nsmbls.yaml'#								||
# 	algos.override(nsmbls)
# 	return algos.dikt
# def loadPlats():
# 	""" """
# 	plats = ['sklearn', 'pheonix','prophet','tensorflow']
# 	return plats

def split(data, columns, splits):  # ||
	""" """
	df = DataFrame(data, columns=columns)  # ||
	if len(splits) == 3:  # ||
		UUIDs = df.iloc[:, splits[0]]  # ||
		CATs = None  # ||
		FTRs = df.iloc[:, splits[1]:splits[2]]  # ||
	else:
		UUIDs = df.iloc[:, splits[0]]
		CATs = df.iloc[:, splits[1]:splits[2]]
		FTRs = df.iloc[:, splits[2]:splits[3]]  # ||
	return UUIDs, CATs, FTRs


def _split(data, columns, splits=[]):  # ||
	""" """  # ||
	df = DataFrame(data, columns=columns)  # ||
	osplit, sliver = None, ()  # ||
	for nsplit in splits:
		if osplit != None:
			sliver = (*sliver, df.iloc[:, osplit:nsplit])  # ||
		osplit = nsplit
	if len(sliver) == 2:  # ||
		ids = sliver[0]
		cats = None
		ftrs = sliver[1]
	else:  # ||
		ids = sliver[0]
		cats = sliver[1]
		ftrs = sliver[2]
	return ids, cats, ftrs  # ||


def decompress(self, nfile, npath, date=None):
	""" """
	date = dt.datetime.now()
	with zipf.ZipFile(f'{npath}/numerai_dataset_{date}.zip', "r") as z:
		z.extractall(npath)
	return self


# ===============================================================================||
"""
"""
# ===============================================================================||
