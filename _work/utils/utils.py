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
from os.path import abspath, dirname, join
import datetime as dt, gc, zipfile as zipf  # , multiprocessing as mp#					||
# ===============================================================================||
from memory_profiler import profile
from pandas import DataFrame

try:
	from joblib import jobdump, jobload
except Exception as e:
	pass
# ===============================================================================||
from excalc import ts as calcts  # ||
from condor import condor, thing  # ||
from fxsquirl import encoder

# ===============================================================================||
here = join(dirname(__file__), '')  # ||
log = True

# ===============================================================================||

def _procData(cfg, params, dbo, fx, targs, filtr=None, ftrs=None, limit=0.8):
	"""Process logicizer using provided data and configurations
		generalize to the processor module	"""
	print('ProcData Params', params)
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







# def loadAlgos():
# 	"""Load configured algorythms from module configuration files"""
# 	algos = f'{here}z-data_/algos.yaml'#								||
# 	algos = condor.instruct(algos).load()
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
