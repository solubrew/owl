#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
'''
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
from excalc import data as calcd, tree as calctr, exam, ts as calcts#							||
from condor import condor, thing#										||
from fxsquirl.orgnql import fonql, sonql, yonql#								||
from fxsquirl import analyzer, collector, encoder, worker
#===============================================================================||
here = join(dirname(__file__),'')#												||
there = abspath(join('../../..'))#												||set path at pheonix level
version = '0.0.0.0.0.0'#														||
log = True
#===============================================================================||
def _procData(cfg, params, dbo, fx, targs, filtr=None, ftrs=None, limit=0.8):
	'''Process logicizer using provided data and configurations
		generalize to the processor module

	'''
	print('ProcData Params', params)
	if 'tables' in cfg.keys():
		tables = cfg['tables']
	elif 'views' in cfg.keys():
		tables = cfg['views']
	if len(tables) <= 1:
		table = tables[0].lower()
#	print('Process Cfg', cfg)
	#clctr.setReader({'table': [table]})
	rdr = dbo.read({'table': [table]}, {}, {'page': 200000})
	#need to develop a selective read so that ids can be used to select
	done = None
	while True:
		dataSET = next(rdr, None)
		if dataSET == None or dataSET.dfs[table].empty:
			break
		dset = dataSET.dfs[table]
		params['ids'], CATs, params['ftrs'] = split(dset,dset.columns,cfg['split'])#		||
		if filtr != None:
#			print('Filter Features')
			vUUIDs = CATs.iloc[:,0].isin([filtr])#								||Filter Categories
			params['ftrs'] = params['ftrs'][vUUIDs].reset_index(drop=True)#		||
#			print('IDs', len(params['ids']))
			params['ids'] = params['ids'][vUUIDs].reset_index(drop=True)#		||
#			print('Filter', filtr, 'Filtered IDs', len(params['ids']))
			if len(params['ids']) == 0:
				continue
		if ftrs != None:
			pass
			#build in a feature filter for restricting and analyzing
#		print('Targets', len(targs), 'IDs', len(params['ids']))
		params['targs'] = targs[targs.iloc[:,0].isin(params['ids'])].iloc[:,cfg['targBIN']].reset_index(drop=True)
#		print('Filtered Targets', len(params['targs']))
#		print('Run Function', fx)
		done = fx(**params)#													||Run Function
		if done == None:
			return done
	return done
def _runAlgo(FTRs, TARGs, model, args, how='full'):
	'''Modulize model and fit piecewise if possible and if not fit fully'''
	done = False
	print('FTRs', FTRs.shape, 'TARGs', TARGs.shape)
	if FTRs.shape[0] != TARGs.shape[0]:
		return done
	if isinstance(model, str):
		model = condor.instruct(model).modulize().obj(**args)
	if done == False and how == 'partial':
		try:
			model.partialfit(FTRs,TARGs.values.ravel())
			done = True
			print('Partial Fit Ravel Attempt Successful')
		except Exception as e:
			if "object has no attribute 'partialfit'" in e.__str__():
				how = 'full'
			print('Fit Ravel Attempt Failed',e)
	if done == False and how == 'partial':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]#	||
			model.partialfit(FTRs, TARGfloat)
			done = True
			print('Partial Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			print('Fit Float Attempt Failed',e)
	if done == False and how == 'partial':
		try:
			TARGint = [int(y) for y in TARGs.values.ravel()]#			||
			model.partialfit(FTRs, TARGint)
			done = True
			print('Partial Fit Int Attempt Successful')
			del TARGint
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			print('Partial Fit Int Attempt Failed',e)
	if done == False and how == 'partial':
		try:
			model.partialfit(FTRs,TARGs)
			done = True
			print('Partial Fit Failed')
		except Exception as e:
			if "object has no attribute 'partialfit'" in e:
				how = 'full'
			print('Partial Fit Failed',e)
	if done == False and how == 'full':
		try:
			model.fit(FTRs,TARGs.values.ravel())
			done = True
			print('Full Fit Ravel Attempt Successful')
		except Exception as e:
			print('Fit Ravel Attempt Failed',e)
	if done == False and how == 'full':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]#	||
			model.fit(FTRs, TARGfloat)
			done = True
			print('Full Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			print('Fit Float Attempt Failed',e)
	if done == False and how == 'full':
		try:
			TARGfloat = [float(y) for y in TARGs.values.ravel()]#	||
			FTRsfloat = []
			for row in FTRs:
				for x in row:
					FTRsfloat.append(float(x))
			model.fit(FTRsfloat, TARGfloat)
			done = True
			print('Full Fit Float Attempt Successful')
			del TARGfloat
		except Exception as e:
			print('Fit Float Attempt Failed',e)
	if done == False and how == 'full':
		try:
			TARGint = [int(y) for y in TARGs.values.ravel()]#		||
			model.fit(FTRs, TARGint)
			done = True
			print('Full Fit Int Attempt Successful')
			del TARGint
		except Exception as e:
			print('Full Fit Int Attempt Failed',e)
	if done == False and how == 'full':
		try:
			model.fit(FTRs,TARGs)
			done = True
			print('Full Fit Successful')
		except Exception as e:
			print('Full Fit Failed',e)
	del FTRs
	del TARGs
	gc.collect()
	if done == False:
		return done
	return model
def _save(writer, res, name, code, model, scores={}, start=None):#				||
	'''Save Predictions and Model Object along with a id of the DataSet'''#		||
	predictCOLs = ['targetid','target','predicts','foldcycl','algoname',#		||
										'platform', 'runtype','targetname']#	||

	#change code to standard table
	table = 'predictions'#														||
	#add column for fold cycl, algo_name, platform, outputtype, targetname
	#
	#add memoryt usage to metrics
	#
	try:
		wrdata = {code: {'records': res,'columns': predictCOLs}}#				||
		writer.write(wrdata)#													||
	except Exception as e:
		print('Prediction Write Failed',e,'Save Code', wrdata.keys())#										||
	del wrdata
	try:#																		||
		cfs = model.coef_#														||
	except:
		cfs = None
	print('Save Model',model)
	try:#																		||
		params = model.get_params()#											||
	except:
		print('Save Model Params Failed for ', model)#							||
		params = None
	if scores == None:
		scores = {}
	scores['time'] = calcts.readable(next(thing.when().gen('object'))-start)#	||
	data = [code, name, model.get_params(), cfs, scores]#						||
	serialize = [encoder.engine(data).serialize('list').out]#					||
	seriCols = ['algocode', 'algoname', 'coefs','args', 'scores']#				||
	try:#																		||
		wrdata = {'models': {'records': serialize, 'columns': seriCols}}#		||
		writer.write(wrdata)#													||
	except Exception as e:#														||
		print('Model Score Write Failed',e)#									||
# def loadAlgos():
# 	'''Load configured algorythms from module configuration files'''
# 	algos = f'{here}z-data_/algos.yaml'#								||
# 	algos = condor.instruct(algos).load()
# 	nsmbls = f'{here}z-data_/nsmbls.yaml'#								||
# 	algos.override(nsmbls)
# 	return algos.dikt
# def loadPlats():
# 	''' '''
# 	plats = ['sklearn', 'pheonix','prophet','tensorflow']
# 	return plats
def split(data, columns, splits):#												||
	''' '''
	df = DataFrame(data, columns=columns)#										||
	if len(splits) == 3:#														||
		UUIDs = df.iloc[:,splits[0]]#											||
		CATs = None#															||
		FTRs = df.iloc[:,splits[1]:splits[2]]#									||
	else:
		UUIDs = df.iloc[:,splits[0]]
		CATs = df.iloc[:,splits[1]:splits[2]]
		FTRs = df.iloc[:,splits[2]:splits[3]]#									||
	return UUIDs, CATs, FTRs
def _split(data, columns, splits=[]):#											||
	''' '''#																	||
	df = DataFrame(data, columns=columns)#										||
	osplit, sliver = None, ()#													||
	for nsplit in splits:
		if osplit != None:
			sliver = (*sliver, df.iloc[:,osplit:nsplit])#						||
		osplit = nsplit
	if len(sliver) == 2:#														||
		ids = sliver[0]
		cats = None
		ftrs = sliver[1]
	else:#																		||
		ids = sliver[0]
		cats = sliver[1]
		ftrs = sliver[2]
	return ids, cats, ftrs#														||
def foldDataSets(db, table, tag=None):
	''
	if tag == None:
		tag = 'CYCL{0}'.format(thing.what().uuid().ruuid[-5:])#					||
	else:
		return tag
	with sonql.doc(db) as dbo:
		datar = dbo.read({'tables': [table], 'page': page})
		cnt = 0
		while True:
			cnt += 1
			datao = next(datar, None)
			if datao == None:
				break
			trainc = datao.dikt[table]['columns']
			df = DataFrame(datao.dikt[table]['records'], columns=trainc)#	||
			datao.dikt[table]['records'] = []
			fd = selector.engine(df).fold('RandomSelection')
			del df
			splitPT = 315
			trainset = fd.dset0.iloc[:, 0:splitPT]
			trainvalidset = fd.dset1.iloc[:, 0:splitPT]
			with sonql.doc(db) as dbi:
				dbi.write({tag+'Train': {'columns': trainc[:splitPT], 'records': trainset}})#	||
				dbi.write({tag+'Trainvalid':{'columns': trainc[:splitPT], 'records': trainvalidset}})#	||
			del trainset
			del trainvalidset
			gc.collect()
	return tag#															||
def decompress(self, nfile, npath, date=None):
	''' '''
	date = dt.datetime.now()
	with zipf.ZipFile(f'{npath}/numerai_dataset_{date}.zip', "r") as z:
		z.extractall(zfile)
	z.close()
	return self
