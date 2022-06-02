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
def ensemble(self, nsmbl=None):#											||=>define method
	'''Combine algorythms provided into a single logic tree

	rewrite the function below to utilize the _procdata function if possible
	increase dynamics and pulling defaults from the Predictor config file

	'''#					||
	ensmbls = '{0}z-data_/nsmbls.yaml'
	self.config.override(ensmbls)



	nsmblModels = filterModels(db, 'LogLoss', limit)
	drops = ['{0}id', 'foldcycl', 'algoname', 'platform', 'runtype',
				'targetname', 'creby', 'creon','modby','modon','dlt','actv']#	||
	unqinc = thing.what().uuid().ruuid[-5:]
	t = f'prdcts{unqinc}'
	hold = []
	page = 300000
	caps = []
	for i in range(pdepth+len(nsmblModels)):#									||iterate number of CAPs
		caps.append('cap{0}'.format(thing.what().uuid().ruuid[-5:]))#			||create center anchoring predictions
	#build dfault table using number of columns built up from models, targets, and CAPs
	cols = []
	for model in nsmblModels:#													||Iterate ensemble models
		for targn in range(4):#													||Iterate number of targets range
			cols.append(f'{model}_predict_targ{targn}'.lower())#	||Form prediction table name
			if f'targ{targn}' not in cols:
				cols.append(f'targ{targn}')
	cols += caps + ['mean', 'stddev', 'mad']
	first = True
	for model in nsmblModels:#													||
		for targn in range(4):#													||
			table = f'{model}_predict_targ{targn}'.lower()#			||
			rdr = sonql.doc(db).read({'tables': [table], 'page': page})#		||Define reader
			while True:#														||
				data = next(rdr, None)#											||Page through data
				if data == None:#												||
					first = False#												||
					page = 10
					break#														||
				nsmbl = DataFrame(data.dikt[table]['records'],#					||
									columns=data.dikt[table]['columns'])#		||
				for i in drops:
					nsmbl.drop(i.format(table), 1, inplace=True)#				||
				if first == True or hold == None:
					pad = [0.5 for x in range(nsmbl.shape[0])]
					for col in cols:
						nsmbl[col] = pad
					nsmbl[f'targ{targn}'] = nsmbl['target']
					nsmbl[table] = nsmbl['predicts']
					nsmbl.drop('predicts', 1, inplace=True)
					nsmbl.drop('target', 1, inplace=True)
					with sonql.doc(db) as dbi:
						data = calcd.df2lists(nsmbl)
						wrdata = {t: {'records': data, 'columns': nsmbl.columns}}
						dbi.write(wrdata)#										||
				else:
#					print('Run not first', model, targn)
					nsmbl[table] = nsmbl['predicts']
					id = nsmbl['targetid']
					nsmbl[f'targ{targn}'] = nsmbl['target']
					nsmbl.drop('predicts', 1, inplace=True)
					nsmbl.drop('target', 1, inplace=True)
					wrdata = []
					for rown in range(len(nsmbl)):#iterate through rows of dataframe
						for col in nsmbl.columns:# iterate through list of columns only one column is going to be per table
							if col == 'targetid':
								continue
							wrdata.append({col: nsmbl[col].iloc[rown],
										'where':{'targetid': id.iloc[rown]}})#	||
					with sonql.doc(db) as dbi:
#						print('Update', t, wrdata)
						dbi.update({t: wrdata})#									||
def _nsmbl(predicts):
	r = None
	for df in predicts:
		for col in df:
			if r == None:
				r = col
				continue
			r = geometricAVG(r, col)#											||
	return r

def meanshifter(db, df, scores, how='score', algocode=None):
	''' '''
	df['cap3864e'] = [0.5 for x in range(df.shape[0])]
	df['mean'] = df.mean(axis=1)
	df['stddev'] = df.std(axis=1)
	df['mad'] = df.mad(axis=1)
	meanshift1, meanshift2, meanshift3, meanshift4 = [], [], [], []
	for r in range(df.shape[0]):
		if df['mean'].iloc[r] > 0.5:
			shift = df['mean'].iloc[r]+df['stddev'].iloc[r]/21#	||
			shift2 = df['mean'].iloc[r]+df['stddev'].iloc[r]/2#	||
			shift3 = df['mean'].iloc[r]-df['stddev'].iloc[r]/21#	||
			shift4 = df['mean'].iloc[r]-df['stddev'].iloc[r]/2
		else:
			shift = df['mean'].iloc[r]-df['stddev'].iloc[r]/21#					||
			shift2 = df['mean'].iloc[r]-df['stddev'].iloc[r]/2
			shift3 = df['mean'].iloc[r]+df['stddev'].iloc[r]/21#				||
			shift4 = df['mean'].iloc[r]+df['stddev'].iloc[r]/2#					||
		meanshift1.append(shift)
		meanshift2.append(shift2)
		meanshift3.append(shift3)
		meanshift4.append(shift4)
	df['meanshift'] = meanshift1
	df['meanshift2'] = meanshift2
	df['meanshift3'] = meanshift3
	df['meanshift4'] = meanshift4
	return df
def meanconcentrator(concentrate=0.5):
	'''Add detail to strength the mean towards a given value '''
	return df
