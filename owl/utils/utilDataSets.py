# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""  #																			||
---  #																			||
<(META)>:  #																	||
	DOCid: <^(UUID)^>  #														||
	name:  #																	||
	description: >  #															||
	expirary: <[expiration]>  #													||
	version: <[version]>  #														||
	authority: document|this  #													||
	security: sec|lvl2  #														||
	<(WT)>: -32  #																||
"""
# -*- coding: utf-8 -*-
# ================================Core Modules===================================||
from os.path import abspath, dirname, join

# ===============================================================================||
from pandas import DataFrame
from squirl.orgnql import sonql

# ===============================================================================||
from subtrix import thing

# ===============================================================================||
here = join(dirname(__file__), '')  # ||
there = abspath(join('../../..'))  # ||set path at pheonix level
version = '0.0.0.0.0.0'  # ||
log = True
page = 1000000
# ===============================================================================||
pxcfg = join(here, '_data_', 'cmds.yaml')

def foldDataSets(db, table, tag=None):
	''
	if tag == None:
		increment = thing.What().uuid().ruuid[-5:]
		tag = f'CYCL{increment}'
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
			df = DataFrame(datao.dikt[table]['records'], columns=trainc)  # ||
			datao.dikt[table]['records'] = []
			fd = selector.engine(df).fold('RandomSelection')
			del df
			splitPT = 315
			trainset = fd.dset0.iloc[:, 0:splitPT]
			trainvalidset = fd.dset1.iloc[:, 0:splitPT]
			with sonql.doc(db) as dbi:
				dbi.write({f'{tag}Train': {'columns': trainc[:splitPT],
				                           'records': trainset}})  # ||
				dbi.write({f'{tag}Trainvalid': {'columns': trainc[:splitPT],
				                                'records': trainvalidset}})  # ||
			del trainset
			del trainvalidset
			gc.collect()
	return tag  # ||
def foldDataSets_a000(db, table, tag=None):
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
