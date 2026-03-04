#sklearn algorythms are handled directly by processing the yaml functions
#any pre/post processing of data for feeding sklearn will be written here
#@@@@@@@@@@@@@@@@@@@@@@ Pheonix.Molecules.Logicizer.PyOD @@@@@@@@@@@@@@@@@@@@@@@||
'''  #																			||
---  #																			||
<(META)>:  #																	||
	DOCid:   #						||
	name: Pheonix.Molecules.Logicizer.PyOD  #									||
	description: >  #															||
		Integrate Python Outlier Detection Model with Pheonix Logicizer #		||
	expirary: <[expiration]>  #													||
	version: <[version]>  #														||
	path: <[LEXIvrs]>  #														||
	outline: <[outline]>  #														||
	authority: document|this  #													||
	security: sec|lvl2  #														||
	<(WT)>: -32  #																||
''' #																			||
# -*- coding: utf-8 -*-#														||
#================================Core Modules===================================||
from os.path import abspath, dirname, join
#===============================================================================||

#===============================================================================||
here = join(dirname(__file__),'')#									 			||
there = join(dirname(__file__))#												||
where = abspath(join('..'))#													||set path at pheonix level
module_path = abspath(join('../../../'))#										||
version = '0.0.0.0.0.0'#														||
#===============================================================================||
class engine(object):
	def __init__(self):
		''
