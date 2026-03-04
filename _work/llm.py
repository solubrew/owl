# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
'''  #																			||
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
'''  # ||
# -*- coding: utf-8 -*-#														||
# =================================Core Modules==================================||
from os.path import abspath, dirname, join

# ===============================================================================||
here = join(dirname(__file__), '')  # ||
log = True
page = 200000  #
# ===============================================================================||
pxcfg = f'{here}_data_/owl.yaml'  # ||use default configuration

class Chat():
	""""""
	def __init__(self):
		"""
		
		"""
		self.model = GPT4All(model_name='gpt4all-falcon-q4_0.gguf')
		
	def answer(self, input):
		"""
		
		:param input:
		:return:
		"""
		output = self.model.generate(prompt=input)
		return output