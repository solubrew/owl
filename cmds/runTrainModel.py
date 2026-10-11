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
from zz_work.llm import Chat

# ===============================================================================||
here = join(dirname(__file__), '')  # ||
log = True
logma = Logma(__name__)

# ===============================================================================||
pxcfg = join(here, '_data_', 'owl.yaml')

def runr(args):
	""" """
	if args[1] == 'chat':
		chat = Chat()
		logma.info(chat.answer('Hello'))


if __name__ == '__main__':
	runr(argv)

# ==========================Source Materials=============================||
"""

"""
# =====


