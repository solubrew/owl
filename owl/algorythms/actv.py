# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
	docid:
	name:
	description: >
	version: 0.0.0.0.0.0
	authority: filesystem
	security: seclvl2
	<(WT)>: -32
"""
# -*- coding: utf-8 -*
# ======================================Standard Library Modules======================================================||
from os.path import abspath, dirname, join

# ======================================3rd Party Library Modules=====================================================||
import numpy as np

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import kahndor
from kahndor.logma import Logma

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')


# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||

class linear(object):
	def fx(self, x):
		""""""
		return x

	def dx(self, x):
		""""""
		return x


class sigmoid():
	def fx(self, x):
		"""The Sigmoid function, which describes an S shaped curve.
			The weighted sum of the inputs normalise between 0 and 1."""
		return 1 / (1 + np.exp(-x))  #applying the sigmoid function

	def dx(self, x):
		"""The derivative of the Sigmoid function.
			The gradient of the Sigmoid curve. Indicates how confident we
			are about the existing weight."""
		return x * (1 - x)  #computing derivative to the Sigmoid function


class tanh():
	def fx(self, x):
		""""""
		return x

	def dx(self, x):
		""""""
		return x


class relu(object):
	def fx(self, x):
		'Rectified Linear function x (numpy array or scalar)'
		if np.isscalar(x):
			y = np.max((x, 0))
		else:
			zero_aux = np.zeros(x.shape)
			meta_x = np.stack((x, zero_aux), axis=-1)
			y = np.max(meta_x, axis=-1)
		return y

	def dx(self, x):
		'Derivative for Rectified Linear function z (numpy array or scalar)'
		y = 1 * (x > 0)
		return y


class leakyrelu(object):
	def fx(self, x):
		""""""
		return x

	def dx(self, x):
		""""""
		return x


class swish(object):
	def fx(self, x):
		""""""
		return x

	def dx(self, x):
		""""""
		return x
