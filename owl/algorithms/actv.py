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
from os.path import dirname, join
from typing import Any

# ======================================3rd Party Library Modules=====================================================||
import numpy as np

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor.logma import Logma

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)
logma.info("actv module initialized")

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')


# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||

class linear:
	def fx(self, x) -> Any:
		"""Identity function."""
		return x

	def dx(self, x) -> Any:
		"""Derivative of identity."""
		return x


class sigmoid:
	def fx(self, x) -> Any:
		"""The Sigmoid function, which describes an S shaped curve.
		The weighted sum of the inputs normalise between 0 and 1."""
		return 1 / (1 + np.exp(-x))  # applying the sigmoid function

	def dx(self, x) -> Any:
		"""The derivative of the Sigmoid function.
		The gradient of the Sigmoid curve. Indicates how confident we
		are about the existing weight."""
		return x * (1 - x)  # computing derivative to the Sigmoid function


class tanh:
	def fx(self, x) -> Any:
		"""Hyperbolic tangent function."""
		return np.tanh(x)

	def dx(self, x) -> Any:
		"""Derivative of hyperbolic tangent."""
		return 1.0 - np.tanh(x) ** 2


class relu:
	def fx(self, x) -> Any:
		"""Rectified Linear function x (numpy array or scalar)."""
		if np.isscalar(x):
			y = np.max((x, 0))
		else:
			zero_aux = np.zeros(x.shape)
			meta_x = np.stack((x, zero_aux), axis=-1)
			y = np.max(meta_x, axis=-1)
		return y

	def dx(self, x) -> Any:
		"""Derivative for Rectified Linear function z (numpy array or scalar)."""
		y = 1 * (x > 0)
		return y


class leakyrelu:
	def fx(self, x) -> Any:
		"""Leaky ReLU function."""
		return np.where(x > 0, x, 0.01 * x)

	def dx(self, x) -> Any:
		"""Derivative of Leaky ReLU."""
		return np.where(x > 0, 1.0, 0.01)


class swish:
	def fx(self, x) -> Any:
		"""Swish activation function: x * sigmoid(x)."""
		return x * (1 / (1 + np.exp(-x)))

	def dx(self, x) -> Any:
		"""Derivative of Swish."""
		sig = 1 / (1 + np.exp(-x))
		return sig + x * sig * (1 - sig)
