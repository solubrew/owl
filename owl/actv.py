#@@@@@@@@@@@@@@@ Numerai Operation Command Line Interface @@@@@@@@@@@@@@||
'''  #																	||
---  #																	||
<(META)>:  #															||
	DOCid:   #						||
	name: Numerai Runner  #					||
	description: >  #													||
		Functions for Completeing a Prediction Operation for the  #		||
		Numerai Dataset  #												||
	expirary: <[expiration]>  #											||
	version: <[version]>  #												||
	path: <[LEXIvrs]>  #												||
	outline: <[outline]>  #												||
	authority: document|this  #											||
	security: sec|lvl2  #												||
	<(WT)>: -32  #														||
''' #																	||
# -*- coding: utf-8 -*-#												||
#===========================Core Modules================================||
from os.path import abspath, dirname, join
#=======================================================================||

#=======================================================================||
here = join(dirname(__file__),'')#							   			||
there = join(dirname(__file__))#										||
where = abspath(join('..'))#									   		||set path at pheonix level
module_path = abspath(join('../../../'))#							   ||
version = '0.0.0.0.0.0'#												||
#=======================================================================||
#lambda x: scipy.special.expit(x)
class linear():
	def fx():
		''
	def dx():
		''
		return x
class sigmoid():
	def fx(self, x):
		'''The Sigmoid function, which describes an S shaped curve.
			The weighted sum of the inputs normalise between 0 and 1.'''
		return 1 / (1 + np.exp(-x))#applying the sigmoid function
	def dx(self, x):
		'''The derivative of the Sigmoid function.
			The gradient of the Sigmoid curve. Indicates how confident we
			are about the existing weight.'''
		return x * (1 - x)#computing derivative to the Sigmoid function
class tanh():
	def fx(self, x):
		''
	def dx(self, x):
		''
class relu(object):
	def fx(self, x):
		'Rectified Linear function x (numpy array or scalar)'
		if np.isscalar(x):
			y = np.max((x, 0))
		else:
			zero_aux = np.zeros(x.shape)
			meta_x = np.stack((x , zero_aux), axis = -1)
			y = np.max(meta_x, axis = -1)
		return y
	def dx(self, x):
		'Derivative for Rectified Linear function z (numpy array or scalar)'
		y = 1 * (x > 0)
		return y
class leakyrelu(object):
	def fx(self, x):
		''
	def dx(self, x):
		''
class swish(object):
	def fx(self, x):
		''
	def dx(self, x):
		''
