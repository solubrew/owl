#@@@@@@@@@@@@@@@@Pheonix.Molecules.Logicizer.NN@@@@@@@@@@@@@@@@@@@@@@@@@||
'''
---
<(META)>:
	DOCid:
	name: Molecules Level Logicizer Module NN Python Document
	description: >
	version: 0.0.0.0.0.0
	path: <[LEXIvrs]>/panda/LEXI/LEXI.yaml
	outline:
	expire: <^[expire]^>
	authority: document|this
	security: sec|lvl2
	<(WT)>: -32
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================||
import datetime as dt, math, time, random#										||
from os.path import abspath, dirname, join
#===============================================================================||
from numpy import random
#===============================================================================||
from condor import condor#										||
from owl.actv import sigmoid#							||
#===============================================================================||
here = join(dirname(__file__),'')#												||
there = abspath(join('../../..'))#												||set path at pheonix level
version = '0.0.0.0.0.0'#														||
random.seed(32)#																||seeding for random number generation
#===============================================================================||
class perceptronNN(object):
	#A class for sigmoid neuron
	def __init__(self, n_FTRs, n_TARGs, lr=0.001, n_hlayers=1, actv=None,
																	cfg=None):#	||
		pxcfg = '{0}z-data_/neuralnet.yaml'.format(here)#								||use default configuration
		self.config = condor.instruct(pxcfg).override(cfg)#						||load configuration file
		self.n_FTRs = n_FTRs
		self.n_TARGs = n_TARGs
		self._setLearningRate(lr)#												||Set Learning Rate
		self._constructNetwork(n_FTRs, n_TARGs, n_hlayers)#						||
		self._setActivation(actv)#												||Set Activation Function
	def _constructNetwork(self, n_inputs, n_outputs, n_hlayers):
		'Dynamically Construct Network Layers or Override from Configuration'
		self.LAYERs = {0: {'wts': [], 'size': n_inputs}}
		if self.config.dikt['ActiveNetwork'] == None:
			for layer in range(1, n_hlayers+1):
				self.LAYERs[layer] = {}
				size = int(self.LAYERs[layer-1]['size']*0.666666+n_outputs)#	||2/3rds pervious layer + output layer
				wts = random.normal(0.0, pow(size, -0.5), (size, n_inputs))#	||
				self.LAYERs[layer]['wts'] = wts[0]#								||Set Initial Input Weights
				self.LAYERs[layer]['size'] = size#								||
			self.LAYERs[layer+1] = {}
			wts = random.normal(0.0, pow(n_outputs, -0.5), (size, n_outputs))#	||
			self.LAYERs[layer+1]['wts'] = wts[0]#									||Set Initial Output Weights
			self.LAYERs[layer+1]['size'] = n_outputs#							||
		else:#																	||
			self.LAYERs = self.config.dikt['ActiveNetwork']['Construction']#		||
		print(self.LAYERs)
#		with open('layers.txt', 'w') as doc:
#			doc.write(json.dumps(self.LAYERs))
		return self
	def _setActivation(self, actv):
		''
		if actv == None:
			self.actvFx = lambda x: scipy.special.expit(x)
		else:
			self.actvFx = actv
		return self
	def _setLearningRate(self, lr):
		'Initialize Learning Rate and Adjust as needed'
		#need to be able to call this when accuracy drops below a certain threshold from a maximum
		#then adjust the learning rate and restart the learning process 2 iterations back

		self.lr = lr
		return self
	def _setWeigths():
		''
		return self
	def backPropagate(self, ftr, targ, prediction, wts, n_layer):
		''
#		print('Train Prediction', prediction, 'Targ', targ, 'FTR', ftr)
		error = targ - prediction
#		print('Error', error)
#		wts[0] = wts[0] + self.lr * error
#		for i in range(len(ftr)):
#			wts[i] = wts[i] + self.lr * error * ftr[i]
		wts[0] = wts[0] + self.lr * error
		for i in range(len(ftr)-1):
			wts[i + 1] = wts[i + 1] + self.lr * error * ftr[i]
#		self.LAYERs[n_layer]['wts'] = dot(wts, error*self.lr)#					||Multiply error & learning rate by each member
		self.LAYERs[n_layer]['wts'] = wts
		return
	def feedForward(self, X):
		''
		# if self.act_f == 'sigmoid':
		# 	g = lambda x: self.sigmoid(x)
		# elif self.act_f == 'relu':
		# 	g = lambda x: self.relu(x)

		A = [None] * self.n_layers
		Z = [None] * self.n_layers

		input_layer = X
		for n_layer in range(self.n_layers - 1):
			n_examples = input_layer.shape[0]
			if self.bias_flag:# 								||Add bias element to every example in input_layer
				input_layer = np.concatenate((np.ones([n_examples ,1]) ,input_layer), axis=1)
			A[n_layer] = input_layer
			# Multiplying input_layer by theta_weights for this layer

			Z[n_layer + 1] = np.matmul(input_layer,  self.theta_weights[n_layer].transpose() )
			# Activation Function
			output_layer = self.actvFx(Z[n_layer + 1])

			# Current output_layer will be next input_layer
			input_layer = output_layer
		A[self.n_layers - 1] = output_layer

		return A, Z
	def fit(self, FTRs, TARGs, type='predict', n_epoch=None):
		''
		print('Run Fit')
		self.predictions = []
		for row in FTRs:
			for n_layer in self.LAYERs.keys():
				print('Layers', n_layer)
				wts = self.LAYERs[n_layer]['wts']
				if wts == []:
					continue
				prediction = self.predict(row, wts)
		self.predictions.append(prediction)
		return self
	def train(self, FTRs, TARGs, type='predict', n_epoch=None):
		'fit each row of data to each layer of the NN nepoch times'
		print('Run Train')
#		print('Fit FTRs', FTRs[0])
#		print('Fit TARGs', TARGs[0])
		if type == 'predict':
			n_epoch = 1
		for epoch in range(n_epoch):
			for n_layer in self.LAYERs.keys():
				wts = self.LAYERs[n_layer]['wts']
				if wts == []:
					continue
				if n_layer == len(self.LAYERs.keys())-1:
					self.predictions = []
				for n_row in range(len(FTRs)):
					print('Row Number', len(FTRs[n_row]), len(wts))
					if type == 'train' and n_layer != len(self.LAYERs.keys())-1:
						prediction = self.predict(FTRs[n_row], wts)
						self.backPropagate(FTRs[n_row], TARGs[n_row], prediction, wts, n_layer)
		return self
	def predict(self, FTRsRow, weights):
		''
		print('Weights', weights)
		activation = weights[0]
		for i in range(len(FTRsRow)-1):
#			print('Predict I', i)
			activation += weights[i + 1] * FTRsRow[i]
		return 1.0 if activation >= 0.0 else 0.0
# 		actv = wts[0]
# 		print('Predict Weights', weights)
# 		print('Features Row', FTRsRow)
# #		print('Predict', len(FTRsRow), len(weights))
# 		for i in range(len(FTRsRow)-1):
# 			print('Predict I', i)
# 			actv += weights[i] * FTRsRow[i]
# 		print('ACTV',actv)
# 		return 1.0 if actv >= 0.0 else 0.0
# #		return self.actvFx(actv)
	def __sigmoid(self, x):
		'''The Sigmoid function, which describes an S shaped curve.
			We pass the weighted sum of the inputs through this function to
			normalise them between 0 and 1.'''
		return 1 / (1 + exp(-x))
	def __sigmoid_derivative(self, x):
		'''The derivative of the Sigmoid function.
			This is the gradient of the Sigmoid curve.
			It indicates how confident we are about the existing weight.'''
		return x * (1 - x)
class autoencoderNN(perceptronNN):
	'Generative Unsupervised Learning Model for Convolutional Image Vectors'
	'''
	https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/autoencoder.py
	'''
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
	def encode(self):
		''
		return self
	def decode(self):
		''
		return self
class boltzmannmachineNN(perceptronNN):
	'Unconstrained stochastic generative '
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class convolutionalNN(perceptronNN):
	'MLP utilizing convolutional linear statistics for neuron reduction'
	'''
	https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
	'''
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
	def backPropagation(self):
		''
		return self
	def downSampling(self):
		''
		return self
class deconvolutionalNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
	def stackRBMNN(self):
		''
		return self
class restrictedBMNN(boltzmannmachineNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
	def errorCorrection(self):
		'Constrastive Divergence method for updating weights'
		return self
class deepbeliefNN(restrictedBMNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
	def stackRBMNN(self):
		''
		return self
class deepboltzmannmachineNN(restrictedBMNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class deepconvolutionalNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class deepconvolutionalinversegraphicsNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class deepresidualNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class denoisingAENN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class echostateNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class extremelearningmachineNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class generativeadversarialNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
	def generator(self):
		''
		return self
	def discriminator(self):
		''
		return self
class gatedrecurrentunitNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class hopfieldNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class kohonenNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class liquidstatemachineNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class recurrentNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class longshortmemoryNN(recurrentNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class markovchainNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class neuralturingmachineNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class radialbasisNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class sparseAENN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class supportvectormachineNN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
class variationAENN(perceptronNN):
	def __init__(self, inodes, hnodes, onodes, lr, layers):
		perceptronNN.__init__(inodes, hnodes, onodes, lr, layers)
#==========================Source Materials=============================||
'''

'''
#============================:::DNA:::==================================||
'''
---
<@[datetime]@>:
	<[class]>:
		version: <[active:.version]>
		test:
		description: >
			<[description]>
		work:
			- <@[work_datetime]@>
<[datetime]>:
	here:
		version: <[active:.version]>
		test:
		description: >
			<[description]>
		work:
			- <@[work_datetime]@>
'''
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
