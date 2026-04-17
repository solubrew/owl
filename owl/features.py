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
from sklearn.decomposition import PCA
from pandas import DataFrame

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import kahndor
from kahndor.logma import Logma

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')

class FeatureAnalysis():
	"""The FeatureAnalysis class performs feature analysis on a DataFrame using Principal Component Analysis (PCA).

	Args:
	    df (pandas.DataFrame): The input DataFrame to be processed.
	    n (int, optional): The number of output features to be generated. Default is 10.

	Attributes:
	    df (pandas.DataFrame): The input DataFrame.
	    output_features (int): The number of output features.
	    pca (object): The PCA object.
	    results (numpy.ndarray): The result of the PCA transformation.
	    analysis (pandas.DataFrame): The analysis results, including the feature names and their explained variance ratios.

	"""
	def __init__(self, df, n=10):
		"""
		Initializes an instance of the class.

		:param df: The input DataFrame to be processed.
		:type df: pandas.DataFrame
		:param n: The number of output features to be generated. Default is 10.
		:type n: int

		"""
		self.df = df
		self.output_features = n
		self.pca = None

	def PCA(self):
		"""
		Performs Principal Component Analysis (PCA) on the data.

		:return: None
		"""
		self.pca = PCA(n_components=self.output_features)

	def fit(self):
		"""
		Fits the data using the PCA algorithm.

		:return: None
		"""
		if self.pca is not None:
			self.results = self.pca.fit_transform(self.df)
		return self

	def analyze(self):
		"""
		Analyzes the results of a feature extraction process.

		:return: The analysis results, including the feature names and their explained variance ratios.
		:rtype: DataFrame
		"""
		feature_names = [f"Component {i}" for i in range(1, self.output_features + 1)]
		columns = ['feature_name', 'explained_variance_ratio']
		self.analysis = DataFrame(list(zip(feature_names, self.results.explained_variance_ratios)), columns=columns)
		return self

	def explain(self):
		""""""




# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||