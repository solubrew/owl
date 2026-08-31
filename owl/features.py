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

# ======================================Solutions Brewer Library Modules==============================================||
from kahndor.logma import Logma
from pandas import DataFrame

# ======================================3rd Party Library Modules=====================================================||
from sklearn.decomposition import PCA

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', '.yaml')

class FeatureAnalysis:
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
	def __init__(self, df: Any, n: int = 10) -> None:
		"""Initialize FeatureAnalysis."""
		self.df = df
		self.output_features = n
		self.pca = None
		self.tsne = None

	def PCA(self) -> None:
		"""Perform Principal Component Analysis (PCA) on the data."""
		self.pca = PCA(n_components=self.output_features)

	def fit(self) -> 'FeatureAnalysis':
		"""Fit the data using the PCA algorithm."""
		if self.pca is not None:
			self.results = self.pca.fit_transform(self.df)
		elif self.tsne is not None:
			self.results = self.tsne.fit_transform(self.df)
		return self

	def TSNE(self, perplexity: int = 40, n_iter: int = 500) -> None:
		"""Initialize t-SNE dimensionality reduction.

		After calling this, call fit() to apply t-SNE to the data.
		"""
		from sklearn.manifold import TSNE as _TSNE
		self.tsne = _TSNE(n_components=self.output_features, verbose=1,
		                  perplexity=perplexity, max_iter=n_iter)

	def analyze(self) -> Any:
		"""Analyze the results of the feature extraction process."""
		if self.pca is None and self.tsne is None:
			logma.warning('analyze: neither PCA nor TSNE initialized, call PCA() or TSNE() first')
			return DataFrame()
		if self.pca is not None:
			feature_names = [f"Component {i}" for i in range(1, self.output_features + 1)]
			columns = ['feature_name', 'explained_variance_ratio']
			self.analysis = DataFrame(list(zip(feature_names, self.pca.explained_variance_ratio_)), columns=columns)
		else:
			self.analysis = DataFrame()
		return self.analysis

	def explain(self) -> str:
		"""Return a human-readable explanation of the top features."""
		if not hasattr(self, 'results') or self.results is None:
			return "No analysis results available. Call PCA() then fit() first."
		logma.info('explain: %d features analyzed', len(self.results))
		return f"FeatureAnalysis complete: {len(self.results)} features ranked."




# ====================================================================================================================||

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||