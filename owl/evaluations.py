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
try:
	from pandas_profiling import ProfileReport
except Exception as e:
	print('pandas_profiling failed due to ', e)
from numpy import histogram
try:
	from sklearn import metrics  # ||
except Exception as e:
	print('sklearn failed due to ', e)
try:
	from tsfresh import extract_features
	from tsfresh import select_features
	from tsfresh.utilities.dataframe_functions import impute
	from tsfresh import extract_relevant_features
except Exception as e:
	print('tsfresh failed due to ', e)
try:
	from memory_profiler import profile
except Exception:
	print('No Memory Profiler')
# ======================================Solutions Brewer Library Modules==============================================||
from kahndor import kahndor
from kahndor.logma import Logma
from subtrix import thing

# ====================================================================================================================||
here = join(dirname(__file__), '')  # ||
logma = Logma(__name__)

# ====================================================================================================================||
pxcfg = join(here, '_data_', 'evaluations.yaml')


class Evaluation(object):
	"""Analyze input data using available and/or identified methods
		build in the ability to analyze the data being handled by the program and
		when best to encode it to an inmemory database or ondisk database
	"""  # ||=>Describe class

	def __init__(self, cfg={}):  # ||=>Initialize class instance
		self.config = kahndor.instruct(pxcfg).override(cfg)  # ||load configuration file
		self.cycle = thing.what().uuid().ruuid[-5:]  # ||Collection Id

	def initAnalyzer(self, data):
		""" """
		self.data = data
		return self

	def accuracy(self, probabilities):
		"""Caclculate the accuracy of the prbabilities provided vs the known
			data """
		self.accuracy = metrics.accuracy_score(self.probabilities, self.data)  # ||
		return self

	def accuracy_details(self):  # ||=>define method
		"""Calculate and return details of correct prediction scores"""  # ||
		self.num_correct = 0  # ||
		for i in self.determinent:  # ||
			self.num_correct += i  # ||
		self.num_total = len(self.determinent)  # ||
		self.corrperc = self.num_correct / self.num_total * 100.0  # ||calc correct percentage
		return self  # ||

	def accuracy_map(self, x, actvtr):  # ||
		"""Calculate Accuracy of knowns to predictions"""  # ||
		self.determinent = []  # ||
		for i in range(len(self.obsv)):  # ||
			if self.knowns[i] == 1 and self.obsv[i] > actvtr:  # ||
				self.determinent.append(1)  # ||
			elif self.knowns[i] == 0 and self.obsv[i] < actvtr:  # ||
				self.determinent.append(1)  # ||
			else:  # ||
				self.determinent.append(0)  # ||
		self.accuracy()  # ||=>display message
		return self  # ||

	def attribute(self):
		""" """
		l = self.data.shape(1) + 1
		for i in range(l):
			isCategorical = analizer(self.data[i]).isCategory()
			if isCategorical == True:
				cols.append(i)
		self.dikt = mutate_data.restructure(self.data).recurdexer(cols, True).data
		return self

	def auc_score(self, probabilities):
		"""Score the Area Under the Curve for the given Probabilities"""
		#		print('Probability columns', probabilities.columns)
		#		if 'ids' in probabilities.columns:
		#			probabilities.drop(columns='ids', inplace=True)
		try:
			self.auc = metrics.roc_auc_score(self.data, probabilities)  # ||Area Under the Receiver Operating Characteristic Curve
		except Exception as e:
			self.auc = None
			d = {'Data': self.data, 'Probabilities': probabilities}  # ||
			print(f'AUC LogLoss Analysis of Probability Failed {d} {e}')
		return self.auc

	def boundary(self):  # ||
		""" """
		self.bndry, cnt, dimcnt = {}, 0, 0  # ||
		for a, b in self.segment:  # ||
			for i in a:  # ||
				if i not in b:  # ||
					j += 1  # ||
			if j == 1:  # ||
				self.bndry[dimcnt].append(a, b)  # ||
				dimcnt += 1  # ||
		return self  # ||=>

	def branch(self):
		"""Define branching coordinate system space"""
		return self

	def calculate(self, calctype: str):
		""" """
		if calctype == 'financial':
			fincalc()  # run basic financial calculations against given data
		elif calctype == 'process':
			pass  # run basic process calculations against given data
		elif calctype == 'scientific':
			pass
		elif calctype == 'statistical':
			pass
		elif calctype == 'Capability':
			pass
		elif calctype == 'Probability':
			pass
		elif calctype == 'Varability':
			pass
		elif calctype == 'Scalability':
			pass
		return self

	def correlations(self):
		"""Compute correlations between each 2 columns of dataset"""
		self.correlations = {}
		c = self.profile['correlations']
		self.correlations['pearson'] = c['pearson']['rating']['rating']
		self.correlations['spearman'] = c['spearman']['rating']['rating']
		self.correlations['kendall'] = c['kendall']['rating']['rating']

	def digitmapping(self):
		'not sure what this is'
		for i in self.num.split('.'):
			sn.append(i)
		if len(sn) > 2:
			debug.fixthis(module + '.inspect.magnitutde')
		rn, ln, l, = [sn[0], sn[1], 10]
		self.magnitude = len(rn) ** l
		self.sigfigs = len(ln)
		self.diglist = list(self.num)
		return self

	def first(self, data):
		"""Move this to the data calcgen module probably"""
		# print('First Data', data)
		return list(data)[0]

	def getStats(self):
		''
		for frame in self.data:
			stat = calcs.stuff(frame)
			self.aset[stat.hashd] = stat.dikt['Stats']
		return self

	def histogram(self, dbin=None):  # ||
		""" """
		drange = ''
		if dbin == None:
			dbin = self.data.max - self.data.min
		self.histogram, self.binedges = histogram(self.data, bins=dbin,
		                                          range=None, normed=False,
		                                          weights=None, density=None)  # ||
		return self

	def isCategory(self, actlvl=1):
		cat = 0
		if self.dtype == 'LIST':
			self.cdtype = analizer.engine(self.data[0])
		if self.cdtype == 'STR':
			cat += 3
		self.uniqueness()
		if self.uniqueness['percent'] == 0.50:
			return True
		if self.uniqueness['percent'] > 0.50:
			cat += 0
		elif self.uniqueness['percent'] <= 0.25:
			cat += 2
		else:
			cat += 1
		if cat > actlvl:
			return True

	def logloss(self, probabilities):
		"""Measure accuracy of predictions"""
		try:
			logloss = metrics.log_loss(self.data, probabilities)
		except Exception as e:
			logloss = None
			m = 'LogLoss Analysis of Probability Failed'
			d = {'Data': self.data, 'Probabilities': probabilities}
			print(f'LogLoss {m} {d} {e}')
		return logloss

	def percMax(self):
		"""Calculate Percentage Max entries for each column in dataset"""
		for col in self.data.columns:
			max = self.data[col].max()
			self.data['pmax_{0}'.format(col)] = self.data[col].apply(x / max, 1)
		return self

	def profile(self):
		"""Expose data set analsis profile from pandas_profiling module"""
		self.profile = self.data.description_set
		for section in self.profile['variables']['channels']['sections'].keys():
			sectd = self.profile['variables']['channels']['sections'][section]
			tables = next(monql.doc(sectd['content']).read()).getTables()
			self.profile['variables']['channels']['sections'][section] = tables
		self.basic = self.profile['table']

		return self

	def residual(self):
		""" """
		r = {}
		for k in data.keys():
			if k == 'prime':
				d0 = data[k]['dset']
				r[data[k]['dhash']] = [0 for x in range(len(d0))]  # ||
			d1 = data[k]['dset']
			r[data[k]['dhash']] = []
			for i in range(len(d0)):
				r[data[k]['dhash']].append(d0[i] - d1[i])
		return self

	def run(self, cfg={}):
		"""Run will allow for detailed data analysis instructions to be created
			and submitted to  the analyzer class via a configuration file"""
		if 'centralities' in cfg.keys():
			pass
		elif 'variabilities' in cfg.keys():
			pass
		elif 'distributions' in cfg.keys():
			histogram()
			pass
		elif 'correlations' in cfg.keys():
			pass
		elif 'probablities' in cfg.keys():
			pass

	def save(self):
		"""Store analysis as a data set in a default database configuration"""
		return self

	def segments(self):  # ||=>Define method
		'Calculate all line segments'  # ||=>Desribe method
		if self.coords == 'cartesian':  # ||
			spt, cnt = [], 0  # ||
			for pt in pts.keys():  # ||
				if spt == []:  # ||
					spt = pt  # ||
					continue  # ||
				if [pt, spt] != self.segment.values():  # ||
					self.segmnt[0] = [pt, spt]  # ||
		elif self.coords == 'polar':  # ||
			spt, cnt = [], 0  # ||
		return self  # ||=>

	def sigmamap(self, lvl: int = 9, sigfig: int = 3):
		"""Map out the Sigma Distribution to the given level for the current
			dataset """
		if self.test('normal_gaussian'):
			for i in range(lvl):
				self.sigma[i] = stddev * i
		else:
			self.sigma = {}  # {i = None for i in range(lvl)}
		return self

	def centerality(self):
		""" """
		if round(float(self.mean), sigfig) > round(float(self.median), sigfig):
			return -1
		elif round(float(self.mean), sigfig) == round(float(self.median), sigfig):
			return 0
		else:
			return 1

	def test(self, predicts, how='full'):
		if how == 'full':
			auc = self.auc_score(predicts)
			logloss = self.logloss(predicts)
			# stats = self.statistical()
			#			scores = {'AUC': auc, 'LogLoss': logloss, 'Mean': self.mean,
			#						'Median': self.median, 'StdDev': self.stddev,
			#						'Length': self.length, 'Width': self.width}#			||
			scores = {'AUC': auc, 'LogLoss': logloss, 'Mean': 0,
			          'Median': 0, 'StdDev': 0,
			          'Length': 0, 'Width': 0}  # ||

		#			print('Scoring Successful',scores)
		return scores

	def variability(self):
		"""Calculate Variablility as the difference between an array of
			statistics"""
		self.CoV = ''  # Population Coefficient of Variation - Error terms of stdDev Simple moving average
		self.smplCoV = ''  # (1+1/4n)*(sstdev/xbar)
		self.e_stddev = e_calc.stddev
		self.e_median = e_calc.median
		self.centrality_gap = self.e_mean - self.e_median
		self.variability = {'sma': calc.sma, 'stddev': calc.stddev,
		                    'centrality_gap': calc.mean - calc.median,
		                    'e_sma': e_calc.sma, 'e_std': e_std,
		                    'e_centrality_gap': e_centrality_gap}
		calc = self.variation()
		e = stuff(self.data).mean_errors().error_terms  # calculate mean error of each datapoint
		e_calc = self.variation()  # calculate the mean and median spread of the error terms
		self.stddev = variablity['stddev']
		self.sma = variablity['sma']
		self.digitmap = self.mapper(self.variablity['unique']).mapped
		self.variance = Decimal(variance().sample(data))  # ||
		self.stddev = Decimal(stdDev(self.variance).sample(data))  # ||standard deviation
		self.threesigma = Decimal(3 * self.stddev)  # ||
		self.digitmap = self.mapper(self.variablity['unique']).mapped  # ||variation mapping
		if self.mean == None:  #
			self.centralities()  #
		self.sigfig = sigfig  #
		if self.scope == 'sample':  #
			n, xbar, some = Decimal(len(data)), self.geometric_mean, 0  #
			for x in data:
				residual = Decimal((Decimal(x) - Decimal(xbar)) ** 2)
				some += residual
			self.variance = Decimal(some / (n - 1))
			if self.variance == None:
				self.stddev = math.sqrt(float(variance().sample(data)))
			else:
				self.stddev = math.sqrt(float(self.variance))
		elif self.scope == 'population':
			n = Decimal(len(data))
			some = 0
			for x in data:
				residual = Decimal((Decimal(x) - Decimal(xbar)) ** 2)
				some += residual
			self.variance = Decimal(some / n)
			if self.variance == None:
				self.stddev = math.sqrt(float(variance().population(data)))
			else:
				self.stddev = math.sqrt(float(self.variance))
		self.mad = findMAD(self.data)  # ||find Median Abs Deviation
		return self

	def volatility(self):
		"""Calculate change of series"""
		self.hist_volatility = 0
		self.impld_volatility = 0
		self.idx_volatility = 0
		self.intraday_volatility = 0
		return self


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
		self.tsne = None

	def PCA(self):
		"""
		Performs Principal Component Analysis (PCA) on the data.

		:return: None
		"""
		self.pca = PCA(n_components=self.output_features)

	def TSNE(self, perplexity=40, n_iter=500):
		"""

		:return:
		"""
		self.tsne = TSNE(n_components=self.output_features, verbose=1, perplexity=perplexity, n_iter=n_iter)

	def fit(self):
		"""
		Fits the data using the PCA algorithm.

		:return: None
		"""
		if self.pca is not None:
			self.results = self.pca.fit_transform(self.df)
		elif self.tsne is not None:
			self.results = self.tsne.fit_transform(self.df)
		return self

	def analyze(self, show=False):
		"""
		Analyzes the results of a feature extraction process.

		:return: The analysis results, including the feature names and their explained variance ratios.
		:rtype: DataFrame
		"""
		feature_names = [f"Component {i}" for i in range(1, self.output_features + 1)]
		columns = ['feature_name', 'explained_variance_ratio']
		self.analysis = DataFrame(list(zip(feature_names, self.results.explained_variance_ratios)), columns=columns)
		if show:
			self._show()
		return self

	def explain(self):
		""""""

	def _show(self):
		""""""
		plt.pie(self.analysis['feature_name'], labels=self.analysis['explained_variance_ratio'], autopct='%1.1f%%')
		plt.show()


# ====================================================================================================================||
"""
https://en.wikipedia.org/wiki/Coefficient_of_variation

"""
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||