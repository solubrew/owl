# -*- coding: utf-8 -*-
#==============================Core Modules=====================================||
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import codecs, io, os, re, sys
#===============================================================================||
here = os.path.abspath(os.path.dirname(__file__))
#===============================================================================||
version_file = open(os.path.join(here, 'owl', '__init__.py'), 'r')
__version__ = re.sub(r".*\b__version__\s+=\s+'([^']+)'.*", r'\1',
	[line.strip() for line in version_file if '__version__' in line].pop(0))
version_file.close()
#===============================================================================||
def read(*filenames, **kwargs):
	encoding = kwargs.get('encoding', 'utf-8')
	sep = kwargs.get('sep', '\n')
	buf = []
	for filename in filenames:
		with io.open(filename, encoding=encoding) as f:
			buf.append(f.read())
	return sep.join(buf)

short_description = "Machine Learning Framework with algorithm evaluation"
try:
	long_description = read('README.md')
except IOError:
	long_description = "See README.md where installed."

class PyTest(TestCommand):
	''' '''
	def finalize_options(self):
		''' '''
		TestCommand.finalize_options(self)
		self.test_args = []
		self.test_suite = True

	def run_tests(self):
		''' '''
		import pytest
		errcode = pytest.main(self.test_args)
		sys.exit(errcode)

tests_require = ['pytest']
setup(name='owl', version='0.1.0',
		description='Machine Learning Framework with algorithm evaluation',
		url='https://github.com/solutionsbrewer/owl',
		author='Solutions Brewer',
		author_email='tigrDev@solutionsbrewer.com',
		license='MIT',
		tests_require=tests_require,
		include_package_data=True,
		install_requires=[
			'backoff',
			'click',
			'six',
		],
		cmdclass={'test': PyTest},
		packages=find_packages(),
		platforms='any', zip_safe=False,
		classifiers=[
			"Development Status :: 4 - Beta",
			"Intended Audience :: Developers",
			"License :: OSI Approved :: MIT License",
			"Operating System :: POSIX :: Linux",
			"Operating System :: MacOS :: MacOS X",
			"Operating System :: Microsoft :: Windows",
			"Programming Language :: Python :: 3",
			"Programming Language :: Python :: 3.8",
			"Programming Language :: Python :: 3.9",
			"Programming Language :: Python :: 3.10",
			"Programming Language :: Python :: 3.11",
			"Programming Language :: Python :: 3.12",
		],
		python_requires='>=3.8',
		extras_require={'testing': tests_require, })
#===============================================================================||
