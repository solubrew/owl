# -*- coding: utf-8 -*-
"""
Owl - Machine Learning Framework

A Python module for running various machine learning algorithms against data
using YAML configurations.
"""

__version__ = '0.1.0'
__author__ = 'Solutions Brewer'

# Integration modules (these work standalone)
from owl.sources import (
    DataSource,
    OwlStorage,
    MicroMole,
    MicroStash,
    DictionarySource,
    TableSource,
    FileSource,
    RecordSource,
)

from owl.metrologic_integration import (
    Metrologic,
    OwlMetrics,
    ModelMetrics,
    ExperimentTracker,
)

__all__ = [
    '__version__',
    # Data sources (MicroMole)
    'DataSource',
    'OwlStorage',
    'MicroMole',
    'MicroStash',
    'DictionarySource',
    'TableSource',
    'FileSource',
    'RecordSource',
    # Metrics (Metrologic)
    'Metrologic',
    'OwlMetrics',
    'ModelMetrics',
    'ExperimentTracker',
]
