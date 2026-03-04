# -*- coding: utf-8 -*-
"""
Owl + MicroMole Integration

Provides data sourcing and storage capabilities using MicroMole.
"""

import sys
import os

# Add micromole to path if not installed
_micromole_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'projects', 'micromole')
if os.path.exists(_micromole_path):
    sys.path.insert(0, _micromole_path)

from micromole.micromole import MicroMole
from micromole.storage import MicroStash
from micromole.sources import (
    DictionarySource, 
    TableSource, 
    FileSource, 
    RecordSource
)

__all__ = [
    'MicroMole',
    'MicroStash', 
    'DictionarySource',
    'TableSource',
    'FileSource',
    'RecordSource',
    'DataSource',
    'OwlStorage',
]

class DataSource:
    """
    Unified data source for Owl using MicroMole.
    
    Provides a common interface for loading data from various sources
    (dictionaries, tables, files, records) and converting to Owl format.
    """
    
    def __init__(self, source_type='dictionary', **kwargs):
        """
        Initialize a data source.
        
        Args:
            source_type: One of 'dictionary', 'table', 'file', 'record'
            **kwargs: Source-specific parameters
        """
        self.source_type = source_type
        self.kwargs = kwargs
        self._source = None
        
    def load(self, data=None):
        """
        Load data from the source.
        
        Args:
            data: Data to load (dict, list, file path, etc.)
            
        Returns:
            Data in Owl-compatible format
        """
        if self.source_type == 'dictionary':
            self._source = DictionarySource(data or {})
            return self._source.get_data()
        elif self.source_type == 'table':
            self._source = TableSource(data, **self.kwargs)
            return self._source.get_data()
        elif self.source_type == 'file':
            self._source = FileSource(data, **self.kwargs)
            return self._source.get_data()
        elif self.source_type == 'record':
            self._source = RecordSource(data, **self.kwargs)
            return self._source.get_data()
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
    
    def to_dataframe(self, data=None):
        """Convert source data to pandas DataFrame."""
        raw_data = data or (self._source.get_data() if self._source else None)
        if raw_data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        try:
            import pandas as pd
            return pd.DataFrame(raw_data)
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
    
    @classmethod
    def from_dict(cls, data):
        """Create a DataSource from a dictionary."""
        return cls(source_type='dictionary').load(data)
    
    @classmethod
    def from_csv(cls, filepath, **kwargs):
        """Create a DataSource from a CSV file."""
        return cls(source_type='file', file_type='csv').load(filepath)
    
    @classmethod
    def from_json(cls, filepath, **kwargs):
        """Create a DataSource from a JSON file."""
        return cls(source_type='file', file_type='json').load(filepath)


class OwlStorage:
    """
    Owl storage wrapper using MicroStash.
    
    Provides caching and persistence for ML training data,
    model outputs, and evaluation results.
    """
    
    def __init__(self, name='owl_cache', backend='memory', **kwargs):
        """
        Initialize Owl storage.
        
        Args:
            name: Storage namespace
            backend: Storage backend ('memory', 'disk', 'sqlite')
            **kwargs: Backend-specific options
        """
        self.name = name
        self.backend = backend
        self.kwargs = kwargs
        self._stash = None
        
    def _get_stash(self):
        """Lazy initialization of MicroStash."""
        if self._stash is None:
            self._stash = MicroStash(
                name=self.name,
                backend=self.backend,
                **self.kwargs
            )
        return self._stash
    
    def save(self, key, value):
        """Save a value to storage."""
        return self._get_stash().set(key, value)
    
    def load(self, key, default=None):
        """Load a value from storage."""
        return self._get_stash().get(key, default)
    
    def delete(self, key):
        """Delete a value from storage."""
        return self._get_stash().delete(key)
    
    def exists(self, key):
        """Check if a key exists."""
        return self._get_stash().exists(key)
    
    def list_keys(self, pattern=None):
        """List all keys, optionally filtered by pattern."""
        return self._get_stash().keys(pattern)
    
    def clear(self):
        """Clear all stored data."""
        return self._get_stash().clear()
    
    # Convenience methods for ML data
    
    def save_dataset(self, name, X, y=None, metadata=None):
        """Save a dataset with optional labels and metadata."""
        data = {'X': X, 'y': y, 'metadata': metadata or {}}
        return self.save(f'dataset:{name}', data)
    
    def load_dataset(self, name):
        """Load a dataset."""
        return self.load(f'dataset:{name}')
    
    def save_model(self, name, model, metrics=None):
        """Save a trained model with metrics."""
        data = {'model': model, 'metrics': metrics or {}, 'version': '0.1.0'}
        return self.save(f'model:{name}', data)
    
    def load_model(self, name):
        """Load a trained model."""
        return self.load(f'model:{name}')
    
    def save_evaluation(self, name, results):
        """Save evaluation results."""
        return self.save(f'eval:{name}', results)
    
    def load_evaluation(self, name):
        """Load evaluation results."""
        return self.load(f'eval:{name}')
