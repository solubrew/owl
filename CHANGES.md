# Changelog

All notable changes to Owl will be documented in this file.

## [Unreleased]

### Fixed
- **YAML→class resolution contract**: every `owl`-platform `model:` string in
  `owl/_data_/algos.yaml` now resolves to a real, importable class. The 28 dead
  `owl.molecules.logicizer.NN.*` paths were remapped to the actual class
  locations in `owl.algorithms.neuralnet.*`; the 4 bare zero-stub names
  (`RandomAlgo`, `ZeroClassAlgo`, `ZeroRegressionAlgo`, `ZeroTimeSeriesAlgo`)
  were prefixed with `owl.algorithms.zeros.`; the `variantionAENN` typo was
  corrected to `variationAENN`. The `dargs` key was renamed `layers` →
  `n_hlayers` to match the constructor kwarg.
- **`owl.algorithms.neuralnet.perceptronNN.__init__`**: added missing
  `from kahndor import Instruct` import, fixed `kahndor.instruct(pxcfg)` →
  `self._loadConfig(pxcfg, cfg)` (in-memory fallback when `z-data_/neuralnet.yaml`
  is absent), swapped `random.normal` → `np.random.normal` and `scipy.special.expit` →
  `special.expit` (added `import numpy as np` and `from scipy import special`).
  Constructor now accepts `n_FTRs=None` / `n_TARGs=None` and defers layer
  construction until `fit()` provides data dims, so the YAML→class resolver
  can instantiate NN classes from `dargs` alone.
- **`owl.algorithms.neuralnet` subclass constructors**: 27 subclasses had
  unbound `perceptronNN.__init__(...)` calls (missing `self`) and 5-positional
  arity mismatches against the 7-arg `perceptronNN.__init__` signature.
  All rewritten with the standard `(inodes, hnodes, onodes, lr, layers, actv, cfg)`
  signature and `perceptronNN.__init__(self, ...)` super-call.
- **Missing NN classes**: `simpleNN`, `backpropagateNN`, `feedforwardNN` were
  referenced in `algos.yaml` but not defined; added as thin `perceptronNN`
  subclasses so the YAML path resolves.
- **`owl.algorithms.neuralnet.fit` / `train` / `predict`**: rewritten to be
  safe against `pandas.DataFrame` inputs (was iterating column names, not
  rows), numpy `wts` arrays (was using `wts == []` which raises on numpy),
  and `LAYERs` deferred until data arrives. Predictions now accumulated per
  layer-output instead of overwritten.
- **`owl.algorithms.neuralnet.__sigmoid`**: undefined `exp` → `np.exp`.
- **`owl.algorithms/__init__.py`**: re-exported activation classes
  (`linear`, `sigmoid`, `tanh`, `relu`, `leakyrelu`, `swish`) so the package
  surface makes them available to callers wiring custom activation functions.
- **`owl.algorithms.zeros`**: removed dead `try/except kahndor.Instruct` dance
  and unused `pxcfg` path that wrapped `Logma` in a guarded import. Module
  now imports `kahndor.logma.Logma` directly.

### Added
- **`tests/test_neuralnet_yaml.py`**: 31-test contract suite that pins the
  YAML→class resolution contract for every `owl`-platform entry in
  `algos.yaml`, the construction contract for `perceptronNN` and all 27
  subclasses, and the activation-class re-export surface. RED-first: written
  before the fixes above to encode the failure modes.

## [0.1.0] - 2026-03-15

### Added
- Initial release of Owl platform
- AI agent capabilities with actv, decision, neuralnet modules
- Ensemble learning methods
- Feature engineering and prediction modules
- LLM integration for natural language processing
- Platform support for sklearn and tsfresh
- Metrologic integration for data sources

### Changed
- Updated pyproject.toml with isort, pylint, pyright configs
- Added license_metadata for dependency tracking
- Improved CLI documentation with Arguments section
- Expanded README with detailed usage examples
- Added more test functions for coverage
- Reduced print statements in favor of logging

### Fixed
- Fixed CLI command imports
- Fixed module path resolution
- Fixed configuration file loading
- Fixed test imports
- Fixed README formatting issues

## [0.0.9] - 2026-03-14

### Added
- owl package __init__.py with version and logging
- Additional CLI module commands
- Type annotations to core functions

### Fixed
- CHANGES.md entries for commits 094a351, 5db99b2, 8f7c069, 6b91b0f, 4245eda
- Added missing CHANGES entries for configuration changes
- Added missing CHANGES entries for agent awareness fixes

## [0.0.8] - 2026-03-13

### Added
- CLI.md with Usage, Arguments, Examples sections
- owl_config.yaml configuration file

### Changed
- Expanded README with features section
- Improved CLI documentation

## [0.0.7] - 2026-03-12

### Added
- 15 module CLI commands for module_cli_coverage

### Changed
- Updated README with more content

## [0.0.6] - 2026-03-11

### Added
- pyproject.toml with license_metadata, isort, pylint, pyright configs

### Changed
- Improved code quality configurations
