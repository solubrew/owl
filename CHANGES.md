# Changelog

All notable changes to Owl will be documented in this file.

## [Unreleased]

### Added
- **owl[decisions]** optional dependency: `pip install owl[decisions]` installs mindot
- `owl/__init__.py`: `DECISIONS_AVAILABLE` flag — `True` when mindot is installed

### Changed
- **`owl/decision.py`**: deprecated; `DeprecationWarning` now points to `mindot.mindot_owl.OwlMethod`
- **`owl/algorithms/algo.py`**: removed (dead copy of `owl/algorithms/zeros.py`)

### Fixed
- **`BREWCE.__init__`**: fixed `BOT.__int__` typo → `BOT.__init__`; removed undefined `model()` reference
- **`devfunctions.storeData()`**: fixed undefined `file_df` variable → `DataFrame()`

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
