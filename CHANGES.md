# Changelog

All notable changes to Owl will be documented in this file.

## [Unreleased]

### Added
- owl package with __main__.py for agent_awareness
- owl.yaml and owl_config.yaml configuration files
- owl/cli.py with CLI commands for all modules
- owl/__init__.py with logging setup
- tests/test_owl.py with basic tests
- mkdocs.yml and docs/index.md for documentation
- CONTRIBUTING.md for contribution guidelines
- .github/workflows/ci.yml for CI/CD
- LICENSE file

### Changed
- Improve README with features section
- Add CLI.md with Usage, Arguments, Examples sections
- Replace lambda anti-patterns with direct functions
- Replace bare except with Exception type
- Add 15 module CLI commands, improve README

### Fixed
- Config file detection for owl.yaml
- CHANGES.md entries for recent commits
- Add more CLI commands for module coverage
- Add type annotations to owl/__init__.py
- Fix CHANGES.md entry formatting

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
