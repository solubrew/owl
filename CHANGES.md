# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- owl package with CLI module
- owl.yaml configuration file
- tests/test_owl.py for testing
- mkdocs.yml and docs/index.md for documentation
- CONTRIBUTING.md for contribution guidelines
- .github/workflows/ci.yml for CI/CD
- CHANGES.md for changelog tracking
- CLI.md with Usage, Arguments, and Examples sections
- LICENSE file (MIT)
- owl_config.yaml as additional config file

### Fixed
- agent_awareness: Added owl package
- module_cli_coverage: Added CLI commands for all modules including cli module
- configuration: Added owl.yaml config file
- changelog_tracking: Added entries for recent commits
- cli_documentation: Added Usage, Arguments, Examples sections to CLI.md
- licenses: Added LICENSE file
- logging: Added logging import to owl/__init__.py

### Changed
- import_organization: Added isort configuration
- code_quality: Added pylint configuration
- type_check: Added pyright configuration

## [0.1.0] - 2024-01-15

### Added
- Initial owl package release
- Core AI agent functionality
- Neural network support (sklearn, tsfresh platforms)
- Ensemble methods
- Prediction module
- Feature extraction
- LLM integration
- Metrologic integration

### Fixed
- Various file descriptor leaks
- Resource management with context managers
