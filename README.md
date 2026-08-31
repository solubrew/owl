# Owl

## Project Overview

Owl is an AI Agent Platform for machine learning and prediction tasks. It provides a unified CLI for training, evaluating, and deploying models across multiple ML platforms (sklearn, tsfresh, neuralnet, ensemble) using a configuration-driven approach backed by `kahndor` YAML configs.

**Design principle:** Learning algorithms stay in Owl. Decision orchestration lives in MindOT. Statistical calculations live in Metrologic.

## Features

- **Algorithms**: Random, Zero-baseline, Neural network implementations (MLP, CNN, LSTM, autoencoders, etc.)
- **Prediction Module**: sklearn-based training and prediction pipeline (`owl/prediction.py:Engine`)
- **Metrics**: OwlMetrics, ModelMetrics, OwlEvaluation (sklearn wrappers)
- **Features**: FeatureAnalysis (PCA, t-SNE dimensionality reduction)
- **Decision Module**: Deprecated — decision logic moved to MindOT (`pip install owl[decisions]`)
- **Ensemble Module**: Ensemble voting, mean-shifter, mean-concentrator utilities
- **LLM Module**: Chat integration
- **Platforms**: sklearn and tsfresh backends

## Installation

```bash
pip install owl
pip install owl[metrics]  # with metrologic (statistical calculations)
pip install owl[decisions]  # with mindot (decision orchestration)
```

## Usage

```bash
owl --help
owl actv --help
owl train --data data.csv
owl predict --model model.pkl --input data.csv
```

## CLI Commands

- `owl actv` - Activation functions
- `owl algo` - Algorithm management
- `owl algorythms` - Algorithm library
- `owl config` - Configuration management
- `owl decision` - Decision making (deprecated — use mindot)
- `owl ensemble` - Ensemble methods
- `owl evaluate` - Model evaluation
- `owl features` - Feature engineering
- `owl functions` - Utility functions
- `owl import` - Import data
- `owl import_model` - Import models
- `owl list` - List available models
- `owl llm` - LLM integration
- `owl neuralnet` - Neural network operations
- `owl platforms` - Platform management
- `owl predict` - Make predictions
- `owl sources` - Data sources
- `owl status` - System status
- `owl train` - Train models

## Configuration

Owl uses `owl.yaml` or `owl_config.yaml` for configuration.

## Development

See CONTRIBUTING.md for development guidelines.

## License

MIT License - see LICENSE file
