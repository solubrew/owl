# Owl - AI Agent Platform

Owl is an AI agent platform for building and deploying machine learning models.

## Features

- **Neural Networks**: Support for sklearn and tsfresh platforms
- **Ensemble Methods**: Combine multiple models for better predictions
- **Feature Extraction**: Advanced feature engineering capabilities
- **LLM Integration**: Connect with large language models
- **Metrologic Integration**: Metrics and monitoring
- **CLI**: Full command-line interface

## Installation

```bash
pip install owl
```

## Usage

```bash
# Train a model
owl train --data data.csv

# Make predictions
owl predict --model model.pkl

# Evaluate a model
owl evaluate --predictions predictions.csv

# List available models
owl list

# Show status
owl status
```

## Configuration

Owl uses `owl.yaml` for configuration. See the configuration file for options.

## Development

See CONTRIBUTING.md for development guidelines.

## License

MIT License - see LICENSE file.
