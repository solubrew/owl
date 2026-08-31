# Owl CLI

## Purpose

Owl is an AI Agent Platform providing a unified ML model management
CLI. It trains, evaluates, and deploys models across multiple
platforms (sklearn, tsfresh, neuralnet, ensemble) using a
configuration-driven approach backed by `kahndor` YAML configs.

## Usage

```bash
owl [command] [options]
```

## Arguments

| Argument | Description |
|----------|-------------|
| command | The owl command to run |
| options | Optional flags and arguments |

## Examples

```bash
# Train a model
owl train --data data.csv

# Make predictions
owl predict --model model.pkl

# Evaluate a model
owl evaluate --predictions predictions.csv

# List available models
owl list

# Export a model
owl export --model model.pkl --format pickle
```

## Commands

- `actv` - Actv module
- `algo` - Algo module
- `algorythms` - Algorythms module
- `config` - Configuration
- `decision` - Decision module
- `ensemble` - Ensemble module
- `evaluate` - Evaluate module
- `evaluations` - Evaluations module
- `export` - Export module
- `features` - Features module
- `functions` - Functions module
- `import` - Import module
- `import_model` - Import model
- `list` - List available models
- `llm` - LLM module
- `metrologic_integration` - Metrologic integration
- `neuralnet` - Neural network module
- `owl` - Main owl module
- `owl_main` - Owl main
- `platforms` - Platforms module
- `predict` - Predict module
- `sources` - Sources module
- `status` - Show status
- `train` - Train a model
