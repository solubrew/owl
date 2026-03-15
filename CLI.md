# CLI Documentation

## Commands

### train
Train a model.

```
owl train --config <config_file>
```

### predict
Make predictions.

```
owl predict --model <model_file> --input <input_file>
```

### evaluate
Evaluate a model.

```
owl evaluate --model <model_file> --data <data_file>
```

### export
Export a model.

```
owl export --model <model_file> --format <format> --output <output_file>
```

### import
Import data.

```
owl import --source <source> --format <format>
```

### import_model
Import a model.

```
owl import_model --file <model_file>
```

### config
Manage configuration.

```
owl config --get <key>
owl config --set <key> <value>
```

### list
List available models or datasets.

```
owl list --models
owl list --datasets
```

### status
Show system status.

```
owl status
```
