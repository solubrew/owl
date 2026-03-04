# 🦉 OWL - STATE.md

## Project Overview

**Name:** Owl  
**Type:** Machine Learning Framework  
**Repository:** `file://REDACTED/SBST01/vein/GitVein/owl.git`  
**Branch:** `gamma` (cloned to `morg-ws` for workspace)

---

## Description

Owl is a Python module built to run various machine learning algorithms against a set of data using YAML configurations to select the ML package algorithm and parameters. The goal is to collect data on the evaluation of those machine learning packages and algorithms to provide a prediction path to selecting the correct algorithm.

---

## Structure

```
owl/
├── owl/                  # Main package
│   ├── algorythms/       # ML algorithms (neuralnet, zero_algorythms, actv)
│   ├── platforms/       # Platform integrations (tsfresh, sklearn)
│   ├── utils/           # Utilities (utils, utilDataSets)
│   ├── algo.py
│   ├── decision.py
│   ├── functions.py
│   ├── neuralnet.py
│   └── prediction.py
├── cmds/                # Command runners (runTrain, runTrainModel)
├── _work/               # Work-in-progress modules
├── setup.py
└── setup.cfg
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| backoff | Retry logic |
| click | CLI framework |
| defusedxml | XML security |
| futures | Async support (py2) |
| six | Python 2/3 compat |
| pypiwin32 | Windows support |

---

## Issues Identified

### Critical

| Issue | Location | Description |
|-------|----------|-------------|
| **Template placeholders** | `setup.py` | Contains unreplaced `<[...]>` placeholders (name, description, author, etc.) |
| **Broken setup.py** | `setup.py` | Uses `<[name]>` which doesn't exist - will fail at import |
| **No __init__.py** | `owl/` | Missing main package init file |
| **No version defined** | `owl/__init__.py` | No `__version__` variable |

### Moderate

| Issue | Location | Description |
|-------|----------|-------------|
| **Typo in name** | Multiple | "algorythms" should be "algorithms" |
| **setup.cfg mismatch** | `setup.cfg` | Says `packages = find:` but expects `owl` package |
| **No README.rst** | Root | Referenced in setup.cfg but doesn't exist |
| **No CHANGELOG.rst** | Root | Referenced in setup.cfg but doesn't exist |

### Minor

| Issue | Location | Description |
|-------|----------|-------------|
| **Old classifiers** | `setup.cfg` | Python 3.5, 3.6 - outdated |
| **Old wheel version** | `setup.cfg` | wheel<=0.29.0 - very old |
| **Inconsistent naming** | `_work/` vs `owl/` | Duplicate modules in both |

---

## Recommendation

1. **Fix setup.py** - Replace all `<[...]>` placeholders with actual values
2. **Create owl/__init__.py** - Add version and imports
3. **Create README.rst / CHANGELOG.rst** - Or update setup.cfg
4. **Rename algorythms → algorithms** - Fix typo
5. **Test installation** - Verify `pip install -e .` works

---

## Status

- [x] Cloned from GitVein
- [x] Branched to morg-ws
- [x] Issues fixed (MicroMole + Metrologic integrations added)
- [x] Tested installation

---

*Generated: 2026-03-03*
