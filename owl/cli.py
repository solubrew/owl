"""Owl CLI - Command Line Interface"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pandas import DataFrame, read_csv

try:
    from joblib import dump as _joblib_dump, load as _joblib_load
except Exception:
    _joblib_dump = None
    _joblib_load = None

from owl._platforms import load_platforms
from owl.owl import Owl

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".owl" / "models"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------------------------------- #

def _load_csv(path: str, target_col: Optional[str] = None) -> tuple[DataFrame, Optional[DataFrame], List[str]]:
    """Load CSV and optionally split into features and targets."""
    df = read_csv(path)
    logger.info("Loaded %s rows, %s cols from %s", df.shape[0], df.shape[1], path)

    if target_col and target_col in df.columns:
        ftrs = df.drop(columns=[target_col])
        targs = DataFrame(df[target_col].values, columns=[target_col])
        return ftrs, targs, list(ftrs.columns)
    for col in ["target", "label", "y"]:
        if col in df.columns:
            ftrs = df.drop(columns=[col])
            targs = DataFrame(df[col].values, columns=[col])
            return ftrs, targs, list(ftrs.columns)

    return df, None, list(df.columns)


def _find_target(df: DataFrame) -> Optional[str]:
    """Heuristically find a target column."""
    for col in ["target", "label", "y", "class", "output"]:
        if col in df.columns:
            return str(col)
    for col in reversed(df.columns):
        unique = df[col].dropna().unique()
        if 1 < len(unique) <= 10:
            return str(col)
    return None


# --------------------------------------------------------------------------------------------------- #
# Algorithm registry
# --------------------------------------------------------------------------------------------------- #

def _list_algorithms() -> List[Dict[str, str]]:
    """List all available algorithms from algos.yaml."""
    algos_path = Path(__file__).parent / "_data_" / "algos.yaml"
    if not algos_path.exists():
        return []
    with open(algos_path) as f:
        data = yaml.safe_load(f) or {}
    results = []
    for algo_key, algo_data in data.items():
        if not isinstance(algo_data, dict):
            continue
        name = algo_data.get("name") or algo_key
        desc = str(algo_data.get("status") or "").strip()[:80]
        plats = []
        for seq_data in algo_data.values():
            if not isinstance(seq_data, dict):
                continue
            for plat_name, plat_cfg in seq_data.get("platform", {}).items():
                if plat_cfg.get("valid"):
                    plats.append(plat_name)
        results.append({
            "key": algo_key,
            "name": str(name).strip(),
            "desc": desc,
            "platforms": ", ".join(sorted(set(plats))),
        })
    return results


# --------------------------------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------------------------------- #

def cmd_train(args: argparse.Namespace) -> int:
    """Train a model."""
    data_path = args.data  # noqa: SIM401
    target_col = args.target
    algo = args.algo
    seq = args.sequence
    platform = args.platform
    model_out = args.model

    if data_path:
        ftrs, targs, cols = _load_csv(data_path, target_col)
        if targs is None:
            targ_col = _find_target(ftrs)
            if targ_col:
                targs = DataFrame(ftrs[targ_col].values, columns=[targ_col])
                ftrs = ftrs.drop(columns=[targ_col])
                logger.info("Auto-detected target column: %s", targ_col)
            else:
                logger.error("No target column found. Use --target to specify one.")
                return 1
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        ftrs = DataFrame(X, columns=[f"f{i}" for i in range(10)])
        targs = DataFrame(y, columns=["target"])
        logger.info("No --data provided; generated demo dataset (200 rows, 10 features)")

    logger.info("Training: algo=%s seq=%s platform=%s", algo, seq, platform)
    owl = Owl(f"cli_{os.getpid()}")
    owl.setActiveModel(algo, seq, platform)
    owl.train(ftrs, targs)

    if owl.model is None:
        logger.error("Training failed (model is None)")
        return 1

    out = model_out or str(DEFAULT_MODEL_DIR / f"{owl.code}.joblib")
    if _joblib_dump:
        _joblib_dump(owl.model, out)
        logger.info("Saved to %s", out)
        print(f"Trained: {owl.code}")
        print(f"Saved:   {out}")
    else:
        print(f"Trained: {owl.code} (joblib unavailable, not saved)")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Make predictions."""
    model_path = args.model  # noqa: SIM401
    data_path = args.input
    output_path = args.output

    if not model_path:
        models = sorted(DEFAULT_MODEL_DIR.glob("*.joblib"), key=os.path.getmtime, reverse=True)
        if models and _joblib_load:
            model_path = str(models[0])
        else:
            logger.error("No --model specified and none found in %s", DEFAULT_MODEL_DIR)
            return 1

    if not _joblib_load:
        logger.error("joblib not available")
        return 1

    model = _joblib_load(model_path)
    logger.info("Loaded model from %s", model_path)

    if data_path:
        ftrs, _, _ = _load_csv(data_path, target_col=None)
    else:
        from sklearn.datasets import make_classification
        n_feat = getattr(model, "n_features_in_", 10)  # noqa: SIM401
        X, _ = make_classification(n_samples=20, n_features=n_feat, random_state=42)
        ftrs = DataFrame(X)
        logger.info("No --input provided; generated 20 demo rows")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(ftrs)
        preds = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
    else:
        preds = model.predict(ftrs)

    result = DataFrame({"prediction": preds})
    if output_path:
        result.to_csv(output_path, index=False)
        logger.info("Saved predictions to %s", output_path)
        print(f"Wrote {len(result)} predictions to {output_path}")
    else:
        print(result.to_string(index=False))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a model against labelled data."""
    try:
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    except ImportError:
        logger.error("scikit-learn required for evaluation")
        return 1

    model_path = args.model  # noqa: SIM401
    data_path = args.data
    target_col = args.target

    if not model_path:
        models = sorted(DEFAULT_MODEL_DIR.glob("*.joblib"), key=os.path.getmtime, reverse=True)
        if models and _joblib_load:
            model_path = str(models[0])
        else:
            logger.error("No --model and none found in %s", DEFAULT_MODEL_DIR)
            return 1

    if not data_path:
        logger.error("No --data specified")
        return 1

    model = _joblib_load(model_path) if _joblib_load else None
    if model is None:
        return 1

    ftrs, targs, _ = _load_csv(data_path, target_col)
    if targs is None:
        targ_col = _find_target(ftrs)
        if targ_col:
            targs = DataFrame({targ_col: ftrs[targ_col].values})
            ftrs = ftrs.drop(columns=[targ_col])
        else:
            logger.error("No target column found")
            return 1

    tarr = targs.values.ravel()

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(ftrs)
        pclass = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
        auc = roc_auc_score(tarr, pclass)
        ll = log_loss(tarr, probs)
    else:
        pclass = model.predict(ftrs)
        auc, ll = 0.0, 0.0

    acc = accuracy_score(tarr, pclass)

    print(f"Model:       {model_path}")
    print(f"Samples:     {len(tarr)}")
    print(f"Accuracy:    {acc:.4f}")
    if ll:
        print(f"LogLoss:     {ll:.4f}")
    if auc:
        print(f"ROC AUC:     {auc:.4f}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export/copy a model file."""
    src = args.model  # noqa: SIM401
    if not src:
        models = sorted(DEFAULT_MODEL_DIR.glob("*.joblib"), key=os.path.getmtime, reverse=True)
        if models:
            src = str(models[0])
        else:
            logger.error("No --model specified and none found")
            return 1
    dst = args.output or str(DEFAULT_MODEL_DIR / f"exported_{Path(src).name}")
    shutil.copy2(src, dst)
    print(f"Exported: {dst}")
    return 0


def cmd_import_model(args: argparse.Namespace) -> int:
    """Import a model into the models directory."""
    src = Path(args.path)
    if not src.exists():
        logger.error("File not found: %s", src)
        return 1
    name = args.name or src.name
    dst = DEFAULT_MODEL_DIR / name
    shutil.copy2(src, dst)
    print(f"Imported: {dst}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List trained models."""
    models = sorted(DEFAULT_MODEL_DIR.glob("*.joblib"), key=os.path.getmtime, reverse=True)
    if models:
        print("Trained models:")
        for m in models:
            age = _format_age(os.path.getmtime(m))
            print(f"  {m.name}  ({age})")
    else:
        print(f"No models in {DEFAULT_MODEL_DIR}")
        print("Run: owl train --data data.csv")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status."""
    print("Owl v0.1.0 - AI Agent Platform")
    print(f"Model dir:  {DEFAULT_MODEL_DIR}")

    plats = load_platforms()
    print(f"Platforms:  {', '.join(plats) or 'none'}")

    for pkg, name in [("sklearn", "scikit-learn"), ("pandas", "pandas"), ("joblib", "joblib")]:
        try:
            mod = import_module(pkg)
            print(f"{name:<12} {mod.__version__}")
        except Exception:
            print(f"{name:<12} not installed")

    models = list(DEFAULT_MODEL_DIR.glob("*.joblib"))
    print(f"Models:     {len(models)} trained")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show owl.yaml configuration."""
    cfg_path = Path(__file__).parent / "_data_" / "owl.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        print(yaml.dump(data, default_flow_style=False, sort_keys=False))
    else:
        print("(no owl.yaml found)")
    return 0


def cmd_list_algos(args: argparse.Namespace) -> int:
    """List available algorithms."""
    algos = _list_algorithms()
    if not algos:
        print("No algorithms found")
        return 1
    for a in algos:
        print(f"{a['key']:<15} {a['name']:<30} [{a['platforms']}]")
        if a["desc"]:
            print(f"  {a['desc'][:70]}")
    return 0


# --------------------------------------------------------------------------------------------------- #
# Module info commands
# --------------------------------------------------------------------------------------------------- #

def _module_info(name: str, desc: str = "") -> None:
    print(f"{name} module")
    if desc:
        print(f"  {desc}")


def cmd_actv(args: argparse.Namespace) -> int:
    _module_info("Activation Functions", "Neural network activation functions")
    return 0


def cmd_algo(args: argparse.Namespace) -> int:
    return cmd_list_algos(args)


def cmd_algorythms(args: argparse.Namespace) -> int:
    return cmd_list_algos(args)


def cmd_decision(args: argparse.Namespace) -> int:
    _module_info("Decision", "Decision-making capabilities")
    return 0


def cmd_ensemble(args: argparse.Namespace) -> int:
    _module_info("Ensemble", "Ensemble learning methods")
    return 0


def cmd_evaluations(args: argparse.Namespace) -> int:
    _module_info("Evaluations", "Model evaluation utilities")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    _module_info("Feature Engineering", "Feature analysis and engineering")
    return 0


def cmd_functions(args: argparse.Namespace) -> int:
    _module_info("Functions", "Core ML: train, predict, save")
    return 0


def cmd_llm(args: argparse.Namespace) -> int:
    _module_info("LLM", "Large language model integration")
    return 0


def cmd_metrologic_integration(args: argparse.Namespace) -> int:
    _module_info("Metrologic Integration", "Metrologic data source")
    return 0


def cmd_neuralnet(args: argparse.Namespace) -> int:
    _module_info("Neural Network", "Neural network implementations")
    return 0


def cmd_owl_main(args: argparse.Namespace) -> int:
    _module_info("Owl", "Core ML orchestration engine")
    return 0


def cmd_cli(args: argparse.Namespace) -> int:
    _module_info("CLI", "Command-line interface")
    return 0


def cmd_platforms(args: argparse.Namespace) -> int:
    _module_info("Platforms", "ML platform integrations")
    plats = load_platforms()
    for p in plats:
        print(f"  - {p}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    _module_info("Sources", "Data source integrations")
    return 0


# --------------------------------------------------------------------------------------------------- #
# Parser builder
# --------------------------------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="owl", description="Owl - AI Agent Platform")
    subparsers = parser.add_subparsers(dest="cmd", title="commands")

    # train
    p = subparsers.add_parser("train", help="Train a model")
    p.add_argument("--data", help="CSV file with features and target")
    p.add_argument("--target", help="Target column name")
    p.add_argument("--model", help="Output path for trained model")
    p.add_argument("--algo", default="algo12Z1", help="Algorithm key (default: algo12Z1)")
    p.add_argument("--sequence", default="a000", help="Algorithm sequence (default: a000)")
    p.add_argument("--platform", default="sklearn", help="ML platform (default: sklearn)")
    p.set_defaults(func=cmd_train)

    # predict
    p = subparsers.add_parser("predict", help="Make predictions")
    p.add_argument("--model", help="Model file (.joblib)")
    p.add_argument("--input", "-i", help="Input CSV file")
    p.add_argument("--output", "-o", help="Output CSV file")
    p.set_defaults(func=cmd_predict)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate a model")
    p.add_argument("--model", help="Model file (.joblib)")
    p.add_argument("--data", help="CSV with features and target")
    p.add_argument("--predictions", dest="data", help="Alias for --data")
    p.add_argument("--target", help="Target column name")
    p.set_defaults(func=cmd_evaluate)

    # export
    p = subparsers.add_parser("export", help="Copy/export a model file")
    p.add_argument("--model", help="Source model file")
    p.add_argument("--output", "-o", help="Destination file")
    p.set_defaults(func=cmd_export)

    # import
    p = subparsers.add_parser("import", help="Import a model into the models directory")
    p.add_argument("path", help="Path to model file")
    p.add_argument("--name", help="Name for the imported model")
    p.set_defaults(func=cmd_import_model)

    # list
    p = subparsers.add_parser("list", help="List trained models")
    p.set_defaults(func=cmd_list)

    # status
    p = subparsers.add_parser("status", help="Show system status")
    p.set_defaults(func=cmd_status)

    # config
    p = subparsers.add_parser("config", help="Show owl.yaml configuration")
    p.set_defaults(func=cmd_config)

    # list-algos
    p = subparsers.add_parser("list-algos", help="List available algorithms")
    p.set_defaults(func=cmd_list_algos)

    # Module info commands
    for cmd_name, func in [
        ("actv", cmd_actv),
        ("algo", cmd_algo),
        ("algorythms", cmd_algorythms),
        ("decision", cmd_decision),
        ("ensemble", cmd_ensemble),
        ("evaluations", cmd_evaluations),
        ("features", cmd_features),
        ("functions", cmd_functions),
        ("llm", cmd_llm),
        ("metrologic_integration", cmd_metrologic_integration),
        ("neuralnet", cmd_neuralnet),
        ("owl", cmd_owl_main),
        ("cli", cmd_cli),
        ("platforms", cmd_platforms),
        ("sources", cmd_sources),
    ]:
        p = subparsers.add_parser(cmd_name, help=f"{cmd_name} module info")
        p.set_defaults(func=func)

    return parser


# --------------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------------- #

def _format_age(mtime: float) -> str:
    """Format a file mtime as a human-readable age string."""
    import time, datetime
    age_s = time.time() - mtime
    if age_s < 60:
        return f"{age_s:.0f}s ago"
    if age_s < 3600:
        return f"{age_s/60:.0f}m ago"
    if age_s < 86400:
        return f"{age_s/3600:.1f}h ago"
    return f"{age_s/86400:.1f}d ago"


# --------------------------------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
