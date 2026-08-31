"""Tests for owl package."""
import pytest
from owl import __version__, logger, main, get_version, initialize, configure_logging


def test_version() -> None:
    """Test version is defined."""
    assert __version__ == "0.1.0"


def test_main_returns_zero() -> None:
    """Test main function returns 0."""
    assert main() == 0


def test_logger_exists() -> None:
    """Test logger is configured."""
    assert logger is not None


def test_get_version() -> None:
    """Test get_version returns string."""
    v = get_version()
    assert isinstance(v, str)
    assert v == "0.1.0"


def test_initialize() -> None:
    """Test initialize returns True."""
    result = initialize()
    assert result is True


def test_initialize_with_config() -> None:
    """Test initialize with config path."""
    result = initialize(config_path="owl.yaml")
    assert result is True


def test_configure_logging() -> None:
    """Test configure_logging sets level."""
    configure_logging("DEBUG")
    assert logger.level == 10  # DEBUG = 10


def test_configure_logging_invalid() -> None:
    """Test configure_logging with invalid level."""
    configure_logging("INVALID")
    # Should not raise, just use default


# --------------------------------------------------------------------------------------------------- #
# Integration tests for owl.owl.Owl
# --------------------------------------------------------------------------------------------------- #


def test_owl_init():
    """Test Owl initializes with tag."""
    from owl.owl import Owl
    o = Owl("test_run")
    assert o.tag == "test_run"
    assert o.model is None
    assert o.active_model is None
    assert o.models == []


def test_owl_set_active_model():
    """Test setting active model configuration."""
    from owl.owl import Owl
    o = Owl("test")
    o.setActiveModel("algo12Z1", "a000", "sklearn")
    assert o.active_model == {"algoN": "algo12Z1", "seqN": "a000", "platN": "sklearn"}


def test_owl_add_and_set_models():
    """Test model list management."""
    from owl.owl import Owl
    o = Owl("test")
    o.addModel("algo12Z1", "a000", "sklearn")
    assert len(o.models) == 1
    o.setModels([["algo12Z1", "a001", "sklearn"]])
    assert len(o.models) == 1


def test_owl_train_predict_integration():
    """End-to-end: train sklearn LinearRegression and predict."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from pandas import DataFrame
    from owl.owl import Owl

    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ftrs_train = DataFrame(X_train, columns=[f"f{i}" for i in range(5)])
    targs_train = DataFrame(y_train)
    ftrs_test = DataFrame(X_test, columns=[f"f{i}" for i in range(5)])
    ids_test = list(range(len(y_test)))

    # Initialize and train
    o = Owl("test_run")
    o.setActiveModel("algo12Z1", "a000", "sklearn")
    o.train(ftrs_train, targs_train)

    assert o.model is not None, "Model should be trained"
    assert o.code is not None, "Code should be set"

    # Predict
    result = o.predict(
        cycl="test_cycl",
        ftrs=ftrs_test,
        targs=y_test,
        ids=ids_test,
        targn="binary_target",
        what="predict"
    )

    assert not result.empty, "Prediction result should not be empty"
    assert "Predicts" in result.columns, "Result should have Predicts column"
    assert len(result) == len(y_test), "Result length should match test set"
