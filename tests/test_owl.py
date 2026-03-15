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
