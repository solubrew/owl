"""Owl tests"""
import pytest


def test_version():
    """Test version is set"""
    import owl
    assert owl.__version__ == "0.1.0"


def test_cli():
    """Test CLI works"""
    from owl import cli
    assert cli.main(["status"]) == 0
