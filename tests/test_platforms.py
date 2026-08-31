"""Tests for owl._platforms module."""
from __future__ import annotations

import pytest


class TestLoadPlatforms:
    def test_returns_dict(self) -> None:
        from owl._platforms import load_platforms

        result = load_platforms()
        assert isinstance(result, dict)

    def test_contains_sklearn(self) -> None:
        from owl._platforms import load_platforms

        result = load_platforms()
        assert "sklearn" in result

    def test_platforms_keys_consistent(self) -> None:
        from owl._platforms import load_platforms

        first = load_platforms()
        second = load_platforms()
        assert set(first.keys()) == set(second.keys())
