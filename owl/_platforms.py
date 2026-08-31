# @@@@@@@@@@@@@@@@@@Pheonix.Molecules.Logicizer.NN@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
    DOCid:
    name: Platforms Module
    description: >
        Load and manage ML platform integrations (sklearn, tsfresh, etc.)
    version: 0.1.0
    authority: filesystem
    security: sec|lvl2
    <(WT)>: -32
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================||
from __future__ import annotations

import logging
from os.path import dirname, join

logger = logging.getLogger(__name__)

here = join(dirname(__file__), '')

# ===============================================================================||


class Platform:
    """Base class for ML platforms."""
    
    def __init__(self, name: str) -> None:
        self.name = name
    
    def selectPlatform(self) -> str:
        """Return the selected platform name."""
        return self.name


def load_platforms() -> dict[str, Platform]:
    """Load available ML platforms.
    
    Returns:
        Dictionary of platform name -> Platform instance
    """
    platforms: dict[str, Platform] = {}
    
    # sklearn platform
    try:
        from sklearn.linear_model import LinearRegression
        platforms['sklearn'] = Platform('sklearn')
        logger.info("Loaded sklearn platform")
    except ImportError as e:
        logger.warning("sklearn platform not available: %s", e)
    
    # tsfresh platform
    try:
        from tsfresh import extract_features
        platforms['tsfresh'] = Platform('tsfresh')
        logger.info("Loaded tsfresh platform")
    except ImportError as e:
        logger.warning("tsfresh platform not available: %s", e)
    
    return platforms


# ===============================================================================||
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
