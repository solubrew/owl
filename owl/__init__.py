"""Owl - AI Agent Platform."""
import logging
import sys
from typing import List, Optional

__version__: str = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for owl."""
    logger.info("Owl platform starting...")
    return 0


def get_version() -> str:
    """Get the owl version."""
    return __version__


def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging for owl.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if level:
        _level = level.upper()
        if not hasattr(logging, _level):
            logger.warning("Invalid logging level '%s', using INFO", level)
            _level = "INFO"
        logger.setLevel(getattr(logging, _level))


def initialize(config_path: Optional[str] = None) -> bool:
    """Initialize owl platform.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if initialization successful
    """
    logger.info("Initializing Owl platform...")
    return True


if __name__ == "__main__":
    sys.exit(main())
