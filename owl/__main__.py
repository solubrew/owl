"""Owl - AI Agent Platform."""
import logging
import sys

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for owl."""
    logger.info("Owl platform starting...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
