import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure basic logging."""
    logger = logging.getLogger("wealth_index")
    logger.setLevel(log_level)

    # Console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
