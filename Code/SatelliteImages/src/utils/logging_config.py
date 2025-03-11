import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.utils.paths import get_logs_dir


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure basic logging with rotating file handler and console handler."""
    logger = logging.getLogger("image_processing")
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set up rotating file handler
    log_path = get_logs_dir()
    log_path.mkdir(parents=True, exist_ok=True)
    # 5 MB per file, keep 3 backup files
    file_handler = RotatingFileHandler(
        log_path / "wealth_index.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    logger.handlers[0].flush = sys.stdout.flush
    logger.handlers[1].flush = sys.stdout.flush

    return logger
