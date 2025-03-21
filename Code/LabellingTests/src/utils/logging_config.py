import logging
import sys
from logging.handlers import RotatingFileHandler
from paths import get_logs_dir


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure basic logging with rotating file handler and console handler."""
    # Convert string log level to actual level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Clear existing handlers from root logger to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Get the image_processing logger and clear any existing handlers
    logger = logging.getLogger("labelling_test")
    logger.handlers = []  # Clear any existing handlers
    logger.setLevel(numeric_level)
    logger.propagate = False  # Prevent propagation to root logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set up rotating file handler
    log_path = get_logs_dir()
    log_path.mkdir(parents=True, exist_ok=True)
    # 5 MB per file, keep 3 backup files
    file_handler = RotatingFileHandler(
        log_path / "label_tests.log",
        maxBytes=5 * 1024 * 1024,  # 100 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    # Create a custom flush method for the file handler
    original_file_flush = file_handler.flush

    def file_flush_immediately(*args, **kwargs):
        original_file_flush(*args, **kwargs)
        # No need to flush stdout for file handler

    file_handler.flush = file_flush_immediately
    logger.addHandler(file_handler)

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    # Create a custom flush method for the console handler
    original_console_flush = console_handler.flush

    def console_flush_immediately(*args, **kwargs):
        original_console_flush(*args, **kwargs)
        sys.stdout.flush()  # Force flush stdout

    console_handler.flush = console_flush_immediately
    logger.addHandler(console_handler)

    return logger
