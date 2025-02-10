# src/utils.py
import logging
import subprocess
import yaml
from pathlib import Path


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/gps_processing.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_git_root():
    """Get the Git repository root directory."""
    try:
        root_dir = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
            .strip()
            .decode("utf-8")
        )
        return Path(root_dir)
    except subprocess.CalledProcessError:
        raise Exception("This is not a Git repository")
