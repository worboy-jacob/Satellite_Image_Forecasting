# src/utils/helpers.py
import logging
from pathlib import Path
import yaml
import subprocess
from datetime import datetime


###TODO: Clean this up like the WealthIndex logger to stop creating multiple files
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gps_processing_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load and validate configuration from YAML file.
    Looks for config file relative to the src directory.
    """
    try:
        # Get the path to the src directory
        src_dir = Path(__file__).parent.parent
        # Go up one level to the root directory containing config/
        root_dir = src_dir.parent
        # Construct absolute path to config
        config_file = root_dir / config_path

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_file}. "
                "Please ensure config/config.yaml exists in the project root."
            )

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        validate_config(config)
        return config

    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {str(e)}")


def validate_config(config: dict) -> None:
    """Validate configuration structure and required fields."""
    required_sections = ["paths", "processing", "visualization"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")


def get_git_root() -> Path:
    """Get the Git repository root directory."""
    try:
        root_dir = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("utf-8")
        )
        return Path(root_dir)
    except subprocess.CalledProcessError:
        raise EnvironmentError("Not a Git repository")


def ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if necessary."""
    path.mkdir(parents=True, exist_ok=True)
