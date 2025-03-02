# wealth_mapping/utils/paths.py
from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_dir() -> Path:
    """Returns the data directory."""
    return get_project_root() / "data"


def get_results_dir() -> Path:
    """Returns the results directory."""
    return get_data_dir() / "Results"


def get_gps_dir() -> Path:
    """Returns the GPS data directory."""
    return get_data_dir() / "GPS"


def get_shapefiles_dir() -> Path:
    """Returns the shapefiles directory."""
    return get_data_dir() / "ShapeFiles"


def get_logs_dir() -> Path:
    """Returns the logs directory."""
    return get_WealthMapping_dir() / "logs"


def get_Code_dir() -> Path:
    return get_project_root() / "Code"


def get_WealthMapping_dir() -> Path:
    return get_Code_dir() / "WealthMapping"


def get_config_path() -> Path:
    return get_WealthMapping_dir() / "config"


def get_wealthindex_dir() -> Path:
    return get_results_dir() / "WealthIndex"
