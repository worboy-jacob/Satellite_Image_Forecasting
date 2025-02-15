# src/utils/paths.py
from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_dir() -> Path:
    """Returns the data directory."""
    return get_project_root() / "data"


def get_results_dir() -> Path:
    """Returns the results directory."""
    return get_data_dir() / "Results" / "WealthIndex"


def get_dhs_dir() -> Path:
    """Returns the DHS data directory."""
    return get_data_dir() / "DHS"


def get_logs_dir() -> Path:
    """Returns the logs directory."""
    return get_WealthIndex_dir() / "logs"


def get_Code_dir() -> Path:
    return get_project_root() / "Code"


def get_WealthIndex_dir() -> Path:
    return get_Code_dir() / "WealthIndex"


def get_configs_dir() -> Path:
    return get_WealthIndex_dir() / "config"
