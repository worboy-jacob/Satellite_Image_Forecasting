from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).parents[2]


def get_base_dir() -> Path:
    """Return the base directory of the project."""
    return BASE_DIR


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return BASE_DIR / "Code" / "LabelImagery" / "logs"


def get_data_dir() -> Path:
    """Return the data directory."""
    return BASE_DIR / "data"


def get_config_dir() -> Path:
    """Return the config directory."""
    return BASE_DIR / "Code" / "LabelImagery" / "config"


def get_results_dir() -> Path:
    """Return the results directory."""
    return get_data_dir() / "Results"


def get_grid_path() -> Path:
    """Return the grid directory."""
    return get_results_dir() / "SatelliteImageData" / "grids"


def get_wealthindex_dir() -> Path:
    """Return the wealth index with geometry directory."""
    return get_results_dir() / "GPSResults" / "WealthGPS"


def get_wealthindex_paths() -> Path:
    """Return the paths to the individual wealth index files."""
    return [
        get_wealthindex_dir() / "SN_2023_output.gpkg",
        get_wealthindex_dir() / "SN_2019_output.gpkg",
        get_wealthindex_dir() / "SN_2018_output.gpkg",
        get_wealthindex_dir() / "SN_2017_output.gpkg",
        get_wealthindex_dir() / "SN_2016_output.gpkg",
        get_wealthindex_dir() / "SN_2015_output.gpkg",
        get_wealthindex_dir() / "GH_2022_output.gpkg",
    ]


def get_output_dir() -> Path:
    return get_results_dir() / "LabelledGrids"
