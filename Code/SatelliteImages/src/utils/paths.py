from pathlib import Path
import os

# Get the base directory of the project
BASE_DIR = Path(__file__).parents[4]  # Go up 3 levels from this file


def get_base_dir() -> Path:
    """Return the base directory of the project."""
    return BASE_DIR


def get_SatelliteImage_dir() -> Path:
    return get_base_dir() / "Code" / "SatelliteImages"


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_SatelliteImage_dir() / "logs"


def get_data_dir() -> Path:
    """Return the data directory."""
    return BASE_DIR / "data"


def get_config_dir() -> Path:
    """Return the config directory."""
    return get_SatelliteImage_dir() / "config"


def get_shapefiles_dir() -> Path:
    """Return the shapefiles directory."""
    return get_data_dir() / "ShapeFiles"


def get_country_shapefile(country_name: str) -> Path:
    """Return the path to a country's shapefile directory."""
    return get_shapefiles_dir() / country_name


def get_results_dir() -> Path:
    """Return the results directory."""
    return get_data_dir() / "results" / "SatelliteImageData"


def get_country_year_dir(country_name: str, year: int) -> Path:
    """Return the directory for a specific country-year pair."""
    dir_path = get_results_dir() / f"{country_name}_{year}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def find_shapefile(country_name: str) -> Path:
    """Find the .shp file for a given country."""
    shapefile_dir = get_country_shapefile(country_name)

    # Find the .shp file
    shp_files = list(shapefile_dir.glob("*.shp"))
    if len(shp_files) > 1:
        # If multiple .shp files, log a warning and use the first one
        print(
            f"Warning: Multiple .shp files found in {shapefile_dir}. Using {shp_files[0].name}"
        )

    return shp_files[0]


def get_processed_pairs_file() -> Path:
    return get_results_dir() / "processed_country_year_pairs.json"
