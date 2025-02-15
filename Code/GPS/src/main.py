###TODO: better commenting and decorators

from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import time

from data.loader import DataLoader
from data.paths import PathManager
from gps_processing.processor import GPSDataProcessor
from gps_processing.grid import GridProcessor
from visualization.map_creator import WealthMapVisualizer
from utils.helpers import setup_logging, load_config
from utils.constants import COUNTRY_CRS


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def process_country_year(
    country_code: str,
    year: int,
    processors: Dict[str, Any],
    paths: PathManager,
    data_loader: DataLoader,
    vmin: float,
    vmax: float,
) -> None:
    """
    Process single country-year combination.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {country_code} - {year}")

    try:
        # Load required data
        gps_data = data_loader.load_gps_data(paths.find_gps_data(country_code, year))
        wealth_data = data_loader.filter_wealth_data(country_code, year)
        boundary = data_loader.load_gps_data(paths.find_shapefile(country_code))

        # Process GPS data
        processed_gps = processors["gps"].process_gps_data(
            gps_data, wealth_data, COUNTRY_CRS[country_code]
        )

        # Create wealth grid
        grid_data = processors["grid"].create_grid(boundary, processed_gps)

        # Create visualization
        output_path = paths.create_output_filename(country_code, year)
        processors["viz"].create_wealth_map(
            grid_data,
            boundary,
            output_path,
            f"Wealth Distribution - {country_code} {year}",
            vmin=vmin,
            vmax=vmax,
        )

    except Exception as e:
        logger.error(f"Error processing {country_code}-{year}: {e}")
        raise


def main() -> None:
    """
    Main execution function.
    """
    start_time = time.time()
    logger = setup_logging()

    try:
        # Initialize
        project_root = get_project_root()
        config = load_config()

        # Initialize path manager with project root
        paths = PathManager(config, project_root)
        data_loader = DataLoader(config)

        # Initialize processors
        processors = {
            "gps": GPSDataProcessor(config),
            "grid": GridProcessor(config),
            "viz": WealthMapVisualizer(config),
        }

        # Load wealth index data
        data_loader.load_wealth_index(paths.find_wealth_index())

        # Get available data points
        data_points = data_loader.get_available_data_points()

        if not data_points:
            logger.error("No valid data points found")
            return

        # Calculate global wealth range for consistent visualization
        wealth_ranges = []
        for country_code, year in data_points:
            wealth_data = data_loader.filter_wealth_data(country_code, year)
            wealth_ranges.extend(wealth_data["wealth_index"])

        vmin, vmax = min(wealth_ranges), max(wealth_ranges)

        # Process each country-year combination
        for country_code, year in data_points:
            process_country_year(
                country_code, year, processors, paths, data_loader, vmin, vmax
            )

        execution_time = time.time() - start_time
        logger.info(
            f"Processing completed successfully in {execution_time:.2f} seconds"
        )

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
