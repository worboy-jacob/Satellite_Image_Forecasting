# main.py
from src.utils import setup_logging, load_config, get_git_root
from src.gps_processor import GPSDataProcessor
from src.grid_processor import GridProcessor
from src.visualization import WealthMapVisualizer
from pathlib import Path
import logging
from typing import Dict, Any
import time


def construct_paths(config: Dict[Any, Any], git_root: Path) -> Dict[str, Path]:
    """
    Construct all necessary file paths from configuration.

    Args:
        config: Configuration dictionary
        git_root: Git repository root path

    Returns:
        Dictionary containing all constructed paths
    """
    paths = {}

    # Construct GPS data path
    paths["gps_data"] = (
        git_root
        / config["paths"]["data_subfolder"]
        / config["paths"]["gps_folder"]
        / config["paths"]["gps_file"]
    )

    # Construct wealth index path
    paths["wealth_index"] = git_root / config["paths"]["wealth_index_file"]

    # Construct shapefile path
    paths["shapefile"] = (
        git_root
        / config["paths"]["data_subfolder"]
        / config["paths"]["shapefile_folder"]
        / config["paths"]["shapefile"]
    )

    # Construct output path
    paths["output"] = (
        git_root / config["paths"]["output_folder"] / config["paths"]["output_file"]
    )

    # Ensure output directory exists
    paths["output"].parent.mkdir(parents=True, exist_ok=True)

    return paths


def validate_paths(paths: Dict[str, Path]) -> None:
    """
    Validate that all input paths exist.

    Args:
        paths: Dictionary of paths to validate

    Raises:
        FileNotFoundError: If any required input file is missing
    """
    required_inputs = ["gps_data", "wealth_index", "shapefile"]
    for key in required_inputs:
        if not paths[key].exists():
            raise FileNotFoundError(f"Required input file not found: {paths[key]}")


def main() -> None:
    """
    Main execution function with proper error handling and logging.
    """
    start_time = time.time()

    # Setup logging and load configuration
    logger = setup_logging()
    logger.info("Starting GPS data processing")

    try:
        # Load configuration and construct paths
        config = load_config()
        git_root = get_git_root()
        paths = construct_paths(config, git_root)

        # Validate input paths
        validate_paths(paths)

        # Initialize processors
        logger.info("Initializing processors")
        gps_processor = GPSDataProcessor(config)
        grid_processor = GridProcessor(config)
        visualizer = WealthMapVisualizer(config)

        # Process GPS data
        logger.info("Loading and processing GPS data")
        gps_data = gps_processor.load_gps_data(paths["gps_data"])
        wealth_data = gps_processor.load_wealth_index(paths["wealth_index"])

        # Load shapefile
        logger.info("Loading shapefile data")
        shapefile_data = gps_processor.load_gps_data(paths["shapefile"])

        # Process grid
        logger.info("Creating and processing grid")
        result = grid_processor.create_grid(shapefile_data, wealth_data)

        # Save results
        logger.info(f"Saving results to {paths['output']}")
        result.to_file(paths["output"], driver="GPKG")

        # Create visualization
        logger.info("Generating visualization")
        visualizer.create_heatmap(result, shapefile_data)

        execution_time = time.time() - start_time
        logger.info(
            f"Processing completed successfully in {execution_time:.2f} seconds"
        )

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        logger.info("Process finished")


if __name__ == "__main__":
    main()
