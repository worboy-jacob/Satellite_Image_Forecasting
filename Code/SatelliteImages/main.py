"""
Main entry point for the satellite imagery processing pipeline.

Controls the full workflow from grid generation through data download,
processing, repair, and fusion of Sentinel-2 and VIIRS imagery.
"""

import sys
import json
from pathlib import Path

# Add project root to path to enable imports regardless of execution location
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import Config
from src.utils.paths import (
    get_base_dir,
    find_shapefile,
    get_config_dir,
    get_processed_pairs_file,
    get_results_dir,
)
from src.grid.grid_generator import get_or_create_grid
from src.download.sentinel import download_sentinel_for_country_year
from src.download.viirs import download_viirs_for_country_year
from src.download.repair_failures import repair_country_year_failures
from src.download.repair_failures import scan_for_missing_data
from src.fusion.data_fusion import (
    combine_sentinel_viirs_data,
    scan_for_missing_data_combined,
)


def load_processed_pairs():
    """Load the set of processed country-year pairs from JSON file."""
    processed_pairs_file = get_processed_pairs_file()
    if processed_pairs_file.exists():
        with open(processed_pairs_file, "r") as f:
            return set(json.load(f))
    return set()


def save_processed_pair(country_name, year):
    """Save a processed country-year pair to the JSON file."""
    pair_key = f"{country_name}_{year}"
    processed_pairs = load_processed_pairs()
    processed_pairs.add(pair_key)

    with open(get_processed_pairs_file(), "w") as f:
        json.dump(list(processed_pairs), f)


def is_pair_processed(country_name, year):
    """Check if a country-year pair has been processed."""
    pair_key = f"{country_name}_{year}"
    return pair_key in load_processed_pairs()


def save_problematic_cells(country_name, year, sentinel_failures, viirs_failures):
    """Save problematic cells that couldn't be repaired to a JSON file."""
    combined_dir = get_results_dir() / "Images" / "Combined" / country_name / str(year)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Document unresolved failures for future reference or manual intervention
    problematic_cells = {
        "country": country_name,
        "year": year,
        "sentinel_failures": sentinel_failures,
        "viirs_failures": viirs_failures,
        "timestamp": str(Path(__file__).stat().st_mtime),
    }

    problematic_cells_file = combined_dir / "problematic_cells.json"
    with open(problematic_cells_file, "w") as f:
        json.dump(problematic_cells, f, indent=2)


def get_failed_cells(data_type, country_name, year):
    """Get list of cells that failed processing for a data type."""
    failures_dir = (
        get_results_dir()
        / "Images"
        / data_type.capitalize()
        / country_name
        / str(year)
        / "failures"
    )

    if not failures_dir.exists():
        return []

    failure_log_file = failures_dir / "failure_log.jsonl"
    if not failure_log_file.exists():
        return []

    failed_cells = set()
    with open(failure_log_file, "r") as f:
        for line in f:
            try:
                failure = json.loads(line.strip())
                cell_id = failure.get("cell_id")
                if cell_id is not None and cell_id != "global" and cell_id != "batch":
                    # Convert string cell_id to int if needed for consistent handling
                    if isinstance(cell_id, str) and cell_id.isdigit():
                        cell_id = int(cell_id)
                    if isinstance(cell_id, int):
                        failed_cells.add(cell_id)
            except json.JSONDecodeError:
                continue

    return list(failed_cells)


def process_country_year_pair(config, country_name, year, grid_gdf, logger):
    """Process a single country-year pair completely."""
    max_repair_attempts = 100

    # Skip if already processed
    if is_pair_processed(country_name, year):
        logger.info(f"Skipping already processed {country_name}, year {year}")
        return True

    logger.info(f"Processing {country_name}, year {year}")

    # === STEP 1: SENTINEL DATA ACQUISITION ===
    logger.info(f"Downloading Sentinel-2 data for {country_name}, year {year}")
    download_sentinel_for_country_year(
        config=config,
        country_name=country_name,
        year=year,
        grid_gdf=grid_gdf,
    )

    # === STEP 2: VIIRS DATA ACQUISITION ===
    logger.info(f"Downloading VIIRS data for {country_name}, year {year}")
    download_viirs_for_country_year(
        config=config,
        country_name=country_name,
        year=year,
        grid_gdf=grid_gdf,
    )

    # Step 3: Scan for missing data
    logger.info(
        f"Scanning for missing data in Sentinel files for {country_name}, year {year}"
    )
    scan_for_missing_data(data_type="sentinel", country_name=country_name, year=year)

    logger.info(
        f"Scanning for missing data in VIIRS files for {country_name}, year {year}"
    )
    scan_for_missing_data(data_type="viirs", country_name=country_name, year=year)

    # Step 4: Repair failures for Sentinel data
    logger.info(f"Repairing any failed Sentinel data for {country_name}, year {year}")
    sentinel_repair_results = repair_country_year_failures(
        data_type="sentinel",
        country_name=country_name,
        year=year,
        grid_gdf=grid_gdf,
        config=config,
        max_repair_attempts=max_repair_attempts,
    )

    # Step 5: Repair failures for VIIRS data
    logger.info(f"Repairing any failed VIIRS data for {country_name}, year {year}")
    viirs_repair_results = repair_country_year_failures(
        data_type="viirs",
        country_name=country_name,
        year=year,
        grid_gdf=grid_gdf,
        config=config,
        max_repair_attempts=max_repair_attempts,
    )

    # Get any remaining failed cells
    sentinel_failures = get_failed_cells("sentinel", country_name, year)
    viirs_failures = get_failed_cells("viirs", country_name, year)

    # If there are still failures after repair attempts, save them to a file and skip combination
    if sentinel_failures or viirs_failures:
        logger.warning(
            f"Some cells could not be repaired for {country_name}, year {year}"
        )
        logger.warning(
            f"Sentinel failures: {len(sentinel_failures)}, VIIRS failures: {len(viirs_failures)}"
        )
        save_problematic_cells(country_name, year, sentinel_failures, viirs_failures)
        logger.warning(
            f"Skipping data combination for {country_name}, year {year} due to unresolved failures"
        )
        return False

    # Step 6: Combine Sentinel and VIIRS data
    logger.info(f"Combining Sentinel and VIIRS data for {country_name}, year {year}")
    # Keep originals until we verify the combined data is valid
    combine_result = combine_sentinel_viirs_data(
        countries=[country_name],
        years=[year],
        max_workers=12,
        skip_existing=True,
        delete_originals=False,  # Keep originals for now
    )

    # Step 7: Scan combined data
    logger.info(f"Scanning combined data for {country_name}, year {year}")
    combined_scan_result = scan_for_missing_data_combined(
        max_workers=12, countries=[country_name], years=[year]
    )

    # Mark as processed
    save_processed_pair(country_name, year)
    logger.info(f"Marked {country_name}, year {year} as processed")

    return True


def main():
    """Main entry point for the satellite image processing pipeline."""
    try:
        # Load configuration
        config_path = get_config_dir() / "config.yaml"
        config = Config(config_path).config
        logger = setup_logging(config.get("log_level", "INFO"))
        logger.info("Starting satellite image processing pipeline")

        # Get countries from config
        countries = config.get("countries", [])
        logger.info(f"Processing {len(countries)} countries")

        # Verify shapefiles exist for each country
        for country in countries:
            country_name = country["name"]
            try:
                shapefile_path = find_shapefile(country_name)
                logger.info(f"Found shapefile for {country_name}: {shapefile_path}")
            except FileNotFoundError as e:
                logger.error(f"Error finding shapefile: {e}")
                raise

        # Generate grids for each country
        cell_size_km = config.get("cell_size_km", 5)
        min_area_percent = config.get("min_area_percent", 40.0)
        logger.info(
            f"Using grid cell size of {cell_size_km}km with minimum {min_area_percent}% area overlap"
        )

        # Process each country
        for country in countries:
            country_name = country["name"]
            country_crs = country["crs"]
            years = country.get("years", [])

            # Generate grid for this country
            logger.info(f"Generating grid for {country_name} using CRS {country_crs}")
            grid_gdf = get_or_create_grid(
                country_name=country_name,
                cell_size_km=cell_size_km,
                target_crs=country_crs,
                min_area_percent=min_area_percent,
            )
            logger.info(f"Generated grid with {len(grid_gdf)} cells for {country_name}")

            # Process each year for this country - one at a time
            for year in years:
                process_country_year_pair(config, country_name, year, grid_gdf, logger)

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
