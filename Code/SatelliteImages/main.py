# scripts/main.py
import sys
import json
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import Config
from src.utils.paths import (
    get_base_dir,
    find_shapefile,
    get_config_dir,
    get_processed_pairs_file,
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

            # Process each year for this country
            for year in years:
                logger.info(
                    f"Processing Sentinel-2 data for {country_name}, year {year}"
                )
                download_sentinel_for_country_year(
                    config=config,
                    country_name=country_name,
                    year=year,
                    grid_gdf=grid_gdf,
                )
                download_viirs_for_country_year(
                    config=config,
                    country_name=country_name,
                    year=year,
                    grid_gdf=grid_gdf,
                )

        for country in countries:
            country_name = country["name"]
            years = country.get("years", [])
            for year in years:
                if is_pair_processed(country_name, year):
                    logger.info(
                        f"Skipping already processed {country_name}, year {year}"
                    )
                    continue
                # Scan for missing data in Sentinel files
                logger.info(
                    f"Scanning for missing data in Sentinel files for {country_name}, year {year}"
                )
                scan_for_missing_data(
                    data_type="sentinel", country_name=country_name, year=year
                )

                # Scan for missing data in VIIRS files
                logger.info(
                    f"Scanning for missing data in VIIRS files for {country_name}, year {year}"
                )
                scan_for_missing_data(
                    data_type="viirs", country_name=country_name, year=year
                )

                # Repair any failures for Sentinel data
                logger.info(
                    f"Repairing any failed Sentinel data for {country_name}, year {year}"
                )
                repair_country_year_failures(
                    data_type="sentinel",
                    country_name=country_name,
                    year=year,
                    grid_gdf=grid_gdf,
                    config=config,
                )

                # Repair any failures for VIIRS data
                logger.info(
                    f"Repairing any failed VIIRS data for {country_name}, year {year}"
                )
                repair_country_year_failures(
                    data_type="viirs",
                    country_name=country_name,
                    year=year,
                    grid_gdf=grid_gdf,
                    config=config,
                )

                save_processed_pair(country_name, year)
                logger.info(f"Marked {country_name}, year {year} as processed")

        logger.info("Combining data")
        combine_sentinel_viirs_data()
        logger.info("Scanning combined data")
        result = scan_for_missing_data_combined(max_workers=8)
        logger.info(f"Scan for missing data completed with result: {result}")

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
