# Modified main.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
from label import load_grid_cells, load_wealth_data, calculate_coverage, apply_threshold
from logging_config import setup_logging
from paths import (
    get_config_dir,
    get_grid_path,
    get_wealthindex_paths,
    get_output_dir,
)
from pathlib import Path


def main():
    """Run the analysis using configuration from config.yaml."""
    logger = setup_logging("INFO")
    logger.info("Starting wealth coverage threshold analysis for multiple years...")
    config_path = get_config_dir() / "config.yaml"

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Extract configuration values
    threshold = config.get("threshold", 30)  # Default to 30% if not specified
    countries = config.get(
        "countries", ["SN", "GH"]
    )  # Default to both countries if not specified
    force_reprocess = config.get(
        "force_reprocess", False
    )  # Default to not reprocessing existing files

    logger.info(f"Using coverage threshold: {threshold}%")
    logger.info(f"Processing countries: {', '.join(countries)}")
    logger.info(f"Force reprocessing existing files: {force_reprocess}")

    grid_base_path = get_grid_path()
    wealth_gpkg_paths = get_wealthindex_paths()
    output_dir = get_output_dir()

    # Validate required configuration
    if not grid_base_path.exists():
        logger.error(f"Error: Grid directory {grid_base_path} does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            sys.exit(1)

    # Country code to full name mapping
    country_names = {"SN": "Senegal", "GH": "Ghana"}

    # Load grid cells for each country
    grid_gdfs = {}
    for country_code in countries:
        country_name = country_names.get(country_code)
        if not country_name:
            logger.warning(f"Unknown country code: {country_code}, skipping")
            continue

        grid_path = grid_base_path / f"{country_name}_grid_5km.gpkg"
        if not grid_path.exists():
            logger.warning(f"Grid file not found for {country_name}: {grid_path}")
            continue

        try:
            logger.info(f"Loading grid cells for {country_name}...")
            grid_gdf = load_grid_cells(grid_path)
            if country_name == "Senegal":
                grid_gdf.to_crs("EPSG:32628", inplace=True)
            elif country_name == "Ghana":
                grid_gdf.to_crs("EPSG:32630", inplace=True)
            logger.info(f"Grid for {country_name} loaded with crs: {grid_gdf.crs}")
            grid_gdfs[country_code] = grid_gdf
        except Exception as e:
            logger.error(f"Fatal error loading grid data for {country_name}: {e}")
            continue

    if not grid_gdfs:
        logger.error("No valid grid data loaded. Exiting.")
        sys.exit(1)

    # Filter wealth paths for selected countries
    filtered_wealth_paths = []
    for path in wealth_gpkg_paths:
        path_str = str(path)
        country_code = os.path.basename(path_str).split("_")[0]
        if country_code in countries:
            filtered_wealth_paths.append(path)

    if not filtered_wealth_paths:
        logger.error("No wealth data files found for selected countries. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(filtered_wealth_paths)} wealth data files for processing")

    # Get list of existing output files
    existing_files = [f.name for f in output_dir.glob("*_wealthindex_labelled.gpkg")]
    if existing_files and not force_reprocess:
        logger.info(f"Found {len(existing_files)} existing output files")

    # Process each wealth index file
    files_processed = 0
    files_skipped = 0

    for wealth_path in filtered_wealth_paths:
        try:
            # Extract country and year from filename
            filename = os.path.basename(str(wealth_path))
            parts = filename.split("_")
            country_code = parts[0]
            year = parts[1]

            if country_code not in grid_gdfs:
                logger.warning(
                    f"Skipping {filename} as grid data for {country_code} is not loaded"
                )
                continue

            country_name = country_names.get(country_code)

            # Check if output file already exists
            output_filename = f"{country_name}_{year}_wealthindex_labelled.gpkg"
            output_path = output_dir / output_filename

            if output_path.exists() and not force_reprocess:
                logger.info(
                    f"Skipping {country_name} {year} - output file already exists"
                )
                files_skipped += 1
                continue

            logger.info(f"Processing {country_name} {year} wealth data...")

            # Load wealth data
            wealth_gdf = load_wealth_data(wealth_path)

            # Apply the grid to wealth data calculation
            grid_gdf = grid_gdfs[country_code].copy()
            if country_code == "SN":
                wealth_gdf_new = wealth_gdf.to_crs("EPSG:32628")
                wealth_gdf = wealth_gdf_new
            elif country_code == "GH":
                wealth_gdf_new = wealth_gdf.to_crs("EPSG:32630")
                wealth_gdf = wealth_gdf_new

            logger.info(f"Calculating coverage for {country_name} {year}...")
            result_gdf = calculate_coverage(grid_gdf, wealth_gdf)

            # Apply threshold
            logger.info(
                f"Applying {threshold}% threshold to {country_name} {year} data..."
            )
            thresholded_gdf = apply_threshold(result_gdf, threshold)

            # Save output
            logger.info(f"Saving results to {output_path}")
            thresholded_gdf.to_file(output_path, driver="GPKG")
            logger.info(f"Successfully processed {country_name} {year} data")
            files_processed += 1

        except Exception as e:
            logger.error(f"Error processing wealth data for {wealth_path}: {e}")
            continue

    logger.info(
        f"Analysis complete! Processed {files_processed} files, skipped {files_skipped} existing files"
    )


if __name__ == "__main__":
    main()
