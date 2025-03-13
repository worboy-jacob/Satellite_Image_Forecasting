# scripts/main.py
import sys
from pathlib import Path
import os

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import Config
from src.utils.paths import get_base_dir, find_shapefile, get_config_dir
from src.grid.grid_generator import get_or_create_grid
from src.download.sentinel import download_sentinel_for_country_year
from src.download.viirs import download_viirs_for_country_year
from src.download.repair_failures import repair_country_year_failures
from src.download.repair_failures import scan_for_missing_data
from src.fusion.data_fusion import combine_sentinel_viirs_data, DataIntegrityChecker


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
                # Scan for missing data in Sentinel files
                logger.info(
                    f"Scanning for missing data again in Sentinel files for {country_name}, year {year}"
                )
        # Run comprehensive data integrity check across all processed data
        logger.info("Running comprehensive data integrity check for all processed data")
        checker = DataIntegrityChecker(
            countries=[country["name"] for country in countries],
            years=list(
                set(year for country in countries for year in country.get("years", []))
            ),
            max_workers=config.get("max_workers", 4),
        )

        # Run the check and get results
        integrity_results = checker.check_all_data()

        # Log summary of integrity results
        sentinel_total = integrity_results["sentinel"]["total_cells"]
        sentinel_issues = (
            integrity_results["sentinel"]["cells_with_missing_bands"]
            + integrity_results["sentinel"]["cells_with_empty_arrays"]
            + integrity_results["sentinel"]["cells_with_errors"]
        )

        viirs_total = integrity_results["viirs"]["total_cells"]
        viirs_issues = (
            integrity_results["viirs"]["cells_with_missing_bands"]
            + integrity_results["viirs"]["cells_with_empty_arrays"]
            + integrity_results["viirs"]["cells_with_errors"]
        )

        matching_percent = 0
        if integrity_results["matching"]["total_cells"] > 0:
            matching_percent = (
                integrity_results["matching"]["matching_cells"]
                / integrity_results["matching"]["total_cells"]
                * 100
            )

        logger.info(f"Data integrity check complete:")
        logger.info(
            f"Sentinel: {sentinel_issues}/{sentinel_total} cells have issues ({sentinel_issues/max(1, sentinel_total)*100:.1f}%)"
        )
        logger.info(
            f"VIIRS: {viirs_issues}/{viirs_total} cells have issues ({viirs_issues/max(1, viirs_total)*100:.1f}%)"
        )
        logger.info(
            f"Matching: {integrity_results['matching']['matching_cells']}/{integrity_results['matching']['total_cells']} cells match ({matching_percent:.1f}%)"
        )
        logger.info(f"Detailed reports saved to {checker.output_dir}")
        # Combine Sentinel and VIIRS data after repairs and scans
        logger.info(
            f"Combining Sentinel and VIIRS data for {country_name}, year {year}"
        )

        # Log success
        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
