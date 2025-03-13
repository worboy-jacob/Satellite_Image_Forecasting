import numpy as np
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Set
import json
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import pandas as pd

from src.utils.paths import get_results_dir

logger = logging.getLogger("data_integrity")


class DataIntegrityChecker:
    """Check data integrity for Sentinel and VIIRS cells."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sentinel_expected_bands: Optional[List[str]] = None,
        viirs_expected_bands: Optional[List[str]] = None,
        max_workers: int = 4,
    ):
        """
        Initialize the data integrity checker.

        Args:
            output_dir: Directory to save reports (defaults to results/Images/Reports)
            countries: List of countries to process (defaults to all available)
            years: List of years to process (defaults to all available)
            sentinel_expected_bands: Expected Sentinel bands/indices
            viirs_expected_bands: Expected VIIRS bands/indices
            max_workers: Maximum number of parallel workers
        """
        # Set default output directory
        if output_dir is None:
            output_dir = get_results_dir() / "Reports"
        self.output_dir = output_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set countries and years
        self.countries = countries
        self.years = years

        # Set default expected bands
        if sentinel_expected_bands is None:
            self.sentinel_expected_bands = [
                "rgb",
                "nir",
                "swir1",
                "swir2",
                "ndvi",
                "built_up",
            ]
        else:
            self.sentinel_expected_bands = sentinel_expected_bands

        if viirs_expected_bands is None:
            self.viirs_expected_bands = ["nightlights", "gradient"]
        else:
            self.viirs_expected_bands = viirs_expected_bands

        # Set max workers
        self.max_workers = max_workers

        # Base directories
        self.sentinel_base_dir = get_results_dir() / "Images" / "Sentinel"
        self.viirs_base_dir = get_results_dir() / "Images" / "VIIRS"

        # Initialize results
        self.results = {
            "sentinel": {
                "total_cells": 0,
                "cells_with_missing_bands": 0,
                "cells_with_empty_arrays": 0,
                "cells_with_errors": 0,
                "missing_bands_count": {},
                "cells_details": [],
            },
            "viirs": {
                "total_cells": 0,
                "cells_with_missing_bands": 0,
                "cells_with_empty_arrays": 0,
                "cells_with_errors": 0,
                "missing_bands_count": {},
                "cells_details": [],
            },
            "matching": {
                "countries_with_data": 0,
                "country_years_with_data": 0,
                "sentinel_only_cells": 0,
                "viirs_only_cells": 0,
                "matching_cells": 0,
                "cells_details": [],
            },
        }

    def check_all_data(self) -> Dict:
        """
        Check data integrity for all specified countries and years.

        Returns:
            Dictionary with check results
        """
        logger.info("Starting data integrity check")

        # Determine available countries if not specified
        if self.countries is None:
            self.countries = set()
            # Find all countries with either Sentinel or VIIRS data
            for data_type, base_dir in [
                ("sentinel", self.sentinel_base_dir),
                ("viirs", self.viirs_base_dir),
            ]:
                for country_dir in base_dir.iterdir():
                    if country_dir.is_dir():
                        self.countries.add(country_dir.name)
            self.countries = sorted(list(self.countries))

        logger.info(f"Found {len(self.countries)} countries to check")

        # Process each country
        for country in self.countries:
            self._check_country(country)

        # Generate summary reports
        self._generate_reports()

        return self.results

    def _check_country(self, country: str) -> None:
        """
        Check data integrity for a specific country.

        Args:
            country: Country name
        """
        logger.info(f"Checking data for country: {country}")

        # Determine available years if not specified
        country_years = self.years
        if country_years is None:
            country_years = set()
            # Find years with either Sentinel or VIIRS data
            for data_type, base_dir in [
                ("sentinel", self.sentinel_base_dir),
                ("viirs", self.viirs_base_dir),
            ]:
                country_dir = base_dir / country
                if country_dir.exists():
                    for year_dir in country_dir.iterdir():
                        if year_dir.is_dir() and year_dir.name.isdigit():
                            country_years.add(int(year_dir.name))
            country_years = sorted(list(country_years))

        if not country_years:
            logger.warning(f"No years found for country {country}")
            return

        # Check if country has any data
        sentinel_country_dir = self.sentinel_base_dir / country
        viirs_country_dir = self.viirs_base_dir / country

        if sentinel_country_dir.exists() or viirs_country_dir.exists():
            self.results["matching"]["countries_with_data"] += 1

        # Process each year
        for year in country_years:
            self._check_country_year(country, year)

    def _check_country_year(self, country: str, year: int) -> None:
        """
        Check data integrity for a specific country-year pair.

        Args:
            country: Country name
            year: Year to check
        """
        logger.info(f"Checking {country}, year {year}")

        # Get directories for this country-year
        sentinel_year_dir = self.sentinel_base_dir / country / str(year)
        viirs_year_dir = self.viirs_base_dir / country / str(year)

        # Check if both data types exist for this country-year
        has_sentinel = sentinel_year_dir.exists()
        has_viirs = viirs_year_dir.exists()

        if has_sentinel or has_viirs:
            self.results["matching"]["country_years_with_data"] += 1

        # Collect cell IDs from both sources
        sentinel_cells = set()
        viirs_cells = set()

        # Get Sentinel cells
        if has_sentinel:
            for cell_dir in sentinel_year_dir.glob("cell_*"):
                if cell_dir.is_dir() and (cell_dir / "processed_data.npz").exists():
                    cell_id = cell_dir.name.replace("cell_", "")
                    sentinel_cells.add(cell_id)

        # Get VIIRS cells
        if has_viirs:
            for cell_dir in viirs_year_dir.glob("cell_*"):
                if cell_dir.is_dir() and (cell_dir / "processed_data.npz").exists():
                    cell_id = cell_dir.name.replace("cell_", "")
                    viirs_cells.add(cell_id)

        # Find matching and non-matching cells
        matching_cells = sentinel_cells.intersection(viirs_cells)
        sentinel_only = sentinel_cells - matching_cells
        viirs_only = viirs_cells - matching_cells

        # Update counts
        self.results["matching"]["sentinel_only_cells"] += len(sentinel_only)
        self.results["matching"]["viirs_only_cells"] += len(viirs_only)
        self.results["matching"]["matching_cells"] += len(matching_cells)

        # Log cell matching details
        for cell_id in sentinel_only:
            self.results["matching"]["cells_details"].append(
                {
                    "country": country,
                    "year": year,
                    "cell_id": cell_id,
                    "status": "sentinel_only",
                }
            )

        for cell_id in viirs_only:
            self.results["matching"]["cells_details"].append(
                {
                    "country": country,
                    "year": year,
                    "cell_id": cell_id,
                    "status": "viirs_only",
                }
            )

        for cell_id in matching_cells:
            self.results["matching"]["cells_details"].append(
                {
                    "country": country,
                    "year": year,
                    "cell_id": cell_id,
                    "status": "matching",
                }
            )

        # Create tasks for parallel processing
        sentinel_tasks = []
        viirs_tasks = []

        # Create Sentinel tasks
        for cell_id in sentinel_cells:
            cell_dir = sentinel_year_dir / f"cell_{cell_id}"
            data_file = cell_dir / "processed_data.npz"
            if data_file.exists():
                sentinel_tasks.append(
                    (
                        "sentinel",
                        country,
                        year,
                        cell_id,
                        data_file,
                        self.sentinel_expected_bands,
                    )
                )

        # Create VIIRS tasks
        for cell_id in viirs_cells:
            cell_dir = viirs_year_dir / f"cell_{cell_id}"
            data_file = cell_dir / "processed_data.npz"
            if data_file.exists():
                viirs_tasks.append(
                    (
                        "viirs",
                        country,
                        year,
                        cell_id,
                        data_file,
                        self.viirs_expected_bands,
                    )
                )

        # Process tasks in parallel
        all_tasks = sentinel_tasks + viirs_tasks

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._check_single_cell,
                    data_type,
                    country,
                    year,
                    cell_id,
                    data_file,
                    expected_bands,
                ): (data_type, cell_id)
                for data_type, country, year, cell_id, data_file, expected_bands in all_tasks
            }

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Checking {country} {year}",
            ):
                data_type, cell_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error checking {data_type} cell {cell_id}: {e}")
                    self.results[data_type]["cells_with_errors"] += 1

    def _check_single_cell(
        self,
        data_type: str,
        country: str,
        year: int,
        cell_id: str,
        data_file: Path,
        expected_bands: List[str],
    ) -> Dict:
        """
        Check a single cell's data integrity.

        Args:
            data_type: Data type ('sentinel' or 'viirs')
            country: Country name
            year: Year
            cell_id: Cell ID
            data_file: Path to the data file
            expected_bands: List of expected bands

        Returns:
            Dictionary with check results
        """
        result = {
            "country": country,
            "year": year,
            "cell_id": cell_id,
            "data_type": data_type,
            "file": str(data_file),
            "missing_bands": [],
            "empty_arrays": [],
            "available_bands": [],
            "has_issues": False,
        }

        try:
            # Count this cell
            self.results[data_type]["total_cells"] += 1

            # Load data file
            with np.load(data_file) as data:
                available_bands = list(data.files)
                result["available_bands"] = available_bands

                # Check for missing bands
                missing_bands = [
                    band for band in expected_bands if band not in available_bands
                ]
                result["missing_bands"] = missing_bands

                if missing_bands:
                    self.results[data_type]["cells_with_missing_bands"] += 1
                    result["has_issues"] = True

                    # Update missing bands count
                    for band in missing_bands:
                        if band not in self.results[data_type]["missing_bands_count"]:
                            self.results[data_type]["missing_bands_count"][band] = 0
                        self.results[data_type]["missing_bands_count"][band] += 1

                # Check for empty arrays
                empty_arrays = []
                for band in available_bands:
                    array = data[band]
                    if array.size == 0 or np.all(array == 0) or np.all(np.isnan(array)):
                        empty_arrays.append(band)

                result["empty_arrays"] = empty_arrays

                if empty_arrays:
                    self.results[data_type]["cells_with_empty_arrays"] += 1
                    result["has_issues"] = True

            # Add result to cell details if there are issues
            if result["has_issues"]:
                self.results[data_type]["cells_details"].append(result)

            return result

        except Exception as e:
            logger.error(f"Error checking {data_type} cell {cell_id}: {e}")
            self.results[data_type]["cells_with_errors"] += 1

            result["error"] = str(e)
            result["has_issues"] = True
            self.results[data_type]["cells_details"].append(result)

            return result

    def _generate_reports(self) -> None:
        """Generate comprehensive reports from the check results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base_name = f"data_integrity_check_{timestamp}"

        # Save full JSON report
        json_report_file = self.output_dir / f"{report_base_name}.json"
        with open(json_report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate summary report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "sentinel": {
                "total_cells": self.results["sentinel"]["total_cells"],
                "cells_with_issues": (
                    self.results["sentinel"]["cells_with_missing_bands"]
                    + self.results["sentinel"]["cells_with_empty_arrays"]
                    + self.results["sentinel"]["cells_with_errors"]
                ),
                "issue_percentage": 0,
                "missing_bands_summary": self.results["sentinel"][
                    "missing_bands_count"
                ],
            },
            "viirs": {
                "total_cells": self.results["viirs"]["total_cells"],
                "cells_with_issues": (
                    self.results["viirs"]["cells_with_missing_bands"]
                    + self.results["viirs"]["cells_with_empty_arrays"]
                    + self.results["viirs"]["cells_with_errors"]
                ),
                "issue_percentage": 0,
                "missing_bands_summary": self.results["viirs"]["missing_bands_count"],
            },
            "matching": {
                "countries_with_data": self.results["matching"]["countries_with_data"],
                "country_years_with_data": self.results["matching"][
                    "country_years_with_data"
                ],
                "sentinel_only_cells": self.results["matching"]["sentinel_only_cells"],
                "viirs_only_cells": self.results["matching"]["viirs_only_cells"],
                "matching_cells": self.results["matching"]["matching_cells"],
                "total_cells": (
                    self.results["matching"]["sentinel_only_cells"]
                    + self.results["matching"]["viirs_only_cells"]
                    + self.results["matching"]["matching_cells"]
                ),
                "matching_percentage": 0,
            },
        }

        # Calculate percentages
        if summary["sentinel"]["total_cells"] > 0:
            summary["sentinel"]["issue_percentage"] = (
                summary["sentinel"]["cells_with_issues"]
                / summary["sentinel"]["total_cells"]
                * 100
            )

        if summary["viirs"]["total_cells"] > 0:
            summary["viirs"]["issue_percentage"] = (
                summary["viirs"]["cells_with_issues"]
                / summary["viirs"]["total_cells"]
                * 100
            )

        if summary["matching"]["total_cells"] > 0:
            summary["matching"]["matching_percentage"] = (
                summary["matching"]["matching_cells"]
                / summary["matching"]["total_cells"]
                * 100
            )

        # Save summary report
        summary_file = self.output_dir / f"{report_base_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Create CSV reports for easier analysis

        # Sentinel cells with issues
        if self.results["sentinel"]["cells_details"]:
            df_sentinel = pd.DataFrame(self.results["sentinel"]["cells_details"])
            sentinel_csv = self.output_dir / f"{report_base_name}_sentinel_issues.csv"
            df_sentinel.to_csv(sentinel_csv, index=False)

        # VIIRS cells with issues
        if self.results["viirs"]["cells_details"]:
            df_viirs = pd.DataFrame(self.results["viirs"]["cells_details"])
            viirs_csv = self.output_dir / f"{report_base_name}_viirs_issues.csv"
            df_viirs.to_csv(viirs_csv, index=False)

        # Cell matching status
        if self.results["matching"]["cells_details"]:
            df_matching = pd.DataFrame(self.results["matching"]["cells_details"])
            matching_csv = self.output_dir / f"{report_base_name}_matching_status.csv"
            df_matching.to_csv(matching_csv, index=False)

        logger.info(f"Reports generated and saved to {self.output_dir}")
        logger.info(
            f"Summary: Sentinel issues: {summary['sentinel']['issue_percentage']:.1f}%, VIIRS issues: {summary['viirs']['issue_percentage']:.1f}%"
        )
        logger.info(
            f"Matching cells: {summary['matching']['matching_percentage']:.1f}% ({summary['matching']['matching_cells']} of {summary['matching']['total_cells']})"
        )


def combine_sentinel_viirs_data(
    output_dir: Optional[Path] = None,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    max_workers: int = 4,
    skip_existing: bool = True,
    delete_originals: bool = True,
) -> Dict[str, int]:
    """
    Combine Sentinel and VIIRS data for all cells.

    Args:
        output_dir: Directory to save combined data (defaults to results/Images/Combined)
        countries: List of countries to process (defaults to all available)
        years: List of years to process (defaults to all available)
        max_workers: Maximum number of parallel workers
        skip_existing: Skip cells that already have combined data
        delete_originals: Delete original npz files after successful combination

    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger("data_combiner")

    # Set default output directory
    if output_dir is None:
        output_dir = get_results_dir() / "Images" / "Combined"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base directories
    sentinel_base_dir = get_results_dir() / "Images" / "Sentinel"
    viirs_base_dir = get_results_dir() / "Images" / "VIIRS"

    # Determine available countries if not specified
    if countries is None:
        countries = set()
        # Find all countries with both Sentinel and VIIRS data
        for country_dir in sentinel_base_dir.iterdir():
            if country_dir.is_dir() and (viirs_base_dir / country_dir.name).exists():
                countries.add(country_dir.name)
        countries = sorted(list(countries))

    logger.info(f"Found {len(countries)} countries to process")

    # Initialize statistics
    stats = {
        "total_cells_processed": 0,
        "cells_with_missing_sentinel": 0,
        "cells_with_missing_viirs": 0,
        "cells_with_errors": 0,
        "cells_skipped": 0,
        "cells_successfully_combined": 0,
        "originals_deleted": 0,
    }

    # Process each country
    for country in countries:
        logger.info(f"Processing country: {country}")

        # Determine available years if not specified
        country_years = years
        if country_years is None:
            country_years = set()
            # Find years with both Sentinel and VIIRS data
            sentinel_country_dir = sentinel_base_dir / country
            viirs_country_dir = viirs_base_dir / country

            if not sentinel_country_dir.exists() or not viirs_country_dir.exists():
                logger.warning(
                    f"Missing data directories for country {country}, skipping"
                )
                continue

            for year_dir in sentinel_country_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    if (viirs_country_dir / year_dir.name).exists():
                        country_years.add(int(year_dir.name))
            country_years = sorted(list(country_years))

        if not country_years:
            logger.warning(f"No years found for country {country}")
            continue

        # Process each year
        for year in country_years:
            logger.info(f"Processing {country}, year {year}")

            # Get directories for this country-year
            sentinel_year_dir = sentinel_base_dir / country / str(year)
            viirs_year_dir = viirs_base_dir / country / str(year)
            combined_year_dir = output_dir / country / str(year)

            # Check if directories exist
            if not sentinel_year_dir.exists() or not viirs_year_dir.exists():
                logger.warning(f"Missing data directory for {country} {year}, skipping")
                continue

            # Create output directory
            combined_year_dir.mkdir(parents=True, exist_ok=True)

            # Get list of cell IDs with Sentinel data
            sentinel_cells = {}
            for cell_dir in sentinel_year_dir.glob("cell_*"):
                if cell_dir.is_dir():
                    data_file = cell_dir / "processed_data.npz"
                    if data_file.exists():
                        cell_id = cell_dir.name.replace("cell_", "")
                        sentinel_cells[cell_id] = data_file

            # Get list of cell IDs with VIIRS data
            viirs_cells = {}
            for cell_dir in viirs_year_dir.glob("cell_*"):
                if cell_dir.is_dir():
                    data_file = cell_dir / "processed_data.npz"
                    if data_file.exists():
                        cell_id = cell_dir.name.replace("cell_", "")
                        viirs_cells[cell_id] = data_file

            # Find common cells
            common_cell_ids = set(sentinel_cells.keys()).intersection(
                set(viirs_cells.keys())
            )
            logger.info(
                f"Found {len(common_cell_ids)} cells with both Sentinel and VIIRS data"
            )

            # Create a list of tasks for parallel processing
            tasks = []
            for cell_id in common_cell_ids:
                sentinel_file = sentinel_cells[cell_id]
                viirs_file = viirs_cells[cell_id]
                combined_cell_dir = combined_year_dir / f"cell_{cell_id}"
                combined_file = combined_cell_dir / "combined_data.npz"

                # Create output directory
                combined_cell_dir.mkdir(parents=True, exist_ok=True)

                # Check if already processed
                if skip_existing and combined_file.exists():
                    stats["cells_skipped"] += 1
                    continue

                # Add to tasks
                tasks.append(
                    (
                        cell_id,
                        sentinel_file,
                        viirs_file,
                        combined_file,
                        delete_originals,
                    )
                )

            # Exit early if no tasks
            if not tasks:
                logger.info(f"No cells to process for {country} {year}")
                continue

            # Process tasks in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = {
                    executor.submit(
                        _combine_single_cell,
                        cell_id,
                        sentinel_file,
                        viirs_file,
                        combined_file,
                        delete_originals,
                    ): cell_id
                    for cell_id, sentinel_file, viirs_file, combined_file, delete_originals in tasks
                }

                # Process results as they complete
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Combining {country} {year}",
                ):
                    cell_id = futures[future]
                    try:
                        result = future.result()
                        stats["total_cells_processed"] += 1

                        if result["status"] == "success":
                            stats["cells_successfully_combined"] += 1
                            if result.get("originals_deleted", False):
                                stats["originals_deleted"] += 1
                        elif result["status"] == "missing_sentinel":
                            stats["cells_with_missing_sentinel"] += 1
                        elif result["status"] == "missing_viirs":
                            stats["cells_with_missing_viirs"] += 1
                        elif result["status"] == "error":
                            stats["cells_with_errors"] += 1
                    except Exception as e:
                        logger.error(f"Error processing cell {cell_id}: {e}")
                        stats["cells_with_errors"] += 1

    # Log final statistics
    logger.info(f"Data combination complete. Stats: {stats}")

    # Save statistics to file
    stats_file = output_dir / "combination_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _combine_single_cell(
    cell_id: str,
    sentinel_file: Path,
    viirs_file: Path,
    combined_file: Path,
    delete_originals: bool,
) -> Dict:
    """
    Combine Sentinel and VIIRS data for a single cell.

    Args:
        cell_id: Cell ID
        sentinel_file: Path to Sentinel data file
        viirs_file: Path to VIIRS data file
        combined_file: Path to save combined data
        delete_originals: Whether to delete original files after successful combination

    Returns:
        Dictionary with processing result
    """
    logger = logging.getLogger("data_combiner")
    result = {"cell_id": cell_id}

    try:
        # Check that files exist
        if not sentinel_file.exists():
            result["status"] = "missing_sentinel"
            return result

        if not viirs_file.exists():
            result["status"] = "missing_viirs"
            return result

        # Load Sentinel data
        sentinel_data = {}
        with np.load(sentinel_file) as data:
            for band in data.files:
                sentinel_data[f"sentinel_{band}"] = data[band]

        # Load VIIRS data
        viirs_data = {}
        with np.load(viirs_file) as data:
            for band in data.files:
                viirs_data[f"viirs_{band}"] = data[band]

        # Combine data
        combined_data = {}
        combined_data.update(sentinel_data)
        combined_data.update(viirs_data)

        # Save combined data
        np.savez_compressed(combined_file, **combined_data)

        # Create metadata
        metadata = {
            "cell_id": cell_id,
            "sentinel_bands": [
                band.replace("sentinel_", "") for band in sentinel_data.keys()
            ],
            "viirs_bands": [band.replace("viirs_", "") for band in viirs_data.keys()],
            "combined_bands": list(combined_data.keys()),
            "sentinel_source": str(sentinel_file),
            "viirs_source": str(viirs_file),
            "array_shapes": {band: data.shape for band, data in combined_data.items()},
            "data_types": {
                band: str(data.dtype) for band, data in combined_data.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save metadata
        metadata_file = combined_file.parent / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Delete original files if requested
        if delete_originals:
            # Only delete if combination was successful
            try:
                sentinel_file.unlink()
                viirs_file.unlink()
                result["originals_deleted"] = True
                logger.debug(f"Deleted original files for cell {cell_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to delete original files for cell {cell_id}: {e}"
                )
                result["originals_deleted"] = False

        result["status"] = "success"
        result["combined_bands"] = list(combined_data.keys())
        return result

    except Exception as e:
        logger.error(f"Error combining data for cell {cell_id}: {e}")
        result["status"] = "error"
        result["error"] = str(e)
        return result
