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
import sys
from src.utils.paths import get_results_dir
import traceback
import gc

logger = logging.getLogger("image_processing")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


def combine_sentinel_viirs_data(
    max_workers: int = 8,
    skip_existing: bool = True,
    delete_originals: bool = True,
    countries: List[str] = None,
    years: List[int] = None,
) -> Dict[str, int]:
    """
    Combine Sentinel and VIIRS data for specified countries and years.

    Args:
        max_workers: Maximum number of parallel workers
        skip_existing: Skip cells that already have combined data
        delete_originals: Delete original npz files after successful combination
        countries: List of countries to process (defaults to all available)
        years: List of years to process (defaults to all available)

    Returns:
        Dictionary with processing statistics
    """
    # Clean up before starting
    gc.collect()

    output_dir = get_results_dir() / "Images" / "Combined"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base directories
    sentinel_base_dir = get_results_dir() / "Images" / "Sentinel"
    viirs_base_dir = get_results_dir() / "Images" / "VIIRS"

    available_countries = set()
    # Find all countries with both Sentinel and VIIRS data
    for country_dir in sentinel_base_dir.iterdir():
        if country_dir.is_dir() and (viirs_base_dir / country_dir.name).exists():
            available_countries.add(country_dir.name)

    # Filter countries if specified
    if countries:
        countries_to_process = [c for c in countries if c in available_countries]
    else:
        countries_to_process = sorted(list(available_countries))

    logger.info(f"Found {len(countries_to_process)} countries to process")

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
    for country in countries_to_process:
        logger.info(f"Processing country: {country}")

        country_years = set()
        # Find years with both Sentinel and VIIRS data
        sentinel_country_dir = sentinel_base_dir / country
        viirs_country_dir = viirs_base_dir / country

        if not sentinel_country_dir.exists() or not viirs_country_dir.exists():
            logger.warning(f"Missing data directories for country {country}, skipping")
            continue

        for year_dir in sentinel_country_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                if (viirs_country_dir / year_dir.name).exists():
                    country_years.add(int(year_dir.name))

        # Filter years if specified
        if years:
            years_to_process = [y for y in years if y in country_years]
        else:
            years_to_process = sorted(list(country_years))

        if not years_to_process:
            logger.warning(f"No years found for country {country}")
            continue

        # Process each year
        for year in years_to_process:
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

            # Process tasks in batches
            batch_size = 500  # Process 500 cells at a time
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}"
                )

                # Process batch in parallel
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
                        for cell_id, sentinel_file, viirs_file, combined_file, delete_originals in batch
                    }

                    # Process results as they complete
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Combining {country} {year} batch {i//batch_size + 1}",
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

                            # Explicitly delete the result to free memory
                            del result

                            # Periodically force garbage collection during batch processing
                            if stats["total_cells_processed"] % 100 == 0:
                                gc.collect()

                        except Exception as e:
                            logger.error(f"Error processing cell {cell_id}: {e}")
                            stats["cells_with_errors"] += 1

                # Force garbage collection after each batch
                gc.collect()
                logger.info(
                    f"Completed batch {i//batch_size + 1}, processed {i+len(batch)}/{len(tasks)} cells"
                )

    # Final garbage collection
    gc.collect()

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
    logger = logging.getLogger("image_processing")
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


def scan_for_missing_data_combined(
    expected_bands: Optional[Dict[str, List[str]]] = None,
    max_workers: int = 12,  # Number of parallel workers
    countries: List[str] = None,
    years: List[int] = None,
) -> None:
    """
    Scan processed data files to identify missing bands or indices and create failure logs.
    Uses parallel processing to speed up scanning.

    Args:
        expected_bands: Dictionary mapping data types to lists of expected bands/indices
        max_workers: Maximum number of parallel workers
        countries: List of countries to process (defaults to all available)
        years: List of years to process (defaults to all available)
    """
    logger = logging.getLogger("image_processing")
    logger.info(f"Scanning for missing data in combined set")

    # Define default expected bands if not provided
    if expected_bands is None:
        expected_bands = {
            "combined_data.npz": [
                "sentinel_rgb",
                "sentinel_nir",
                "sentinel_swir1",
                "sentinel_swir2",
                "sentinel_ndvi",
                "sentinel_built_up",
                "viirs_nightlights",
                "viirs_gradient",
            ]
        }

    # Get base directory
    base_dir = get_results_dir() / "Images" / "Combined"
    if not base_dir.exists():
        logger.warning(f"Directory does not exist: {base_dir}")
        return

    # Create failures directory if it doesn't exist
    failures_dir = base_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    failure_log_file = failures_dir / "failure_log.jsonl"

    if failure_log_file.exists():
        failure_log_file.unlink()
        logger.info(f"Deleted existing failure log: {failure_log_file}")

    available_countries = []
    # Find all countries with combined data
    for country_dir in base_dir.iterdir():
        if country_dir.is_dir() and country_dir.name != "failures":
            available_countries.append(country_dir.name)

    # Filter countries if specified
    if countries:
        countries_to_process = [c for c in countries if c in available_countries]
    else:
        countries_to_process = sorted(available_countries)

    # Define the worker function for parallel processing
    def process_cell_dir_combined(cell_dir, country_name, year):
        cell_id = cell_dir.name.replace("cell_", "")
        try:
            cell_id = int(cell_id)
        except ValueError:
            return None

        # Check processed data file
        processed_file = cell_dir / "combined_data.npz"
        if not processed_file.exists():
            # Log complete file missing
            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "country": country_name,
                "year": year,
                "cell_id": cell_id,
                "error_type": "missing_processed_file",
                "error_message": "Processed data file does not exist",
                "details": {"expected_file": str(processed_file)},
            }

            return {
                "status": "failure",
                "cell_id": cell_id,
                "failure_record": failure_record,
            }

        # Load the processed data to check for missing bands
        try:
            with np.load(processed_file) as data:
                available_bands = set(data.files)

                # Check for expected bands
                expected_file_bands = expected_bands.get("combined_data.npz", [])
                missing_bands = [
                    band for band in expected_file_bands if band not in available_bands
                ]

                # Check for zero-size or all-zero arrays
                zero_bands = []
                for band in available_bands:
                    band_data = data[band]
                    if band_data.size == 0 or np.all(band_data == 0):
                        zero_bands.append(band)

                if missing_bands or zero_bands:
                    failure_record = {
                        "timestamp": datetime.now().isoformat(),
                        "country": country_name,
                        "year": year,
                        "cell_id": cell_id,
                        "error_type": "incomplete_data",
                        "error_message": "Missing or zero bands in processed data",
                        "details": {
                            "missing_bands": missing_bands,
                            "zero_bands": zero_bands,
                            "available_bands": list(available_bands),
                        },
                    }

                    return {
                        "status": "failure",
                        "cell_id": cell_id,
                        "failure_record": failure_record,
                    }

                return {"status": "success", "cell_id": cell_id}

        except Exception as e:
            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "country": country_name,
                "year": year,
                "cell_id": cell_id,
                "error_type": "data_check_error",
                "error_message": str(e),
                "details": {"file": str(processed_file)},
                "traceback": traceback.format_exc(),
            }

            return {
                "status": "failure",
                "cell_id": cell_id,
                "failure_record": failure_record,
            }

    # Process cells in parallel
    failures_count = 0
    success_count = 0

    for country in countries_to_process:
        logger.info(f"Scanning country: {country}")
        country_years = []
        country_dir = base_dir / country
        for year_dir in country_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                country_years.append(int(year_dir.name))

        # Filter years if specified
        if years:
            years_to_process = [y for y in years if y in country_years]
        else:
            years_to_process = sorted(country_years)

        if not years_to_process:
            logger.warning(f"No years found for {country}")
            continue

        for year in years_to_process:
            logger.info(f"Processing {country}: {year}")
            country_year_dir = country_dir / str(year)

            if not country_year_dir.exists():
                logger.warning(f"Missing previously found {country} {year} directory")
                continue

            cell_dirs = list(country_year_dir.glob("cell_*"))
            if not cell_dirs:
                logger.warning(f"No cells found for {country} {year}")
                continue

            # Use ThreadPoolExecutor since this is primarily I/O bound
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_cell_dir_combined, cell_dir, country, year)
                    for cell_dir in cell_dirs
                ]

                # Process results as they complete
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Scanning {country} {year}",
                ):
                    result = future.result()
                    if result is None:
                        continue
                    elif result["status"] == "failure":
                        failures_count += 1

                        # Write to failure logs (need to handle file locking)
                        with open(failure_log_file, "a") as f:
                            f.write(json.dumps(result["failure_record"]) + "\n")

                        logger.warning(f"Cell {result['cell_id']}: Found data issues")
                    elif result["status"] == "success":
                        success_count += 1

    logger.info(
        f"Scanning complete. Found {failures_count} cells with missing or invalid data. "
        f"Successfully verified {success_count} cells."
    )

    return {"failures": failures_count, "success": success_count}
