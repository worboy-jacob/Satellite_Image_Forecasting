# repair_failures.py
"""
Repair tool for failed Sentinel and VIIRS data downloads.

This script scans for failure logs generated during Sentinel and VIIRS data processing,
and attempts to repair the failed downloads by retrying the specific operations that failed.

Also includes a function to just scan for missing data and create failure logs.
"""

import json
import logging
import concurrent.futures
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import geopandas as gpd
from tqdm import tqdm
import sys
import traceback

from src.utils.paths import get_results_dir

# Configure logging
logger = logging.getLogger("image_processing")

for handler in logger.handlers:
    handler.flush = sys.stdout.flush


class RepairFailureLogger:
    """Log and persist repair failures to a file."""

    def __init__(self, base_dir: Path, country_name: str, year: int):
        self.base_dir = base_dir
        self.country_name = country_name
        self.year = year
        self.repairs_dir = self.base_dir / country_name / str(year) / "repairs"
        self.repairs_dir.mkdir(parents=True, exist_ok=True)
        self.repair_log_file = self.repairs_dir / "repair_log.jsonl"
        self.repair_summary_file = self.repairs_dir / "repair_summary.json"

    def log_repair_attempt(
        self, cell_id, error_type, original_error, repair_result, details=None
    ):
        """log_repair_attempt Log a repair attempt to the persistent file.

        Args:
            cell_id: The cell that was attempted to be repaired
            error_type: The original error type
            original_error: The original error
            repair_result: Information on the result of the repaid
            details: Specific details on the repair result. Defaults to None.

        Returns:
            Dict: A record of the repair attempt for logging
        """
        details = details or {}

        repair_record = {
            "timestamp": datetime.now().isoformat(),
            "country": self.country_name,
            "year": self.year,
            "cell_id": cell_id,
            "error_type": error_type,
            "original_error": str(original_error),
            "repair_result": repair_result,
            "success": repair_result == "success",
            "details": details,
        }

        # Append to the log file
        with open(self.repair_log_file, "a") as f:
            f.write(json.dumps(repair_record) + "\n")

        # Also create a cell-specific repair file
        cell_repair_file = self.repairs_dir / f"cell_{cell_id}_repair.json"

        # If the file exists, read it first to append
        existing_repairs = []
        if cell_repair_file.exists():
            try:
                with open(cell_repair_file, "r") as f:
                    existing_repairs = json.load(f)
                    if not isinstance(existing_repairs, list):
                        existing_repairs = [existing_repairs]
            except json.JSONDecodeError:
                existing_repairs = []

        # Append the new repair record
        existing_repairs.append(repair_record)

        # Write back to the file
        with open(cell_repair_file, "w") as f:
            json.dump(existing_repairs, f, indent=2)

        return repair_record

    def get_repair_summary(self):
        """get_repair_summary Get a summary of all repair attempts.

        Returns:
            Dict: dictionary containing the repair summary
        """
        if (
            not self.repair_log_file.exists()
        ):  # Checking for existance of a repair attempt
            return {"total_attempts": 0, "successful_repairs": 0, "repairs_by_type": {}}

        repairs = []
        with open(
            self.repair_log_file, "r"
        ) as f:  # Getting all the previous repair info
            for line in f:
                try:
                    repairs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

        # Count repairs by type and success
        repair_types = {}
        successful_repairs = 0

        for repair in repairs:  # Classifying repair
            error_type = repair.get("error_type", "unknown")
            success = repair.get("success", False)

            if error_type not in repair_types:
                repair_types[error_type] = {"attempts": 0, "successes": 0}

            repair_types[error_type]["attempts"] += 1
            if success:
                repair_types[error_type]["successes"] += 1
                successful_repairs += 1

        summary = {
            "total_attempts": len(repairs),
            "successful_repairs": successful_repairs,
            "success_rate": (
                f"{(successful_repairs / len(repairs) * 100):.1f}%"
                if repairs
                else "N/A"
            ),
            "repairs_by_type": repair_types,
            "last_update": datetime.now().isoformat(),
        }

        # Save the summary to a file
        with open(self.repair_summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def repair_country_year_failures(
    data_type: str,
    country_name: str,
    year: int,
    grid_gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
    max_repair_attempts: int = 100,
) -> Dict[str, Any]:
    """repair_country_year_failures Repair all failures for a specific country-year by reprocessing all failed cells.
    This uses a different approach from the per-cell repair, instead rerunning the
    full download pipeline on only the cells with failures.

    Args:
        data_type: Either 'sentinel' or 'viirs'
        country_name: Name of the country
        year: Year to process
        grid_gdf: GeoDataFrame with grid cells
        config: Configuration dictionary
        max_repair_attempts: Maximum number of repair attempts. Defaults to 100.

    Returns:
        Dict[str, Any]: Dictionary with repair results
    """
    logger = logging.getLogger("image_processing")

    # Get the output directory
    output_dir = get_results_dir() / "Images" / data_type.capitalize()

    # Create repair logger
    repair_logger = RepairFailureLogger(output_dir, country_name, year)

    # Create results structure
    results = {
        "country": country_name,
        "year": year,
        "data_type": data_type,
        "repair_attempts": 0,
        "cells_repaired": 0,
        "repair_rounds": [],
        "time_start": datetime.now().isoformat(),
    }

    # Initialize repair attempt counter
    repair_attempt = 0

    # Loop until all failures are fixed or max attempts reached
    while repair_attempt < max_repair_attempts:
        repair_attempt += 1
        logger.info(
            f"Starting repair attempt {repair_attempt}/{max_repair_attempts} for {country_name} {year} ({data_type})"
        )

        # Get the failures directory
        failures_dir = output_dir / country_name / str(year) / "failures"

        if not failures_dir.exists():
            logger.info(f"No failures directory found - all cells appear to be fixed")
            break

        failure_log_file = failures_dir / "failure_log.jsonl"

        if not failure_log_file.exists():
            logger.info(f"No failure log file found - all cells appear to be fixed")
            break

        # Read all failure records
        failures = []
        with open(failure_log_file, "r") as f:
            for line in f:
                try:
                    failure = json.loads(line.strip())
                    failures.append(failure)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse failure log line: {line}")

        if not failures:
            logger.info(f"No failures found - all cells appear to be fixed")
            break

        # Get unique cell IDs from failures
        failed_cell_ids = set()
        for failure in failures:
            cell_id = failure.get("cell_id")
            if cell_id is not None and cell_id != "global" and cell_id != "batch":
                # Convert string cell_id to int if needed
                if isinstance(cell_id, str) and cell_id.isdigit():
                    cell_id = int(cell_id)

                if isinstance(cell_id, int):
                    failed_cell_ids.add(cell_id)

        # Remove any batch failures from the count to focus on individual cells only
        failed_cell_ids = {
            cell_id
            for cell_id in failed_cell_ids
            if not (isinstance(cell_id, str) and cell_id.startswith("batch_"))
        }

        # Check if we have any cells to repair
        if not failed_cell_ids:
            logger.info(f"No cell-specific failures found to repair")
            break

        cells_to_repair = failed_cell_ids

        if not cells_to_repair:
            logger.info(
                f"All remaining failed cells have been marked as persistent failures"
            )
            break

        logger.info(f"Found {len(cells_to_repair)} cells to repair")

        # Create a record for this repair round
        round_record = {
            "attempt": repair_attempt,
            "cells_to_repair": len(cells_to_repair),
            "cell_ids": sorted(list(cells_to_repair)),
            "start_time": datetime.now().isoformat(),
        }

        # Delete existing data for failed cells
        deleted_count = 0
        for cell_id in cells_to_repair:
            cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"

            if cell_dir.exists():
                # Delete metadata.json
                metadata_file = cell_dir / "metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()

                # Delete processed_data.npz
                processed_file = cell_dir / "processed_data.npz"
                if processed_file.exists():
                    processed_file.unlink()

                # Delete any other processing files
                processing_file = cell_dir / ".processing"
                if processing_file.exists():
                    processing_file.unlink()

                deleted_count += 1

        logger.info(f"Deleted data for {deleted_count} cells")

        # Delete the failure log files
        failure_count = 0

        # First delete the main failure log
        if failure_log_file.exists():
            failure_log_file.unlink()
            failure_count += 1

        # Delete individual cell failure files
        for cell_id in cells_to_repair:
            cell_failure_file = failures_dir / f"cell_{cell_id}_failure.json"
            if cell_failure_file.exists():
                cell_failure_file.unlink()
                failure_count += 1

        logger.info(f"Deleted {failure_count} failure log files")

        # Create a filtered grid with only the cells that had errors
        cells_to_repair_list = list(cells_to_repair)
        repair_grid_gdf = (
            grid_gdf[grid_gdf["cell_id"].isin(cells_to_repair_list)]
            .copy()
            .reset_index(drop=True)
        )

        logger.info(
            f"Created filtered grid with {len(repair_grid_gdf)} cells to repair"
        )

        # Call the appropriate download function with the filtered grid
        try:
            if data_type.lower() == "sentinel":
                from src.download.sentinel import download_sentinel_for_country_year

                logger.info(
                    f"Running Sentinel download for {len(repair_grid_gdf)} cells"
                )

                # Pass the filtered grid to the download function to attempt to redownload the cell
                download_sentinel_for_country_year(
                    config, country_name, year, repair_grid_gdf
                )
            elif data_type.lower() == "viirs":
                from src.download.viirs import download_viirs_for_country_year

                logger.info(f"Running VIIRS download for {len(repair_grid_gdf)} cells")
                download_viirs_for_country_year(
                    config, country_name, year, repair_grid_gdf
                )
            else:
                # Recording results of incorrect data_type
                logger.error(f"Unknown data type: {data_type}")
                round_record["status"] = "error"
                round_record["error"] = f"Unknown data type: {data_type}"
                results["repair_rounds"].append(round_record)
                continue

            logger.info(f"Completed download process for {len(repair_grid_gdf)} cells")
            round_record["status"] = "completed"

        except Exception as e:
            logger.error(f"Error during download process: {e}")
            logger.exception("Download error details:")
            round_record["status"] = "error"
            round_record["error"] = str(e)
            results["repair_rounds"].append(round_record)
            continue

        # Scan for any remaining missing data
        logger.info(f"Scanning for any remaining missing data")
        scan_for_missing_data(data_type, country_name, year)

        # Update round record with end time
        round_record["end_time"] = datetime.now().isoformat()
        results["repair_rounds"].append(round_record)

        # Check if we have any failures left
        if not failures_dir.exists() or not failure_log_file.exists():
            logger.info(f"No failure log file exists - all cells appear to be fixed")
            results["cells_repaired"] += len(cells_to_repair)
            break

        # Read failure log again to check if we've made progress
        new_failures = []
        if failure_log_file.exists():
            with open(failure_log_file, "r") as f:
                for line in f:
                    try:
                        failure = json.loads(line.strip())
                        new_failures.append(failure)
                    except json.JSONDecodeError:
                        continue

        # Get new failed cell IDs
        new_failed_cell_ids = set()
        for failure in new_failures:
            cell_id = failure.get("cell_id")
            if cell_id is not None and cell_id != "global" and cell_id != "batch":
                # Convert string cell_id to int if needed
                if isinstance(cell_id, str) and cell_id.isdigit():
                    cell_id = int(cell_id)

                if isinstance(cell_id, int):
                    new_failed_cell_ids.add(cell_id)

        # Calculate which cells were successfully repaired
        repaired_cells = cells_to_repair - new_failed_cell_ids
        results["cells_repaired"] += len(repaired_cells)

        logger.info(
            f"Repaired {len(repaired_cells)} cells, {len(new_failed_cell_ids)} cells still have issues"
        )

        # If we have no more failures, exit the loop
        if not new_failed_cell_ids:
            logger.info("All cells successfully repaired!")
            break

    # Update final results
    results["repair_attempts"] = repair_attempt
    results["time_end"] = datetime.now().isoformat()

    # Get repair summary
    repair_summary = repair_logger.get_repair_summary()
    results["repair_summary"] = repair_summary

    # Save results to file
    results_file = (
        output_dir / country_name / str(year) / "repairs" / "repair_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Completed repair process for {country_name} {year} ({data_type})")
    logger.info(f"Repaired {results['cells_repaired']} cells")

    return results


def scan_for_missing_data(
    data_type: str,
    country_name: str,
    year: int,
    expected_bands: Optional[Dict[str, List[str]]] = None,
    max_workers: int = 12,  # Number of parallel workers
) -> None:
    """scan_for_missing_data Scan processed data files to identify missing bands or indices and create failure logs.
    Skip cells that already have failure logs to avoid duplication.
    Uses parallel processing to speed up scanning.

    Args:
        data_type: Either 'sentinel' or 'viirs'
        country_name: Name of the country
        year: Year to check
        expected_bands: Dictionary mapping data types to lists of expected bands/indices. Defaults to None.
        max_workers: Maximum number of parallel workers. Defaults to 12.

    Returns:
        Dict: Dictionary of information on missing data
    """
    logger = logging.getLogger("image_processing")
    logger.info(
        f"Scanning for missing data in {data_type} for {country_name}, year {year}"
    )

    # Define default expected bands if not provided
    if expected_bands is None:
        if data_type.lower() == "sentinel":
            expected_bands = {
                "processed_data.npz": [
                    "rgb",
                    "nir",
                    "swir1",
                    "swir2",
                    "ndvi",
                    "built_up",
                ]
            }
        elif data_type.lower() == "viirs":
            expected_bands = {"processed_data.npz": ["nightlights", "gradient"]}

    # Get base directory
    base_dir = (
        get_results_dir() / "Images" / data_type.capitalize() / country_name / str(year)
    )
    if not base_dir.exists():
        logger.warning(f"Directory does not exist: {base_dir}")
        return

    # Create failures directory if it doesn't exist
    failures_dir = base_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    failure_log_file = failures_dir / "failure_log.jsonl"

    # Get existing cell failures to avoid rechecking
    existing_failures = set()
    if failure_log_file.exists():
        with open(failure_log_file, "r") as f:
            for line in f:
                try:
                    failure = json.loads(line.strip())
                    cell_id = failure.get("cell_id")
                    if cell_id:
                        existing_failures.add(str(cell_id))
                except:
                    pass

    logger.info(f"Found {len(existing_failures)} cells with existing failure logs")

    # Get all cell directories
    cell_dirs = list(base_dir.glob("cell_*"))
    logger.info(f"Found {len(cell_dirs)} cell directories to scan")

    # Define the worker function for parallel processing
    def process_cell_dir(cell_dir):
        cell_id = cell_dir.name.replace("cell_", "")
        try:
            cell_id = int(cell_id)
        except ValueError:
            return None

        # Skip cells that already have failure logs
        if str(cell_id) in existing_failures:
            logger.debug(f"Skipping cell {cell_id} - already has failure logs")
            return {  # Log the skip for future review
                "status": "skipped",
                "cell_id": cell_id,
                "reason": "existing_failure",
            }

        # Check if cell-specific failure file exists
        cell_failure_file = failures_dir / f"cell_{cell_id}_failure.json"
        if cell_failure_file.exists():
            logger.debug(f"Skipping cell {cell_id} - has individual failure file")
            return {
                "status": "skipped",
                "cell_id": cell_id,
                "reason": "individual_failure_file",
            }

        # Check processed data file
        processed_file = cell_dir / "processed_data.npz"
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

            return {  # Creating a failure for repairs
                "status": "failure",
                "cell_id": cell_id,
                "failure_record": failure_record,
                "cell_failure_file": str(cell_failure_file),
            }

        # Load the processed data to check for missing bands
        try:
            missing_bands = []
            zero_bands = []
            available_bands = []
            with np.load(processed_file) as data:
                available_bands = set(data.files)

                # Check for expected bands
                expected_file_bands = expected_bands.get("processed_data.npz", [])
                missing_bands = [
                    band for band in expected_file_bands if band not in available_bands
                ]

                # Check for zero-size or all-zero arrays
                for band in available_bands:
                    band_data = data[band]
                    if band_data.size == 0 or np.all(band_data == 0):
                        zero_bands.append(band)

                if missing_bands or zero_bands:
                    failure_record = {  # Create a failure if anything is missing or 0
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
                        "cell_failure_file": str(cell_failure_file),
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
                "cell_failure_file": str(cell_failure_file),
            }

    # Process cells in parallel
    failures_count = 0
    skipped_count = 0
    success_count = 0

    batch_size = 500  # Process 500 cells at a time
    for i in range(0, len(cell_dirs), batch_size):
        batch = cell_dirs[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(cell_dirs) + batch_size - 1)//batch_size}"
        )

        # Use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_cell_dir, cell_dir) for cell_dir in batch
            ]

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Scanning batch {i//batch_size + 1}",
            ):
                result = future.result()
                if result is None:
                    continue
                if result["status"] == "skipped":
                    skipped_count += 1
                elif result["status"] == "failure":
                    failures_count += 1

                    # Write to failure logs
                    with open(failure_log_file, "a") as f:
                        f.write(json.dumps(result["failure_record"]) + "\n")

                    # Create cell-specific failure file
                    with open(result["cell_failure_file"], "w") as f:
                        json.dump(result["failure_record"], f, indent=2)

                    logger.warning(f"Cell {result['cell_id']}: Found data issues")
                elif result["status"] == "success":
                    success_count += 1
                del result
                if (skipped_count + failures_count + success_count) % 100 == 0:
                    gc.collect()

        gc.collect()
        logger.info(
            f"Completed batch {i//batch_size + 1}, processed {i+len(batch)}/{len(cell_dirs)} cells"
        )
    gc.collect()
    logger.info(
        f"Scanning complete. Found {failures_count} cells with missing or invalid data. "
        f"Skipped {skipped_count} cells with existing failure logs. "
        f"Successfully verified {success_count} cells."
    )
