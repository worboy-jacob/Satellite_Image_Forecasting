# repair_failures.py
"""
Repair tool for failed Sentinel and VIIRS data downloads.

This script scans for failure logs generated during Sentinel and VIIRS data processing,
and attempts to repair the failed downloads by retrying the specific operations that failed.
"""

import os
import json
import time
import random
import logging
import argparse
import concurrent.futures
import gc
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import requests
from tqdm import tqdm

import traceback

# Import necessary modules from sentinel and viirs
from src.utils.paths import get_data_dir, get_results_dir
from src.processing.resampling import (
    process_and_save_bands,
    process_and_save_viirs_bands,
    cleanup_original_files,
    calculate_nightlight_gradient,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("repair_failures.log"), logging.StreamHandler()],
)

# Configure logging only if not already configured
logger = logging.getLogger("repair_tool")
if not logger.handlers:
    file_handler = logging.FileHandler("repair_failures.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Don't add a StreamHandler here since the root logger will handle console output
    logger.setLevel(logging.INFO)
    logger.propagate = True  # Allow messages to propagate to the root logger

# Constants
BAND_RESOLUTION = {
    # 10m bands
    "B2": 10,
    "B3": 10,
    "B4": 10,
    "B8": 10,
    # 20m bands
    "B5": 20,
    "B6": 20,
    "B7": 20,
    "B8A": 20,
    "B11": 20,
    "B12": 20,
    # 60m bands
    "B1": 60,
    "B9": 60,
    "B10": 60,
}


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
        """Log a repair attempt to the persistent file."""
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

        # Also create a cell-specific repair file for easy lookup
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
        """Get a summary of all repair attempts."""
        if not self.repair_log_file.exists():
            return {"total_attempts": 0, "successful_repairs": 0, "repairs_by_type": {}}

        repairs = []
        with open(self.repair_log_file, "r") as f:
            for line in f:
                try:
                    repairs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

        # Count repairs by type and success
        repair_types = {}
        successful_repairs = 0

        for repair in repairs:
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


def initialize_earth_engine(high_volume=True):
    """Initialize Earth Engine with appropriate endpoint."""
    try:
        project_id = "wealth-satellite-forecasting"

        if high_volume:
            ee.Initialize(
                project=project_id,
                opt_url="https://earthengine-highvolume.googleapis.com",
            )
            logger.info("Initialized Earth Engine with high-volume endpoint")
        else:
            ee.Initialize(project=project_id)
            logger.info("Initialized Earth Engine with standard endpoint")

    except Exception as e:
        logger.warning(
            f"Earth Engine initialization failed: {e}. Attempting to authenticate..."
        )
        try:
            ee.Authenticate()

            if high_volume:
                ee.Initialize(
                    project=project_id,
                    opt_url="https://earthengine-highvolume.googleapis.com",
                )
                logger.info(
                    "Authenticated and initialized Earth Engine with high-volume endpoint"
                )
            else:
                ee.Initialize(project=project_id)
                logger.info(
                    "Authenticated and initialized Earth Engine with standard endpoint"
                )

        except Exception as auth_error:
            logger.error(f"Earth Engine authentication failed: {auth_error}")
            raise


def create_optimized_session(max_workers=10, use_high_volume=True):
    """
    Create an optimized HTTP session with retry logic.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Create a session
    session = requests.Session()

    # Configure retry strategy - aggressive for repair operations
    retry_strategy = Retry(
        total=12,  # More retries for repair operations
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],
        allowed_methods=["GET", "POST", "PUT"],
        respect_retry_after_header=True,
        backoff_jitter=random.uniform(0.1, 0.5),
    )

    # Longer timeouts for repair operations
    timeout = (20, 300)  # 20s connect, 5min read

    # Configure connection pooling - modest for repair tool
    pool_connections = 20
    pool_maxsize = 50

    # Configure the adapter
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry_strategy,
    )

    # Mount the adapter
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default timeout
    session.timeout = timeout

    return session


def scan_failure_logs(
    data_type: str, country_name: str, year: int, base_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Scan for failure logs for a specific country-year.

    Args:
        data_type: Either 'sentinel' or 'viirs'
        country_name: Name of the country
        year: Year to scan
        base_dir: Optional base directory, otherwise uses default results directory

    Returns:
        List of failure records
    """
    if base_dir is None:
        base_dir = get_results_dir() / "Images" / data_type.capitalize()

    failures_dir = base_dir / country_name / str(year) / "failures"

    if not failures_dir.exists():
        logger.info(
            f"No failures directory found for {country_name} {year} ({data_type})"
        )
        return []

    failure_log_file = failures_dir / "failure_log.jsonl"

    if not failure_log_file.exists():
        logger.info(
            f"No failure log file found for {country_name} {year} ({data_type})"
        )
        return []

    # Read all failure records
    failures = []
    with open(failure_log_file, "r") as f:
        for line in f:
            try:
                failure = json.loads(line.strip())
                failures.append(failure)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse failure log line: {line}")

    logger.info(
        f"Found {len(failures)} failure records for {country_name} {year} ({data_type})"
    )
    return failures


def categorize_failures(failures: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize failures by type and group them.

    Args:
        failures: List of failure records

    Returns:
        Dictionary mapping error types to lists of failure records
    """
    categorized = {}

    for failure in failures:
        error_type = failure.get("error_type", "unknown")

        if error_type not in categorized:
            categorized[error_type] = []

        categorized[error_type].append(failure)

    # Log the categories and counts
    for error_type, records in categorized.items():
        logger.info(f"Found {len(records)} failures of type '{error_type}'")

    return categorized


def load_grid_data(
    country_name: str, grid_file: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load the grid data for a country.

    Args:
        country_name: Name of the country
        grid_file: Optional path to grid file

    Returns:
        GeoDataFrame with grid data
    """
    if grid_file:
        grid_path = Path(grid_file)
    else:
        # Use default location
        grid_path = get_data_dir() / "grids" / f"{country_name}_grid.gpkg"

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid_gdf = gpd.read_file(grid_path)
    logger.info(f"Loaded grid with {len(grid_gdf)} cells for {country_name}")
    return grid_gdf


def get_cell_grid_data(grid_gdf: gpd.GeoDataFrame, cell_id: int) -> gpd.GeoDataFrame:
    """
    Extract a single cell from the grid.

    Args:
        grid_gdf: Full grid GeoDataFrame
        cell_id: ID of the cell to extract

    Returns:
        GeoDataFrame with just the requested cell
    """
    cell_gdf = grid_gdf[grid_gdf["cell_id"] == cell_id]

    if len(cell_gdf) == 0:
        raise ValueError(f"Cell ID {cell_id} not found in grid")

    return cell_gdf


def repair_sentinel_cell_failure(
    cell_id: int,
    year: int,
    country_name: str,
    grid_gdf: gpd.GeoDataFrame,
    target_crs: str,
    repair_logger: RepairFailureLogger,
    session: requests.Session,
    early_year: bool = False,
    composite_method: str = "median",
    cloud_threshold: int = 20,
    bands: Optional[List[str]] = None,
) -> bool:
    """
    Repair a failed cell for Sentinel data by reprocessing the entire cell.

    Args:
        cell_id: ID of the cell
        year: Year of the data
        country_name: Name of the country
        grid_gdf: Grid GeoDataFrame
        target_crs: Target CRS
        repair_logger: Logger for repair attempts
        early_year: Whether this is an early year (pre-2017)
        composite_method: Method for compositing
        cloud_threshold: Cloud threshold
        bands: List of bands to process (defaults to standard bands if None)

    Returns:
        True if repair was successful, False otherwise
    """
    logger.info(f"Attempting to repair full cell {cell_id}, {country_name} {year}")

    try:
        # Default bands if not specified
        if bands is None:
            if early_year:
                bands = ["B02", "B03", "B04", "B08"]  # HLS bands
            else:
                bands = ["B2", "B3", "B4", "B8"]  # Sentinel-2 bands

        # Get the cell data
        cell_gdf = get_cell_grid_data(grid_gdf, cell_id)

        # Get output directory
        output_dir = get_results_dir() / "Images" / "Sentinel"
        cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Create a placeholder file to indicate repair is in progress
        placeholder_file = cell_dir / ".repairing"
        with open(placeholder_file, "w") as f:
            f.write(f"Started: {datetime.now().isoformat()}")

        # Get monthly date ranges
        date_ranges = []
        for month in range(1, 13):
            days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month[2] = 29

            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{days_in_month[month]:02d}"
            date_ranges.append((start_date, end_date))

        # Process months in parallel
        month_workers = min(4, len(date_ranges))

        # Track month failures
        month_failures = []

        # Create a function to process a single month
        def process_single_month(month_data):
            month_idx, (start_date, end_date) = month_data
            time.sleep(random.uniform(0.1, 0.5))
            try:
                # Get collection for this month
                if early_year:
                    collection = ee.ImageCollection("NASA/HLS/HLSS30/v002")
                else:
                    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

                # Filter by date and location
                cell_gdf_wgs84 = cell_gdf.to_crs("EPSG:4326")
                bounds = cell_gdf_wgs84.total_bounds
                region = ee.Geometry.Rectangle(bounds)

                collection = collection.filterDate(start_date, end_date)
                collection = collection.filterBounds(region)

                # Filter by cloud cover
                collection = collection.filter(
                    ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)
                )

                # Apply cloud masking if not early year
                if not early_year:

                    def mask_s2_clouds(image):
                        scl = image.select("SCL")
                        mask = (
                            scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                        )
                        return image.updateMask(mask)

                    collection = collection.map(mask_s2_clouds)

                # Get count of images
                count = collection.size().getInfo()

                # Select top images by cloud cover
                if count > 0:
                    # Sort by cloud cover
                    sorted_collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE")

                    # Take up to 5 images per month
                    images_per_month = 5
                    month_selection = sorted_collection.limit(images_per_month)
                    month_selection_count = min(count, images_per_month)

                    return (
                        month_idx,
                        count,
                        month_selection,
                        month_selection_count,
                        False,
                    )
                else:
                    return (month_idx, 0, None, 0, False)

            except Exception as e:
                logger.error(f"Error processing month {month_idx+1}: {e}")
                month_failures.append((month_idx, str(e)))
                return (month_idx, 0, None, 0, True)

        # Process all months
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=month_workers
        ) as executor:
            # Submit month processing tasks
            futures = []
            for month_idx, date_range in enumerate(date_ranges):
                futures.append(
                    executor.submit(process_single_month, (month_idx, date_range))
                )

            # Collect results
            monthly_counts = [0] * 12
            selected_collections = []
            total_selected = 0
            month_error_occurred = False

            for future in concurrent.futures.as_completed(futures):
                try:
                    month_idx, count, month_selection, month_selection_count, error = (
                        future.result()
                    )

                    if error:
                        month_error_occurred = True

                    monthly_counts[month_idx] = count

                    if month_selection is not None:
                        selected_collections.append(month_selection)
                        total_selected += month_selection_count

                except Exception as e:
                    logger.error(f"Error getting month result: {e}")
                    month_error_occurred = True

            # If any month processing failed, log it and exit
            if month_error_occurred:
                error_msg = f"Failed to process one or more months for cell {cell_id}"
                logger.error(error_msg)
                repair_logger.log_repair_attempt(
                    cell_id,
                    "month_processing_error",
                    error_msg,
                    "failed",
                    {"month_failures": month_failures},
                )

                # Clean up placeholder
                if placeholder_file.exists():
                    placeholder_file.unlink()

                return False

        # If we have no images, exit
        if not selected_collections:
            logger.error(f"No images found for cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "no_images_error",
                f"No images found for cell {cell_id}",
                "failed",
                {"monthly_counts": monthly_counts},
            )

            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()

            return False

        # Merge collections
        merged_collection = selected_collections[0]
        for collection in selected_collections[1:]:
            merged_collection = merged_collection.merge(collection)

        # Create original directory
        original_dir = cell_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        # Get dimensions
        cell_gdf_wgs84 = cell_gdf.to_crs("EPSG:4326")
        bounds = cell_gdf_wgs84.total_bounds
        region = ee.Geometry.Rectangle(bounds)

        # Also get the bounds in UTM for calculating dimensions
        grid_cell_utm = cell_gdf.to_crs(target_crs)
        utm_minx, utm_miny, utm_maxx, utm_maxy = grid_cell_utm.total_bounds

        # Calculate width and height in meters
        width_meters = utm_maxx - utm_minx
        height_meters = utm_maxy - utm_miny

        # Process bands in parallel
        band_arrays = {}
        band_failures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(bands)) as executor:
            # Submit band download tasks
            futures = {}

            for band in bands:
                # Get the native resolution for this band
                if early_year:
                    scale = 30
                else:
                    scale = BAND_RESOLUTION.get(band, 10)  # Default to 10m if unknown

                # Calculate expected dimensions in pixels
                width_pixels = int(width_meters / scale)
                height_pixels = int(height_meters / scale)

                # Create the composite
                if composite_method == "median":
                    band_composite = merged_collection.select(band).median()
                elif composite_method == "mean":
                    band_composite = merged_collection.select(band).mean()
                else:
                    band_composite = merged_collection.select(band).median()

                # Get download URL
                url = band_composite.getDownloadURL(
                    {
                        "region": region,
                        "dimensions": f"{width_pixels}x{height_pixels}",
                        "format": "GEO_TIFF",
                        "crs": target_crs,
                    }
                )

                # Create a download function for this band
                def download_band(band, url):
                    max_retries = 8
                    retry_delay = 2
                    tmp_path = None

                    for attempt in range(max_retries):
                        try:
                            # Add jitter to retry delay
                            jitter = random.uniform(0.8, 1.2)
                            actual_delay = retry_delay * jitter

                            # Add timeout with exponential increase
                            timeout = min(300 + attempt * 60, 600)  # 5-10 min

                            logger.info(
                                f"Download attempt {attempt+1}/{max_retries} for band {band}"
                            )
                            response = session.get(url, timeout=timeout)

                            if response.status_code != 200:
                                logger.warning(
                                    f"Failed to download band {band}: HTTP {response.status_code}"
                                )

                                if attempt < max_retries - 1:
                                    logger.info(f"Retrying in {actual_delay:.1f}s")
                                    time.sleep(actual_delay)
                                    retry_delay *= 2  # Exponential backoff
                                    continue
                                else:
                                    logger.error(f"Max retries reached for band {band}")
                                    return None

                            # Save to a temporary file
                            with tempfile.NamedTemporaryFile(
                                suffix=".tif", delete=False
                            ) as tmp:
                                tmp.write(response.content)
                                tmp_path = tmp.name

                            # Read the GeoTIFF
                            with rasterio.open(tmp_path) as src:
                                band_array = src.read(1)
                                logger.info(
                                    f"Downloaded band {band} with shape {band_array.shape}"
                                )

                                # Verify we got reasonable data
                                if band_array.shape[0] < 10 or band_array.shape[1] < 10:
                                    logger.error(
                                        f"Band {band} has unexpectedly low resolution: {band_array.shape}"
                                    )
                                    os.unlink(tmp_path)
                                    return None

                            # Save the band to the original directory
                            output_file = original_dir / f"{band}.tif"

                            # Use actual dimensions from the array
                            height, width = band_array.shape

                            # Calculate pixel size in target CRS units
                            x_size = (utm_maxx - utm_minx) / width
                            y_size = (utm_maxy - utm_miny) / height

                            # Create a proper transform
                            transform = from_origin(utm_minx, utm_maxy, x_size, y_size)

                            with rasterio.open(
                                output_file,
                                "w",
                                driver="GTiff",
                                height=height,
                                width=width,
                                count=1,
                                dtype=band_array.dtype,
                                crs=target_crs,
                                transform=transform,
                            ) as dst:
                                dst.write(band_array, 1)

                            # Remove temporary file
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                            return band_array

                        except Exception as e:
                            logger.error(f"Error downloading band {band}: {e}")

                            # Clean up temporary file if it exists
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass

                            if attempt < max_retries - 1:
                                time.sleep(actual_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                logger.error(
                                    f"Failed to download band {band} after {max_retries} attempts"
                                )
                                return None

                # Submit the download task
                futures[executor.submit(download_band, band, url)] = band

            # Process results
            for future in concurrent.futures.as_completed(futures):
                band = futures[future]
                try:
                    band_array = future.result()
                    if band_array is not None:
                        band_arrays[band] = band_array
                    else:
                        band_failures.append(band)
                except Exception as e:
                    logger.error(f"Error in band download future for {band}: {e}")
                    band_failures.append(band)

        # Check if we have any bands
        if not band_arrays:
            logger.error(f"Failed to download any bands for cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "all_bands_failed",
                f"Failed to download any bands",
                "failed",
                {"band_failures": band_failures},
            )

            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()

            return False

        # Process and save the bands
        npz_path = process_and_save_bands(
            band_arrays=band_arrays,
            output_dir=output_dir,
            country_name=country_name,
            cell_id=cell_id,
            year=year,
        )

        if npz_path is None:
            logger.error(f"Failed to process and save bands for cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "band_processing_error",
                "Failed to process and save bands",
                "failed",
                {
                    "bands_processed": list(band_arrays.keys()),
                    "band_failures": band_failures,
                },
            )

            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()

            return False

        # Save metadata
        try:
            # Get cell centroid in WGS84 for coordinates
            cell_wgs84 = cell_gdf.to_crs("EPSG:4326")
            centroid = cell_wgs84.geometry.iloc[0].centroid

            # Get cell bounds in original CRS
            bounds = cell_gdf.total_bounds

            # Load processed data to get array information
            with np.load(npz_path) as data:
                processed_data = {key: data[key] for key in data.files}

            # Calculate file sizes for each array
            array_sizes = {}
            for key, arr in processed_data.items():
                # Calculate size in bytes (nbytes is the actual memory used)
                size_bytes = arr.nbytes
                array_sizes[key] = size_bytes

            # Categorize the arrays
            indices = [
                key for key in processed_data.keys() if key in ["ndvi", "built_up"]
            ]
            composites = [key for key in processed_data.keys() if key == "rgb"]
            individual_bands = [
                key for key in processed_data.keys() if key in ["nir", "swir1", "swir2"]
            ]

            # Create comprehensive metadata dictionary
            metadata = {
                "country": country_name,
                "cell_id": int(cell_id),
                "year": year,
                "processed_date": datetime.now().isoformat(),
                "repaired": True,  # Mark as repaired
                "npz_path": str(npz_path),
                "npz_file_size_bytes": os.path.getsize(npz_path),
                # Processing parameters
                "composite_method": composite_method,
                "cloud_threshold": cloud_threshold,
                # Spatial information
                "coordinates": {"latitude": centroid.y, "longitude": centroid.x},
                "bounds": {
                    "minx": float(bounds[0]),
                    "miny": float(bounds[1]),
                    "maxx": float(bounds[2]),
                    "maxy": float(bounds[3]),
                },
                "crs": cell_gdf.crs.to_string(),
                # Content information
                "arrays_kept": list(processed_data.keys()),
                "indices_kept": indices,
                "composites_kept": composites,
                "bands_kept": individual_bands,
                # Detailed array information
                "arrays": {
                    name: {
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                        "size_bytes": array_sizes.get(name, 0),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                    for name, arr in processed_data.items()
                },
            }

            # Save metadata to JSON file
            metadata_file = cell_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error creating metadata: {e}")
            # Continue anyway, the important part is the data

        # Log success with partial success if some bands failed
        if band_failures:
            logger.info(
                f"Partially repaired cell {cell_id} - {len(band_failures)} bands failed"
            )
            repair_logger.log_repair_attempt(
                cell_id,
                "cell_repair",
                f"Cell repair partially successful",
                "partial_success",
                {
                    "bands_processed": list(band_arrays.keys()),
                    "band_failures": band_failures,
                },
            )
        else:
            logger.info(f"Successfully repaired cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "cell_repair",
                f"Cell repair successful",
                "success",
                {"bands_processed": list(band_arrays.keys())},
            )

        # Clean up placeholder
        if placeholder_file.exists():
            placeholder_file.unlink()

        return True

    except Exception as e:
        logger.error(f"Error repairing cell {cell_id}: {e}")
        repair_logger.log_repair_attempt(
            cell_id, "cell_repair_error", str(e), "failed", {}
        )

        # Clean up placeholder if it exists
        if placeholder_file and placeholder_file.exists():
            placeholder_file.unlink()

        return False


def repair_viirs_cell_failure(
    cell_id: int,
    year: int,
    country_name: str,
    grid_gdf: gpd.GeoDataFrame,
    target_crs: str,
    session: requests.Session,
    repair_logger: RepairFailureLogger,
    composite_method: str = "median",
    bands: Optional[List[str]] = None,
) -> bool:
    """
    Repair a failed cell for VIIRS data by reprocessing the entire cell.

    Args:
        cell_id: ID of the cell
        year: Year of the data
        country_name: Name of the country
        grid_gdf: Grid GeoDataFrame
        target_crs: Target CRS
        repair_logger: Logger for repair attempts
        composite_method: Method for compositing
        bands: List of bands to process (defaults to ['avg_rad'] if None)

    Returns:
        True if repair was successful, False otherwise
    """
    logger.info(
        f"Attempting to repair full VIIRS cell {cell_id}, {country_name} {year}"
    )

    try:
        # Default bands if not specified
        if bands is None:
            bands = ["avg_rad"]  # Default VIIRS band

        # Get the cell data
        cell_gdf = get_cell_grid_data(grid_gdf, cell_id)

        # Get output directory
        output_dir = get_results_dir() / "Images" / "VIIRS"
        cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Create a placeholder file to indicate repair is in progress
        placeholder_file = cell_dir / ".repairing"
        with open(placeholder_file, "w") as f:
            f.write(f"Started: {datetime.now().isoformat()}")

        # Get date range for the entire year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Create a collection for the entire year
        cell_gdf_wgs84 = cell_gdf.to_crs("EPSG:4326")
        bounds = cell_gdf_wgs84.total_bounds
        region = ee.Geometry.Rectangle(bounds)

        # Get VIIRS collection
        collection = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        collection = collection.filterDate(start_date, end_date)
        collection = collection.filterBounds(region)

        # Get count of images
        count = collection.size().getInfo()

        if count == 0:
            logger.error(f"No VIIRS images found for cell {cell_id} in {year}")
            repair_logger.log_repair_attempt(
                cell_id,
                "no_viirs_images",
                f"No VIIRS images found",
                "failed",
                {"year": year},
            )

            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()

            return False

        logger.info(f"Found {count} VIIRS images for cell {cell_id} in {year}")

        # Create the composite
        if composite_method == "median":
            composite = collection.median()
        elif composite_method == "mean":
            composite = collection.mean()
        else:
            composite = collection.median()

        # Create original directory
        original_dir = cell_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        # Get dimensions for download
        # Also get the bounds in the target CRS for calculating dimensions
        grid_cell_utm = cell_gdf.to_crs(target_crs)
        utm_minx, utm_miny, utm_maxx, utm_maxy = grid_cell_utm.total_bounds

        # Calculate width and height in meters
        width_meters = utm_maxx - utm_minx
        height_meters = utm_maxy - utm_miny

        # Calculate expected dimensions based on VIIRS resolution (~463m)
        viirs_resolution = 463.83  # meters per pixel
        width_pixels = int(width_meters / viirs_resolution)
        height_pixels = int(height_meters / viirs_resolution)

        # Process bands in parallel
        band_arrays = {}
        band_failures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(bands)) as executor:
            # Submit band download tasks
            futures = {}

            for band in bands:
                # Get download URL
                url = composite.select(band).getDownloadURL(
                    {
                        "region": region,
                        "dimensions": f"{width_pixels}x{height_pixels}",
                        "format": "GEO_TIFF",
                        "crs": target_crs,
                    }
                )

                # Create a download function for this band
                def download_band(band, url):
                    max_retries = 8
                    retry_delay = 2
                    tmp_path = None

                    for attempt in range(max_retries):
                        try:
                            # Add jitter to retry delay
                            jitter = random.uniform(0.8, 1.2)
                            actual_delay = retry_delay * jitter

                            # Add timeout with exponential increase
                            timeout = min(300 + attempt * 60, 600)  # 5-10 min

                            logger.info(
                                f"Download attempt {attempt+1}/{max_retries} for band {band}"
                            )
                            response = session.get(url, timeout=timeout)

                            if response.status_code != 200:
                                logger.warning(
                                    f"Failed to download band {band}: HTTP {response.status_code}"
                                )

                                if attempt < max_retries - 1:
                                    logger.info(f"Retrying in {actual_delay:.1f}s")
                                    time.sleep(actual_delay)
                                    retry_delay *= 2  # Exponential backoff
                                    continue
                                else:
                                    logger.error(f"Max retries reached for band {band}")
                                    return None

                            # Save to a temporary file
                            with tempfile.NamedTemporaryFile(
                                suffix=".tif", delete=False
                            ) as tmp:
                                tmp.write(response.content)
                                tmp_path = tmp.name

                            # Read the GeoTIFF
                            with rasterio.open(tmp_path) as src:
                                band_array = src.read(1)
                                logger.info(
                                    f"Downloaded band {band} with shape {band_array.shape}"
                                )

                                # Verify we got reasonable data
                                if band_array.shape[0] < 5 or band_array.shape[1] < 5:
                                    logger.error(
                                        f"Band {band} has unexpectedly low resolution: {band_array.shape}"
                                    )
                                    os.unlink(tmp_path)
                                    return None

                            # Save the band to the original directory
                            output_file = original_dir / f"{band}.tif"

                            # Use actual dimensions from the array
                            height, width = band_array.shape

                            # Calculate pixel size in target CRS units
                            x_size = (utm_maxx - utm_minx) / width
                            y_size = (utm_maxy - utm_miny) / height

                            # Create a proper transform
                            transform = from_origin(utm_minx, utm_maxy, x_size, y_size)

                            with rasterio.open(
                                output_file,
                                "w",
                                driver="GTiff",
                                height=height,
                                width=width,
                                count=1,
                                dtype=band_array.dtype,
                                crs=target_crs,
                                transform=transform,
                            ) as dst:
                                dst.write(band_array, 1)

                            # Remove temporary file
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                            return band_array

                        except Exception as e:
                            logger.error(f"Error downloading band {band}: {e}")

                            # Clean up temporary file if it exists
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass

                            if attempt < max_retries - 1:
                                time.sleep(actual_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                logger.error(
                                    f"Failed to download band {band} after {max_retries} attempts"
                                )
                                return None

                # Submit the download task
                futures[executor.submit(download_band, band, url)] = band

            # Process results
            for future in concurrent.futures.as_completed(futures):
                band = futures[future]
                try:
                    band_array = future.result()
                    if band_array is not None:
                        band_arrays[band] = band_array
                    else:
                        band_failures.append(band)
                except Exception as e:
                    logger.error(f"Error in band download future for {band}: {e}")
                    band_failures.append(band)

        # Check if we have any bands
        if not band_arrays:
            logger.error(f"Failed to download any bands for cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "all_bands_failed",
                f"Failed to download any bands",
                "failed",
                {"band_failures": band_failures},
            )

            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()

            return False

        # Calculate gradient if we have the avg_rad band
        if "avg_rad" in band_arrays:
            try:
                gradient = calculate_nightlight_gradient(band_arrays["avg_rad"])
                if gradient is not None:
                    band_arrays["gradient"] = gradient
                    logger.info(f"Calculated nightlight gradient for cell {cell_id}")
            except Exception as e:
                logger.warning(f"Error calculating gradient: {e}")

        # Process and save the bands
        npz_path = process_and_save_viirs_bands(
            band_arrays=band_arrays,
            output_dir=output_dir,
            country_name=country_name,
            cell_id=cell_id,
            year=year,
        )

        if npz_path is None:
            logger.error(f"Failed to process and save VIIRS bands for cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "band_processing_error",
                "Failed to process and save bands",
                "failed",
                {
                    "bands_processed": list(band_arrays.keys()),
                    "band_failures": band_failures,
                },
            )

            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()

            return False

        # Save metadata
        try:
            # Get cell centroid in WGS84 for coordinates
            cell_wgs84 = cell_gdf.to_crs("EPSG:4326")
            centroid = cell_wgs84.geometry.iloc[0].centroid

            # Get cell bounds in original CRS
            bounds = cell_gdf.total_bounds

            # Load processed data to get array information
            with np.load(npz_path) as data:
                processed_data = {key: data[key] for key in data.files}

            # Calculate file sizes for each array
            array_sizes = {}
            for key, arr in processed_data.items():
                # Calculate size in bytes (nbytes is the actual memory used)
                size_bytes = arr.nbytes
                array_sizes[key] = size_bytes

            # Create comprehensive metadata dictionary
            metadata = {
                "country": country_name,
                "cell_id": int(cell_id),
                "year": year,
                "processed_date": datetime.now().isoformat(),
                "repaired": True,  # Mark as repaired
                "npz_path": str(npz_path),
                "npz_file_size_bytes": os.path.getsize(npz_path),
                # Processing parameters
                "composite_method": composite_method,
                "data_source": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
                # Spatial information
                "coordinates": {"latitude": centroid.y, "longitude": centroid.x},
                "bounds": {
                    "minx": float(bounds[0]),
                    "miny": float(bounds[1]),
                    "maxx": float(bounds[2]),
                    "maxy": float(bounds[3]),
                },
                "crs": cell_gdf.crs.to_string(),
                # Content information
                "arrays_kept": list(processed_data.keys()),
                # Detailed array information
                "arrays": {
                    name: {
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                        "size_bytes": array_sizes.get(name, 0),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                    for name, arr in processed_data.items()
                },
                # Normalization parameters
                "normalization": {
                    "nightlights": {
                        "log_min": 0,
                        "log_max": 4,
                        "method": "log_transform_with_percentile_scaling",
                    },
                    "gradient": {
                        "min": 0,
                        "max": 50,
                        "method": "global_min_max_scaling",
                    },
                },
            }

            # Save metadata to JSON file
            metadata_file = cell_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error creating metadata: {e}")
            # Continue anyway, the important part is the data

        # Log success with partial success if some bands failed
        if band_failures:
            logger.info(
                f"Partially repaired VIIRS cell {cell_id} - {len(band_failures)} bands failed"
            )
            repair_logger.log_repair_attempt(
                cell_id,
                "cell_repair",
                f"Cell repair partially successful",
                "partial_success",
                {
                    "bands_processed": list(band_arrays.keys()),
                    "band_failures": band_failures,
                },
            )
        else:
            logger.info(f"Successfully repaired VIIRS cell {cell_id}")
            repair_logger.log_repair_attempt(
                cell_id,
                "cell_repair",
                f"Cell repair successful",
                "success",
                {"bands_processed": list(band_arrays.keys())},
            )

        # Clean up placeholder
        if placeholder_file.exists():
            placeholder_file.unlink()

        return True

    except Exception as e:
        logger.error(f"Error repairing VIIRS cell {cell_id}: {e}")
        repair_logger.log_repair_attempt(
            cell_id, "cell_repair_error", str(e), "failed", {}
        )

        # Clean up placeholder if it exists
        if placeholder_file and placeholder_file.exists():
            placeholder_file.unlink()

        return False


def repair_country_year_failures(
    data_type: str,
    country_name: str,
    year: int,
    grid_gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
    failure_types: Optional[List[str]] = None,
    max_repair_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Repair all failures for a specific country-year.

    Args:
        data_type: Either 'sentinel' or 'viirs'
        country_name: Name of the country
        year: Year to process
        grid_gdf: GeoDataFrame with grid cells
        config: Configuration dictionary
        failure_types: Optional list of failure types to repair (if None, repair all)
        max_repair_attempts: Maximum number of repair attempts per cell

    Returns:
        Dictionary with repair results
    """
    logger = logging.getLogger("repair_tool")

    # Get country CRS from config
    target_crs = None
    for country_config in config.get("countries", []):
        if country_config.get("name") == country_name:
            target_crs = country_config.get("crs")
            break

    if not target_crs:
        logger.error(f"No CRS defined for country {country_name}")
        return {"status": "error", "message": "No CRS defined for country"}

    # Check if early year for Sentinel
    early_year = False
    if data_type == "sentinel" and year < 2017:
        early_year = True

    # Initialize Earth Engine if not already initialized
    try:
        ee.Image(1).getInfo()
    except:
        initialize_earth_engine()

    # Get output directory
    output_dir = get_results_dir() / "Images" / data_type.capitalize()

    # Scan for failures
    failures_dir = output_dir / country_name / str(year) / "failures"

    if not failures_dir.exists():
        logger.info(
            f"No failures directory found for {country_name} {year} ({data_type})"
        )
        return {"status": "no_failures"}

    failure_log_file = failures_dir / "failure_log.jsonl"

    if not failure_log_file.exists():
        logger.info(
            f"No failure log file found for {country_name} {year} ({data_type})"
        )
        return {"status": "no_failures"}

    # Read all failure records
    failures = []
    with open(failure_log_file, "r") as f:
        for line in f:
            try:
                failure = json.loads(line.strip())
                failures.append(failure)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse failure log line in {failure_log_file}"
                )

    if not failures:
        logger.info(f"No failures found for {country_name} {year} ({data_type})")
        return {"status": "no_failures"}

    logger.info(
        f"Found {len(failures)} failure records for {country_name} {year} ({data_type})"
    )

    # Categorize failures by type
    categorized_failures = {}
    for failure in failures:
        error_type = failure.get("error_type", "unknown")

        if error_type not in categorized_failures:
            categorized_failures[error_type] = []

        categorized_failures[error_type].append(failure)

    # Log the categories and counts
    for error_type, records in categorized_failures.items():
        logger.info(f"Found {len(records)} failures of type '{error_type}'")

    # Filter to specific failure types if requested
    if failure_types:
        categorized_failures = {
            failure_type: failures
            for failure_type, failures in categorized_failures.items()
            if failure_type in failure_types
        }

    if not categorized_failures:
        logger.info(
            f"No matching failure types found for {country_name} {year} ({data_type})"
        )
        return {"status": "no_matching_failures"}

    # Create repair logger
    repair_logger = RepairFailureLogger(output_dir, country_name, year)

    # Repair each failure type
    results = {
        "country": country_name,
        "year": year,
        "data_type": data_type,
        "failure_types": {},
        "total_cells_repaired": 0,
        "total_cells_failed": 0,
        "time_start": datetime.now().isoformat(),
    }

    # Create session for HTTP requests
    session = create_optimized_session(max_workers=10, use_high_volume=True)

    # Process each failure type
    for failure_type, failures in categorized_failures.items():
        logger.info(f"Repairing {len(failures)} failures of type '{failure_type}'")

        # Group failures by cell_id
        failures_by_cell = {}
        for failure in failures:
            cell_id = failure.get("cell_id")
            if cell_id not in failures_by_cell:
                failures_by_cell[cell_id] = []
            failures_by_cell[cell_id].append(failure)

        type_results = {
            "total_cells": len(failures_by_cell),
            "successful_repairs": 0,
            "failed_repairs": 0,
            "cells": {},
        }

        # Process each cell
        for cell_id, cell_failures in failures_by_cell.items():
            # Skip if not a valid cell_id (like 'global')
            if not isinstance(cell_id, (int, str)) or cell_id == "global":
                continue

            # Convert cell_id to int if it's a string representing a number
            try:
                if isinstance(cell_id, str) and cell_id.isdigit():
                    cell_id = int(cell_id)
            except:
                continue

            # Skip if we can't get a valid cell_id
            if not isinstance(cell_id, int):
                continue

            # Check if we've already attempted to repair this cell too many times
            cell_repair_file = (
                output_dir
                / country_name
                / str(year)
                / "repairs"
                / f"cell_{cell_id}_repair.json"
            )
            if cell_repair_file.exists():
                try:
                    with open(cell_repair_file, "r") as f:
                        existing_repairs = json.load(f)
                        if not isinstance(existing_repairs, list):
                            existing_repairs = [existing_repairs]

                    # Count previous attempts for this failure type
                    previous_attempts = sum(
                        1
                        for repair in existing_repairs
                        if repair.get("error_type") == failure_type
                    )

                    if previous_attempts >= max_repair_attempts:
                        logger.info(
                            f"Skipping cell {cell_id} - already attempted repair {previous_attempts} times"
                        )
                        type_results["cells"][str(cell_id)] = {
                            "status": "skipped",
                            "reason": f"Already attempted repair {previous_attempts} times",
                        }
                        continue

                except Exception as e:
                    logger.warning(f"Error reading repair file for cell {cell_id}: {e}")

            try:
                # Get the cell data from the grid
                cell_gdf = grid_gdf[grid_gdf["cell_id"] == cell_id]

                if len(cell_gdf) == 0:
                    logger.warning(f"Cell {cell_id} not found in grid")
                    type_results["cells"][str(cell_id)] = {
                        "status": "skipped",
                        "reason": "Cell not found in grid",
                    }
                    continue

                # Attempt to repair based on data type and failure type
                success = False

                if data_type == "sentinel":
                    success = repair_sentinel_cell_failure(
                        cell_id=cell_id,
                        year=year,
                        country_name=country_name,
                        grid_gdf=grid_gdf,
                        target_crs=target_crs,
                        repair_logger=repair_logger,
                        early_year=early_year,
                        session=session,
                    )

                elif data_type == "viirs":
                    success = repair_viirs_cell_failure(
                        cell_id=cell_id,
                        year=year,
                        country_name=country_name,
                        grid_gdf=grid_gdf,
                        target_crs=target_crs,
                        repair_logger=repair_logger,
                        session=session,
                    )

                # Record result
                if success:
                    # Remove the cell from the failure log
                    remove_cell_from_failure_log(data_type, country_name, year, cell_id)
                    logger.info(f"Successfully repaired cell {cell_id}")
                    repair_logger.log_repair_attempt(
                        cell_id,
                        "cell_repair",
                        "",
                        f"Cell repair successful",
                        "success",
                    )
                    type_results["successful_repairs"] += 1
                    type_results["cells"][str(cell_id)] = {"status": "repaired"}
                else:
                    type_results["failed_repairs"] += 1
                    type_results["cells"][str(cell_id)] = {"status": "failed"}

            except Exception as e:
                logger.error(f"Error repairing cell {cell_id}: {e}")
                logger.exception("Repair error details:")
                type_results["failed_repairs"] += 1
                type_results["cells"][str(cell_id)] = {
                    "status": "error",
                    "error": str(e),
                }

        # Add type results to overall results
        results["failure_types"][failure_type] = type_results
        results["total_cells_repaired"] += type_results["successful_repairs"]
        results["total_cells_failed"] += type_results["failed_repairs"]

    # Get repair summary
    repair_summary = repair_logger.get_repair_summary()
    results["repair_summary"] = repair_summary
    results["time_end"] = datetime.now().isoformat()

    # Save results to file
    results_file = (
        output_dir / country_name / str(year) / "repairs" / "repair_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Completed repairs for {country_name} {year} ({data_type}): "
        f"{results['total_cells_repaired']} cells repaired, "
        f"{results['total_cells_failed']} cells failed"
    )

    return results


def scan_for_missing_data(
    data_type: str,
    country_name: str,
    year: int,
    expected_bands: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Scan processed data files to identify missing bands or indices and create failure logs.
    Skip cells that already have failure logs to avoid duplication.

    Args:
        data_type: Either 'sentinel' or 'viirs'
        country_name: Name of the country
        year: Year to check
        expected_bands: Dictionary mapping data types to lists of expected bands/indices
                       If None, uses defaults for each data type
    """
    logger = logging.getLogger("data_scanner")
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

    # Get existing cell failures to avoid duplication
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

    # Keep track of failures
    failures_count = 0
    skipped_count = 0

    # Scan all cell directories
    for cell_dir in base_dir.glob("cell_*"):
        cell_id = cell_dir.name.replace("cell_", "")
        try:
            cell_id = int(cell_id)
        except ValueError:
            continue

        # Skip cells that already have failure logs
        if str(cell_id) in existing_failures:
            logger.debug(f"Skipping cell {cell_id} - already has failure logs")
            skipped_count += 1
            continue

        # Check if cell-specific failure file exists
        cell_failure_file = failures_dir / f"cell_{cell_id}_failure.json"
        if cell_failure_file.exists():
            logger.debug(f"Skipping cell {cell_id} - has individual failure file")
            skipped_count += 1
            continue

        # Check processed data file
        processed_file = cell_dir / "processed_data.npz"
        if not processed_file.exists():
            # Log complete file missing
            logger.warning(f"Missing processed data file for {cell_id}")
            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "country": country_name,
                "year": year,
                "cell_id": cell_id,
                "error_type": "missing_processed_file",
                "error_message": "Processed data file does not exist",
                "details": {"expected_file": str(processed_file)},
            }

            # Write to failure log
            with open(failure_log_file, "a") as f:
                f.write(json.dumps(failure_record) + "\n")

            # Create cell-specific failure file
            with open(cell_failure_file, "w") as f:
                json.dump(failure_record, f, indent=2)

            failures_count += 1
            continue

        # Load the processed data to check for missing bands
        try:
            with np.load(processed_file) as data:
                available_bands = set(data.files)

                # Check for expected bands
                expected_file_bands = expected_bands.get("processed_data.npz", [])
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
                    logger.warning(
                        f"Cell {cell_id}: Missing bands: {missing_bands}, Zero bands: {zero_bands}"
                    )

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

                    # Write to failure log
                    with open(failure_log_file, "a") as f:
                        f.write(json.dumps(failure_record) + "\n")

                    # Create cell-specific failure file
                    with open(cell_failure_file, "w") as f:
                        json.dump(failure_record, f, indent=2)

                    failures_count += 1

        except Exception as e:
            logger.warning(f"Error checking processed data for cell {cell_id}: {e}")

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

            # Write to failure log
            with open(failure_log_file, "a") as f:
                f.write(json.dumps(failure_record) + "\n")

            # Create cell-specific failure file
            with open(cell_failure_file, "w") as f:
                json.dump(failure_record, f, indent=2)

            failures_count += 1

    logger.info(
        f"Scanning complete. Found {failures_count} new cells with missing or invalid data. Skipped {skipped_count} cells with existing failure logs."
    )


def remove_cell_from_failure_log(
    data_type: str, country_name: str, year: int, cell_id: int
) -> bool:
    """
    Remove all entries for a specific cell from the failure log.

    Args:
        data_type: Either 'sentinel' or 'viirs'
        country_name: Name of the country
        year: Year of the data
        cell_id: ID of the cell to remove from failure log

    Returns:
        True if entries were removed, False otherwise
    """
    logger.info(
        f"Removing cell {cell_id} from {data_type} failure log for {country_name} {year}"
    )

    # Get failure log path
    failures_dir = (
        get_results_dir()
        / "Images"
        / data_type.capitalize()
        / country_name
        / str(year)
        / "failures"
    )
    failure_log_file = failures_dir / "failure_log.jsonl"

    if not failure_log_file.exists():
        logger.warning(f"No failure log file found: {failure_log_file}")
        return False

    # Read all entries, keeping only those that don't match the cell_id
    kept_entries = []
    removed_count = 0

    with open(failure_log_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entry_cell_id = entry.get("cell_id")

                # Convert string cell_id to int if needed
                if isinstance(entry_cell_id, str) and entry_cell_id.isdigit():
                    entry_cell_id = int(entry_cell_id)

                # Keep entries that don't match our cell_id
                if entry_cell_id != cell_id:
                    kept_entries.append(line)
                else:
                    removed_count += 1
            except json.JSONDecodeError:
                # Keep lines we can't parse
                kept_entries.append(line)

    # Only rewrite the file if we removed something
    if removed_count > 0:
        with open(failure_log_file, "w") as f:
            f.writelines(kept_entries)

        logger.info(
            f"Removed {removed_count} entries for cell {cell_id} from failure log"
        )
        return True

    logger.info(f"No entries found for cell {cell_id} in failure log")
    return False
