# viirs.py
import concurrent.futures
import time
import gc
import os
import tempfile
from functools import lru_cache
import hashlib
import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Tuple, Optional
import rasterio
from rasterio.transform import from_origin
import calendar
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
from src.utils.paths import get_data_dir, get_results_dir
import threading
import psutil
import functools
import requests
from urllib3.util.retry import Retry
from functools import lru_cache
import hashlib
import psutil
import gc
from src.processing.resampling import (
    resample_to_256x256,
    process_and_save_viirs_bands,
    cleanup_original_files,
    calculate_nightlight_gradient,
)
import random
import json
import traceback


class FailureLogger:
    """Log and persist failures to a file during processing."""

    def __init__(self, output_dir, country_name, year):
        self.base_dir = output_dir
        self.country_name = country_name
        self.year = year
        self.failures_dir = self.base_dir / country_name / str(year) / "failures"
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log_file = self.failures_dir / "failure_log.jsonl"
        self.lock = threading.Lock()

    def log_failure(self, cell_id, error_type, error_message, details=None):
        """Log a failure to the persistent file."""
        with self.lock:
            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "country": self.country_name,
                "year": self.year,
                "cell_id": cell_id,
                "error_type": error_type,
                "error_message": str(error_message),
                "details": details or {},
                "traceback": traceback.format_exc(),
            }

            # Append to the file immediately
            with open(self.failure_log_file, "a") as f:
                f.write(json.dumps(failure_record) + "\n")

            # Also create a cell-specific failure file for easy lookup
            cell_failure_file = self.failures_dir / f"cell_{cell_id}_failure.json"
            with open(cell_failure_file, "w") as f:
                json.dump(failure_record, f, indent=2)

            return failure_record

    def get_failure_summary(self):
        """Get a summary of all failures."""
        if not self.failure_log_file.exists():
            return {"total_failures": 0, "failures_by_type": {}}

        failures = []
        with open(self.failure_log_file, "r") as f:
            for line in f:
                try:
                    failures.append(json.loads(line.strip()))
                except:
                    pass

        # Count failures by type
        failure_types = {}
        for failure in failures:
            error_type = failure.get("error_type", "unknown")
            if error_type not in failure_types:
                failure_types[error_type] = 0
            failure_types[error_type] += 1

        return {"total_failures": len(failures), "failures_by_type": failure_types}


class ValueTracker:
    """
    Track the range and average of values across multiple cells.
    Thread-safe implementation to collect statistics during parallel processing.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.max_values = {"nightlights": float("-inf"), "gradients": float("-inf")}
        self.min_values = {"nightlights": float("inf"), "gradients": float("inf")}
        self.sum_values = {"nightlights": 0.0, "gradients": 0.0}
        self.count_values = {"nightlights": 0, "gradients": 0}
        self.cell_count = 0

    def update(self, data_type: str, values: np.ndarray, sample_limit: int = 10000):
        """
        Update statistics with values from a new cell.

        Args:
            data_type: Type of data ('nightlights' or 'gradients')
            values: Array of values to include in statistics
            sample_limit: Maximum number of samples to store to avoid memory issues
        """
        if data_type not in self.max_values:
            return

        # Flatten the array and filter out invalid values
        flat_values = values.flatten()
        valid_values = flat_values[np.isfinite(flat_values)]

        if len(valid_values) == 0:
            return

        with self.lock:
            # Update min and max
            cell_min = float(np.min(valid_values))
            cell_max = float(np.max(valid_values))

            self.min_values[data_type] = min(self.min_values[data_type], cell_min)
            self.max_values[data_type] = max(self.max_values[data_type], cell_max)
            cell_sum = float(np.sum(valid_values))
            cell_count = len(valid_values)
            self.sum_values[data_type] += cell_sum
            self.count_values[data_type] += cell_count
            if data_type == "nightlights":  # Only increment once per cell
                self.cell_count += 1

    def get_statistics(self):
        """
        Get the current statistics.

        Returns:
            Dict containing min, max, and average values for each data type
        """
        with self.lock:
            stats = {}

            for data_type in self.max_values:
                count = self.count_values[data_type]
                avg = self.sum_values[data_type] / count if count > 0 else None
                stats[data_type] = {
                    "min": (
                        self.min_values[data_type]
                        if self.min_values[data_type] != float("inf")
                        else None
                    ),
                    "max": (
                        self.max_values[data_type]
                        if self.max_values[data_type] != float("-inf")
                        else None
                    ),
                    "avg": avg,
                    "sample_count": count,
                    "cell_count": self.cell_count,
                }

            return stats

    def log_statistics(self):
        """Log the current statistics."""
        stats = self.get_statistics()

        logger.info("===== VIIRS Value Range Statistics =====")

        for data_type, data_stats in stats.items():
            if data_stats["min"] is not None and data_stats["max"] is not None:
                logger.info(
                    f"{data_type.capitalize()}: "
                    f"Min={data_stats['min']:.4f}, "
                    f"Max={data_stats['max']:.4f}, "
                    f"Avg={data_stats['avg']:.4f if data_stats['avg'] is not None else 'N/A'}"
                )

                if data_type == "nightlights":
                    # For nightlights, also log the log-transformed values
                    log_min = np.log1p(max(0, data_stats["min"]))
                    log_max = np.log1p(data_stats["max"])
                    logger.info(
                        f"Log-transformed {data_type}: "
                        f"Min={log_min:.4f}, "
                        f"Max={log_max:.4f}"
                    )

        logger.info(f"Statistics based on {stats['nightlights']['sample_count']} cells")
        logger.info("========================================")


# Global variables for normalization constants
VIIRS_LOG_MIN = None
VIIRS_LOG_MAX = None
GRADIENT_MIN = None
GRADIENT_MAX = None

# Value tracker for statistics
value_tracker = ValueTracker()

logger = logging.getLogger("image_processing")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


class RequestCounter:
    """Track the number of requests made to Earth Engine."""

    def __init__(self):
        self.counts = {}
        self.lock = threading.Lock()

    def increment(self, cell_id, count=1):
        """Increment the request count for a cell."""
        with self.lock:
            if cell_id not in self.counts:
                self.counts[cell_id] = 0
            self.counts[cell_id] += count

    def get_count(self, cell_id):
        """Get the request count for a cell."""
        with self.lock:
            return self.counts.get(cell_id, 0)

    def get_average(self):
        """Get the average number of requests per cell."""
        with self.lock:
            if not self.counts:
                return 0
            return sum(self.counts.values()) / len(self.counts)

    def get_summary(self):
        """Get a summary of request counts."""
        with self.lock:
            if not self.counts:
                return "No requests tracked"
            return f"Cells: {len(self.counts)}, Total: {sum(self.counts.values())}, Avg: {self.get_average():.2f}, Max: {max(self.counts.values())}"


# Create a global request counter
request_counter = RequestCounter()


def initialize_earth_engine(config):
    """
    Initialize Earth Engine with the appropriate endpoint based on configuration.

    Args:
        config: Configuration dictionary
    """
    import ee

    # Get project ID
    project_id = config.get("project_id", "wealth-satellite-forecasting")

    # Check if high-volume endpoint should be used
    use_high_volume = config.get("use_high_volume_endpoint", True)

    try:
        if use_high_volume:
            # Initialize with high-volume endpoint
            ee.Initialize(
                project=project_id,
                opt_url="https://earthengine-highvolume.googleapis.com",
            )
            logger.info("Initialized Earth Engine with high-volume endpoint")
        else:
            # Standard initialization
            ee.Initialize(project=project_id)
            logger.info("Initialized Earth Engine with standard endpoint")
        setup_request_counting()

    except Exception as e:
        logger.warning(
            f"Earth Engine initialization failed: {e}. Attempting to authenticate..."
        )
        try:
            ee.Authenticate()

            if use_high_volume:
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
            setup_request_counting()

        except Exception as auth_error:
            logger.error(f"Earth Engine authentication failed: {auth_error}")
            raise


# Modified decorator that uses the global size
def dynamic_lru_cache(func):
    """LRU cache decorator that uses the global cache size."""
    # This is a wrapper around the standard lru_cache
    # that reads the global size variable

    # The actual cached function will be stored here
    cached_func = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal cached_func

        # Get the current global cache size
        current_size = _OPTIMAL_CACHE_SIZE

        # If the cached function doesn't exist or the size has changed,
        # create/recreate it with the current size
        if cached_func is None:
            cached_func = functools.lru_cache(maxsize=current_size)(func)

        # Call the cached function
        return cached_func(*args, **kwargs)

    # Add a method to clear the cache
    wrapper.cache_clear = lambda: (
        None if cached_func is None else cached_func.cache_clear()
    )

    return wrapper


def download_viirs_for_country_year(
    config: Dict[str, Any],
    country_name: str,
    year: int,
    grid_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Download VIIRS nightlights data for all cells of a specific country-year pair using optimized parallel processing.

    Args:
        config: Configuration dictionary
        country_name: Name of the country
        year: Year to process
        grid_gdf: GeoDataFrame containing the grid cells
    """
    # Get the output directory
    output_dir = get_results_dir() / "Images" / "VIIRS"

    # Initialize failure logger
    failure_logger = FailureLogger(output_dir, country_name, year)

    try:
        # Start the monitoring threads
        monitor_thread = threading.Thread(target=monitor_request_rate, daemon=True)
        monitor_thread.start()

        recovery_thread = threading.Thread(
            target=monitor_and_recover_processing, daemon=True
        )
        recovery_thread.start()

        # Initialize Earth Engine with appropriate endpoint
        initialize_earth_engine(config)

        # Check if using high-volume endpoint
        use_high_volume = config.get("use_high_volume_endpoint", True)

        # Clean up any leftover processing files
        cleanup_processing_files(output_dir, country_name, year)

        # Get country CRS from config
        country_crs = None
        for country in config.get("countries", []):
            if country.get("name") == country_name:
                country_crs = country.get("crs")
                break

        if not country_crs:
            error_msg = f"No CRS defined for country {country_name}"
            logger.error(error_msg)
            failure_logger.log_failure("global", "config_error", error_msg)
            return

        grid_gdf = grid_gdf.to_crs(country_crs)

        # Get VIIRS configuration
        viirs_config = config.get("viirs", {})
        bands = viirs_config.get("bands", ["avg_rad"])
        composite_method = viirs_config.get("composite_method", "median")

        # Dynamically calculate optimal parameters
        if "memory_per_worker_gb" not in config:
            config["memory_per_worker_gb"] = estimate_memory_per_worker()

        # Calculate optimal number of workers
        max_workers = calculate_optimal_workers(config)

        # Set optimal LRU cache size
        optimal_cache_size = calculate_optimal_cache_size()
        update_lru_cache_size(optimal_cache_size)

        logger.info(
            f"Processing {len(grid_gdf)} cells for {country_name}, year {year} "
            f"using up to {max_workers} workers with "
            f"{'high-volume' if use_high_volume else 'standard'} endpoint"
        )

        # Filter out cells that have already been processed
        cells_to_process = []
        for idx, cell in grid_gdf.iterrows():
            cell_id = cell["cell_id"]
            cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
            metadata_file = cell_dir / "metadata.json"

            if not metadata_file.exists():
                cells_to_process.append((idx, cell))

        if not cells_to_process:
            logger.info(
                f"All cells for {country_name}, year {year} have already been processed"
            )
            return

        logger.info(
            f"Found {len(cells_to_process)} cells to process for {country_name}, year {year}"
        )

        # Create a shared HTTP session with dynamic parameters
        session = create_optimized_session(
            max_workers=max_workers, use_high_volume=use_high_volume
        )

        # Calculate optimal batch size based on memory
        memory_per_cell_gb = config["memory_per_worker_gb"]
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use more of available memory for batch size calculation
        memory_based_batch_size = max(
            10, int((available_memory_gb * 0.9) / memory_per_cell_gb)
        )

        # Cap batch size at a reasonable maximum
        batch_size = min(memory_based_batch_size, 500, len(cells_to_process))

        logger.info(
            f"Processing cells in batches of {batch_size} (memory-based calculation)"
        )

        # Track global progress
        global_progress = {
            "completed": 0,
            "total": len(cells_to_process),
            "last_update": time.time(),
            "stalled_since": None,
        }

        # Variables to track request count for session refresh
        last_request_count = sum(request_counter.counts.values())
        last_active_time = time.time()

        with tqdm(
            total=len(cells_to_process), desc=f"Processing {country_name} {year}"
        ) as pbar:
            # Process in batches to avoid overwhelming memory
            for batch_start in range(0, len(cells_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(cells_to_process))
                current_batch = cells_to_process[batch_start:batch_end]

                logger.info(
                    f"Processing batch {batch_start//batch_size + 1}/{(len(cells_to_process)-1)//batch_size + 1} "
                    f"({len(current_batch)} cells)"
                )

                # Process batch with retry mechanism
                batch_success = False
                for batch_attempt in range(
                    5
                ):  # Increased from 3 to 5 attempts per batch
                    if batch_attempt > 0:
                        logger.info(f"Retry attempt {batch_attempt} for batch")
                        # For retries, refresh the session and clear caches
                        session = create_optimized_session(
                            max_workers=max_workers, use_high_volume=use_high_volume
                        )
                        gc.collect()

                    try:
                        # Use a ThreadPoolExecutor for better control over concurrency
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=max_workers
                        ) as executor:
                            # Submit all tasks
                            future_to_cell = {
                                executor.submit(
                                    process_viirs_cell_optimized,
                                    idx,
                                    cell,
                                    grid_gdf,
                                    country_name,
                                    year,
                                    bands,
                                    composite_method,
                                    country_crs,
                                    output_dir,
                                    session,
                                    failure_logger,  # Pass the failure logger
                                ): (idx, cell)
                                for idx, cell in current_batch
                            }

                            # Process completed tasks as they finish
                            for future in concurrent.futures.as_completed(
                                future_to_cell
                            ):
                                try:
                                    cell_id, success = future.result(
                                        timeout=360  # Increased from 300 to 360 seconds
                                    )
                                    pbar.update(1)
                                    global_progress["completed"] += 1
                                    global_progress["last_update"] = time.time()
                                    global_progress["stalled_since"] = None

                                    if not success:
                                        logger.warning(
                                            f"Failed to process cell {cell_id}"
                                        )
                                except concurrent.futures.TimeoutError:
                                    idx, cell = future_to_cell[future]
                                    cell_id = cell["cell_id"]
                                    error_msg = (
                                        f"Timeout while processing cell {cell_id}"
                                    )
                                    logger.error(error_msg)
                                    # Log the timeout failure
                                    failure_logger.log_failure(
                                        cell_id,
                                        "timeout_error",
                                        error_msg,
                                        {"batch_attempt": batch_attempt},
                                    )
                                    pbar.update(1)
                                    global_progress["completed"] += 1
                                    global_progress["last_update"] = time.time()
                                except Exception as e:
                                    idx, cell = future_to_cell[future]
                                    cell_id = cell["cell_id"]
                                    error_msg = f"Unexpected error processing cell {cell_id}: {str(e)}"
                                    logger.error(error_msg)
                                    # Log the unexpected failure
                                    failure_logger.log_failure(
                                        cell_id,
                                        "unexpected_error",
                                        str(e),
                                        {"batch_attempt": batch_attempt},
                                    )
                                    pbar.update(1)
                                    global_progress["completed"] += 1
                                    global_progress["last_update"] = time.time()

                        # If we get here, batch completed successfully
                        batch_success = True
                        break

                    except Exception as e:
                        error_msg = (
                            f"Batch processing attempt {batch_attempt+1}/5 failed: {e}"
                        )
                        logger.error(error_msg)

                        if batch_attempt < 4:  # Try up to 5 times (0-4)
                            delay = (
                                batch_attempt + 1
                            ) * 45  # Increased from 30 to 45 seconds
                            logger.info(f"Retrying batch in {delay} seconds...")
                            time.sleep(delay)

                if not batch_success:
                    logger.error(f"Failed to process batch after 5 attempts")
                    # Update progress bar for skipped cells
                    pbar.update(len(current_batch))
                    global_progress["completed"] += len(current_batch)
                    global_progress["last_update"] = time.time()
                    # Log the batch failure
                    failure_logger.log_failure(
                        f"batch_{batch_start}_{batch_end}",
                        "batch_error",
                        str(e),
                        {
                            "batch_attempt": batch_attempt,
                            "batch_start": batch_start,
                            "batch_end": batch_end,
                        },
                    )

                # Force garbage collection between batches
                gc.collect()

                # Only refresh session if we've seen signs of problems
                current_request_count = sum(request_counter.counts.values())
                if (
                    current_request_count == last_request_count
                    and time.time() - last_active_time > 60
                ):
                    logger.info("No activity detected, refreshing HTTP session")
                    session = create_optimized_session(
                        max_workers=max_workers, use_high_volume=use_high_volume
                    )

                # Update tracking variables
                last_request_count = current_request_count
                if current_request_count > last_request_count:
                    last_active_time = time.time()

        value_tracker.log_statistics()
        save_value_statistics(output_dir, country_name, year)

        # Log failure summary at the end
        failure_summary = failure_logger.get_failure_summary()
        logger.info(f"Failure summary: {failure_summary}")
        logger.info(f"Completed processing for {country_name}, year {year}")

    except Exception as e:
        error_msg = f"Critical error in download_viirs_for_country_year: {str(e)}"
        logger.error(error_msg)
        logger.exception("Detailed error:")
        # Log the global failure
        failure_logger.log_failure(
            "global", "critical_error", str(e), {"country": country_name, "year": year}
        )
        # Re-raise to allow higher-level handling
        raise


def create_optimized_session(max_workers=None, use_high_volume=True):
    """
    Create an optimized requests session with connection pooling based on system capabilities.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import psutil

    # Create a session
    session = requests.Session()

    # Configure retry strategy based on endpoint
    if use_high_volume:
        retry_strategy = Retry(
            total=12,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],
            allowed_methods=["GET", "POST", "PUT"],
            respect_retry_after_header=True,
            backoff_jitter=random.uniform(0.1, 0.5),
        )
        timeout = (20, 240)
    else:
        retry_strategy = Retry(
            total=8,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522],
            allowed_methods=["GET", "POST", "PUT"],
            respect_retry_after_header=True,
            backoff_jitter=random.uniform(0.1, 0.3),
        )
        timeout = (15, 180)

    # Calculate optimal connection pool size
    if max_workers is None:
        cpu_count = psutil.cpu_count(logical=True)
        max_workers = cpu_count * 2

    # Each worker might have multiple concurrent requests
    connections_per_worker = 4 if use_high_volume else 3

    # Calculate pool sizes with a minimum and maximum
    # IMPORTANT: These values need to be larger to avoid connection pool exhaustion
    pool_connections = min(max(50, max_workers * 2), 200 if use_high_volume else 150)
    pool_maxsize = min(
        max(100, max_workers * connections_per_worker * 2),
        400 if use_high_volume else 300,
    )

    logger.info(
        f"HTTP connection pool: connections={pool_connections}, max_size={pool_maxsize}"
    )

    # Configure the adapter with our calculated pool sizes
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry_strategy,
    )

    # Mount the adapter for both http and https
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default timeout
    session.timeout = timeout

    return session


# Use LRU cache for image collections
@dynamic_lru_cache
def get_viirs_collection_cached(bounds_key, start_date, end_date, bands_key):
    """
    Cached version of get_viirs_collection.

    Args:
        bounds_key: String representation of bounds
        start_date: Start date
        end_date: End date
        bands_key: String representation of bands

    Returns:
        ee.ImageCollection
    """
    # Convert string parameters back to original format
    bounds = [float(x) for x in bounds_key.split("_")]
    bands = bands_key.split("_") if "_" in bands_key else [bands_key]

    # Create a rectangle geometry
    ee_geometry = ee.Geometry.Rectangle(bounds)

    # Get the collection (VIIRS nightlights)
    collection = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")

    # Filter by date and location
    collection = collection.filterDate(start_date, end_date)
    collection = collection.filterBounds(ee_geometry)

    # Select bands if specified
    if bands and bands != ["None"]:
        collection = collection.select(bands)

    def get_collection_size_with_timeout(collection):
        import concurrent.futures
        import time

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(collection.size().getInfo)
            try:
                return future.result(timeout=60)  # 60 second timeout
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"Collection size operation timed out, returning empty collection"
                )
                return 0

    # Use the timeout-protected function
    count = get_collection_size_with_timeout(collection)
    return collection


# Wrapper function to use with the original API
def get_viirs_collection_optimized(
    grid_cell: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
) -> ee.ImageCollection:
    """
    Optimized version of get_viirs_collection that uses caching.
    """
    # Convert grid cell to WGS84
    grid_cell_wgs84 = grid_cell.to_crs("EPSG:4326")

    # Get bounds
    bounds = grid_cell_wgs84.total_bounds
    bounds_key = "_".join(str(x) for x in bounds)

    # Convert bands to a cache-friendly format
    bands_key = "_".join(sorted(bands)) if isinstance(bands, list) else "None"

    # Use the cached function
    return get_viirs_collection_cached(bounds_key, start_date, end_date, bands_key)


def process_viirs_cell_optimized(
    idx,
    cell,
    grid_gdf,
    country_name,
    year,
    bands,
    composite_method,
    target_crs,
    output_dir,
    session,
    failure_logger=None,  # Added failure logger parameter
):
    """
    Optimized version of process_viirs_cell.

    Args:
        idx: Index of the cell in the grid
        cell: The cell data
        grid_gdf: Full grid GeoDataFrame
        country_name: Name of the country
        year: Year to process
        bands: List of bands to download
        composite_method: Method for compositing images
        target_crs: Target CRS
        output_dir: Output directory
        session: Shared HTTP session
        failure_logger: Logger for recording failures

    Returns:
        Tuple of (cell_id, success_flag)
    """
    cell_id = cell["cell_id"]
    cell_gdf = grid_gdf.iloc[[idx]]
    placeholder_file = None

    try:
        measure_memory_usage(before=True, cell_id=cell_id)
        logger.info(f"Processing cell {cell_id} for {country_name}, year {year}")

        # Create output directory early to mark as "in progress"
        cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Create a placeholder file to indicate processing has started
        placeholder_file = cell_dir / ".processing"
        with open(placeholder_file, "w") as f:
            f.write(f"Started: {datetime.now().isoformat()}")

        # Download the data with optimized approach
        band_arrays = download_viirs_data_optimized(
            grid_cell=cell_gdf,
            year=year,
            bands=bands,
            composite_method=composite_method,
            target_crs=target_crs,
            session=session,
            cell_id=cell_id,
            failure_logger=failure_logger,  # Pass the failure logger
        )

        if not band_arrays:
            error_msg = (
                f"No data downloaded for {country_name} cell {cell_id} in {year}"
            )
            logger.warning(error_msg)
            # Log the failure
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "no_data_error", error_msg, {"bands": bands}
                )
            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()
            return cell_id, False

        # Save the data
        try:
            save_viirs_band_arrays(
                band_arrays=band_arrays,
                output_dir=output_dir,
                country_name=country_name,
                cell_id=cell_id,
                year=year,
                grid_cell=cell_gdf,
                target_crs=target_crs,
            )
        except Exception as e:
            error_msg = f"Error saving band arrays for cell {cell_id}: {str(e)}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "save_error", str(e), {"band_count": len(band_arrays)}
                )
            if placeholder_file and placeholder_file.exists():
                placeholder_file.unlink()
            return cell_id, False

        # Save metadata
        try:
            save_viirs_cell_metadata(
                country_name=country_name,
                cell_id=cell_id,
                year=year,
                bands=bands,
                cell_gdf=cell_gdf,
                output_dir=output_dir,
                composite_method=composite_method,
                band_arrays=band_arrays,
            )
        except Exception as e:
            error_msg = f"Error saving metadata for cell {cell_id}: {str(e)}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(cell_id, "metadata_error", str(e), {})
            if placeholder_file and placeholder_file.exists():
                placeholder_file.unlink()
            return cell_id, False

        # Remove placeholder file
        if placeholder_file and placeholder_file.exists():
            placeholder_file.unlink()

        logger.info(
            f"Successfully processed cell {cell_id} for {country_name} in {year}"
        )
        memory_diff = measure_memory_usage(before=False, cell_id=cell_id)
        logger.info(f"Memory used for cell {cell_id}: {memory_diff:.4f} MB")
        return cell_id, True

    except Exception as e:
        error_msg = f"Error processing cell {cell_id}: {str(e)}"
        logger.error(error_msg)
        logger.exception(f"Detailed error for cell {cell_id}:")
        # Log the failure
        if failure_logger:
            failure_logger.log_failure(cell_id, "processing_error", str(e), {})
        # Clean up placeholder if it exists
        try:
            if placeholder_file and placeholder_file.exists():
                placeholder_file.unlink()
        except:
            pass
        return cell_id, False


def download_viirs_data_optimized(
    grid_cell: gpd.GeoDataFrame,
    year: int,
    cell_id,
    bands: List[str],
    composite_method: str = "median",
    target_crs: str = "EPSG:32628",
    session=None,
    failure_logger=None,  # Added failure logger parameter
) -> Dict[str, np.ndarray]:
    """
    Download VIIRS nightlights data for a specific grid cell and year.

    Args:
        grid_cell: GeoDataFrame containing the grid cell
        year: Year to process
        cell_id: Cell identifier for logging
        bands: List of bands to download (usually just ["avg_rad"])
        composite_method: Method for compositing images (median/mean)
        target_crs: Target CRS for the output
        session: HTTP session for downloading
        failure_logger: Logger for recording failures

    Returns:
        Dictionary mapping band names to numpy arrays
    """
    try:
        # Generate date range for the entire year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Get VIIRS collection for the entire year
        try:
            collection = get_viirs_collection_optimized(
                grid_cell=grid_cell,
                start_date=start_date,
                end_date=end_date,
                bands=bands,
            )
        except Exception as e:
            error_msg = f"Error getting VIIRS collection for cell {cell_id}: {str(e)}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "collection_error", str(e), {"year": year, "bands": bands}
                )
            return {}

        # Get the count of images
        try:
            count = collection.size().getInfo()
        except Exception as e:
            error_msg = f"Error getting collection size for cell {cell_id}: {str(e)}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "collection_size_error", str(e), {"year": year}
                )
            return {}

        if count == 0:
            error_msg = f"No VIIRS images found for cell {cell_id} in {year}"
            logger.warning(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "no_images_error", error_msg, {"year": year}
                )
            return {}

        logger.info(f"Found {count} VIIRS images for cell {cell_id} in {year}")

        # Convert grid cell to WGS84 for Earth Engine region definition
        grid_cell_wgs84 = grid_cell.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = grid_cell_wgs84.total_bounds

        # Create a region in WGS84
        region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

        # Also get the bounds in the target CRS for calculating dimensions
        grid_cell_utm = grid_cell.to_crs(target_crs)
        utm_minx, utm_miny, utm_maxx, utm_maxy = grid_cell_utm.total_bounds

        # Calculate width and height in meters
        width_meters = utm_maxx - utm_minx
        height_meters = utm_maxy - utm_miny

        # Create the composite based on the specified method
        try:
            if composite_method == "median":
                composite = collection.median()
            elif composite_method == "mean":
                composite = collection.mean()
            else:
                logger.warning(
                    f"Unknown composite method: {composite_method}, using median"
                )
                composite = collection.median()
        except Exception as e:
            error_msg = f"Error creating composite for cell {cell_id}: {str(e)}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "composite_error", str(e), {"method": composite_method}
                )
            return {}

        # Calculate expected dimensions based on VIIRS resolution (~463m)
        viirs_resolution = 463.83  # meters per pixel
        width_pixels = int(width_meters / viirs_resolution)
        height_pixels = int(height_meters / viirs_resolution)

        logger.info(
            f"Calculated dimensions for cell {cell_id}: {width_pixels}x{height_pixels} pixels"
        )

        # Download each band
        band_arrays = {}

        for band in bands:
            try:
                # Get download URL with explicit dimensions
                url = composite.select(band).getDownloadURL(
                    {
                        "region": region,
                        "dimensions": f"{width_pixels}x{height_pixels}",
                        "format": "GEO_TIFF",
                        "crs": target_crs,
                    }
                )

                # Download with retries
                max_retries = 8  # Increased from 5
                retry_delay = 2
                tmp_path = None

                for attempt in range(max_retries):
                    try:
                        # Add jitter to retry delay to prevent thundering herd
                        jitter = random.uniform(0.8, 1.2)
                        actual_delay = retry_delay * jitter

                        # Add timeout with exponential increase for later attempts
                        timeout = min(
                            300 + attempt * 60, 600
                        )  # Start at 5 min, max 10 min

                        logger.debug(
                            f"Download attempt {attempt+1}/{max_retries} for band {band}, timeout={timeout}s"
                        )
                        response = session.get(url, timeout=timeout)

                        if response.status_code != 200:
                            error_msg = f"Failed to download band {band}: HTTP {response.status_code}"
                            logger.warning(error_msg)

                            if attempt < max_retries - 1:
                                logger.warning(f"Retrying in {actual_delay:.1f}s")
                                time.sleep(actual_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                logger.error(f"Max retries reached for band {band}")
                                if failure_logger:
                                    failure_logger.log_failure(
                                        cell_id,
                                        "band_download_http_error",
                                        error_msg,
                                        {
                                            "band": band,
                                            "attempt": attempt + 1,
                                            "status_code": response.status_code,
                                            "response_text": (
                                                response.text[:1000]
                                                if hasattr(response, "text")
                                                else "N/A"
                                            ),
                                        },
                                    )
                                break

                        # Save to a temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix=".tif", delete=False
                        ) as tmp:
                            tmp.write(response.content)
                            tmp_path = tmp.name

                        # Read the GeoTIFF with rasterio
                        with rasterio.open(tmp_path) as src:
                            band_array = src.read(1)
                            logger.info(
                                f"Downloaded band {band} with shape {band_array.shape}"
                            )

                            # Verify we got reasonable data
                            if band_array.shape[0] < 5 or band_array.shape[1] < 5:
                                error_msg = f"Band {band} has unexpectedly low resolution: {band_array.shape}"
                                logger.error(error_msg)

                                if failure_logger:
                                    failure_logger.log_failure(
                                        cell_id,
                                        "band_resolution_error",
                                        error_msg,
                                        {"band": band, "shape": band_array.shape},
                                    )

                                os.unlink(tmp_path)
                                tmp_path = None
                                break

                        # Remove the temporary file
                        os.unlink(tmp_path)
                        tmp_path = None

                        # Store the band array
                        band_arrays[band] = band_array
                        break  # Success, exit retry loop

                    except Exception as e:
                        error_msg = f"Error downloading band {band}, attempt {attempt+1}/{max_retries}: {str(e)}"
                        logger.warning(error_msg)

                        # Clean up temporary file if it exists
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                                tmp_path = None
                            except:
                                pass

                        if attempt < max_retries - 1:
                            time.sleep(actual_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            if failure_logger:
                                failure_logger.log_failure(
                                    cell_id,
                                    "band_download_error",
                                    str(e),
                                    {"band": band, "attempt": attempt + 1},
                                )
                            logger.error(
                                f"Failed to download band {band} after {max_retries} attempts"
                            )

            except Exception as e:
                error_msg = f"Error processing band {band}: {str(e)}"
                logger.error(error_msg)
                if failure_logger:
                    failure_logger.log_failure(
                        cell_id, "band_processing_error", str(e), {"band": band}
                    )

        # If no bands were successfully downloaded, log and return empty
        if not band_arrays:
            error_msg = f"No bands were successfully downloaded for cell {cell_id}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "all_bands_failed", error_msg, {"bands": bands}
                )
            return {}

        # Calculate gradient if we have the avg_rad band
        if "avg_rad" in band_arrays:
            try:
                # Calculate the nightlight gradient
                value_tracker.update("nightlights", band_arrays["avg_rad"])
                gradient = calculate_nightlight_gradient(band_arrays["avg_rad"])
                if gradient is not None:
                    band_arrays["gradient"] = gradient
                    value_tracker.update("gradients", gradient)
                    logger.info(f"Calculated nightlight gradient for cell {cell_id}")
                else:
                    error_msg = f"Gradient calculation returned None for cell {cell_id}"
                    logger.warning(error_msg)
                    if failure_logger:
                        failure_logger.log_failure(
                            cell_id, "gradient_calculation_none", error_msg, {}
                        )
            except Exception as e:
                error_msg = f"Error calculating nightlight gradient: {str(e)}"
                logger.error(error_msg)
                if failure_logger:
                    failure_logger.log_failure(
                        cell_id, "gradient_calculation_error", str(e), {}
                    )

        return band_arrays

    except Exception as e:
        error_msg = f"Error in download_viirs_data_optimized: {str(e)}"
        logger.error(error_msg)
        logger.exception("Detailed error:")
        if failure_logger:
            failure_logger.log_failure(
                cell_id, "download_data_error", str(e), {"year": year, "bands": bands}
            )
        return {}


def save_viirs_band_arrays(
    band_arrays: Dict[str, np.ndarray],
    output_dir: Path,
    country_name: str,
    cell_id: int,
    year: int,
    grid_cell: gpd.GeoDataFrame,
    target_crs: str,
) -> None:
    """
    Process and save VIIRS band arrays.

    Args:
        band_arrays: Dictionary mapping band names to numpy arrays
        output_dir: Directory to save the files
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data
        grid_cell: The GeoDataFrame containing the cell geometry and CRS
        target_crs: Target CRS to convert to before saving
    """
    if not band_arrays:
        logger.warning(
            f"No band arrays to save for {country_name} cell {cell_id} in {year}"
        )
        return

    # First, save original GeoTIFF files for any processing that needs geospatial info
    # This is temporary and will be cleaned up later
    original_save_path = (
        output_dir / country_name / str(year) / f"cell_{cell_id}" / "original"
    )
    original_save_path.mkdir(parents=True, exist_ok=True)

    # Convert grid cell to target CRS for saving
    grid_cell_target = grid_cell.to_crs(target_crs)

    # Get the bounds in the target CRS
    minx, miny, maxx, maxy = grid_cell_target.total_bounds

    # Save each band as a temporary GeoTIFF
    for band_name, band_array in band_arrays.items():
        if band_array is None:
            continue

        output_file = original_save_path / f"{band_name}.tif"

        # Use actual dimensions from the array
        height, width = band_array.shape

        # Calculate pixel size in target CRS units
        x_size = (maxx - minx) / width
        y_size = (maxy - miny) / height

        # Create a proper transform using the actual bounds and dimensions
        transform = from_origin(minx, maxy, x_size, y_size)

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

    # Now process and save using our optimized format
    npz_path = process_and_save_viirs_bands(
        band_arrays=band_arrays,
        output_dir=output_dir,
        country_name=country_name,
        cell_id=cell_id,
        year=year,
    )

    if npz_path is None:
        logger.warning(f"No processed data was saved for {country_name} cell {cell_id}")

    # Clean up original files to save space
    cleanup_original_files(
        output_dir=original_save_path.parent,  # Go up one level to the cell directory
        country_name=country_name,
        cell_id=cell_id,
        year=year,
    )

    logger.info(f"Processed and saved optimized VIIRS data for cell {cell_id}")


def save_viirs_cell_metadata(
    country_name: str,
    cell_id: int,
    year: int,
    bands: List[str],
    cell_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    composite_method: str,
    band_arrays: Dict[str, np.ndarray],
) -> None:
    """
    Save comprehensive metadata for a processed VIIRS cell.

    Args:
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data
        bands: List of bands that were processed
        cell_gdf: GeoDataFrame containing the cell geometry
        output_dir: Base output directory
        composite_method: Method used for compositing
        band_arrays: Dictionary of band arrays (for shape information)
    """
    # Create output directory
    cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"

    # Check if processed data exists
    npz_path = cell_dir / "processed_data.npz"
    if not npz_path.exists():
        logger.warning(f"No processed data found for cell {cell_id}")
        return

    # Load processed data to get array information
    with np.load(npz_path) as data:
        processed_data = {key: data[key] for key in data.files}

    # Get cell centroid in WGS84 for coordinates
    cell_wgs84 = cell_gdf.to_crs("EPSG:4326")
    centroid = cell_wgs84.geometry.iloc[0].centroid

    # Get cell bounds in original CRS
    bounds = cell_gdf.total_bounds

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
        ###TODO: make these more dynamic
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
        import json

        json.dump(metadata, f, indent=2)

    logger.info(f"Saved comprehensive metadata to {metadata_file}")


# Add to viirs.py

# Global variable to store the LRU cache size
_OPTIMAL_CACHE_SIZE = 200


def update_lru_cache_size(new_size):
    """Update the global LRU cache size."""
    global _OPTIMAL_CACHE_SIZE
    _OPTIMAL_CACHE_SIZE = new_size
    logger.info(f"Updated LRU cache size to {new_size}")


def setup_request_counting():
    """Set up request counting by monkey patching Earth Engine methods."""
    # Store original methods
    original_getInfo = ee.data.getInfo
    original_getList = ee.data.getList
    original_getMapId = ee.data.getMapId
    original_getDownloadId = ee.data.getDownloadId

    # Define wrapped methods
    def counted_getInfo(*args, **kwargs):
        thread_name = threading.current_thread().name
        request_counter.increment(thread_name)
        return original_getInfo(*args, **kwargs)

    def counted_getList(*args, **kwargs):
        thread_name = threading.current_thread().name
        request_counter.increment(thread_name)
        return original_getList(*args, **kwargs)

    def counted_getMapId(*args, **kwargs):
        thread_name = threading.current_thread().name
        request_counter.increment(thread_name)
        return original_getMapId(*args, **kwargs)

    def counted_getDownloadId(*args, **kwargs):
        thread_name = threading.current_thread().name
        request_counter.increment(thread_name)
        return original_getDownloadId(*args, **kwargs)

    # Replace original methods with counted versions
    ee.data.getInfo = counted_getInfo
    ee.data.getList = counted_getList
    ee.data.getMapId = counted_getMapId
    ee.data.getDownloadId = counted_getDownloadId

    logger.info("Set up request counting for Earth Engine methods")


def monitor_request_rate():
    """Monitor the actual request rate to Earth Engine with moving average."""
    last_count = 0
    last_time = time.time()
    rate_history = []

    while True:
        try:
            current_count = sum(request_counter.counts.values())
            current_time = time.time()

            elapsed = current_time - last_time
            requests = current_count - last_count

            if elapsed > 0:
                current_rate = requests / elapsed * 60  # requests per minute
                rate_history.append(current_rate)

                # Keep only the last 5 measurements for moving average
                if len(rate_history) > 5:
                    rate_history.pop(0)

                avg_rate = sum(rate_history) / len(rate_history)

                # Calculate percentage of Earth Engine limit
                limit_percentage = avg_rate / 6000 * 100

                logger.info(
                    f"Earth Engine request rate: {current_rate:.1f} req/min (avg: {avg_rate:.1f}, {limit_percentage:.1f}% of limit)"
                )

            last_count = current_count
            last_time = current_time

        except Exception as e:
            logger.error(f"Error in request rate monitoring: {e}")

        time.sleep(15)


def monitor_and_recover_processing():
    """
    Monitor the processing progress and recover from stalled states.
    This function runs in a separate thread and detects when processing has stalled.
    """
    last_request_count = 0
    last_active_time = time.time()
    consecutive_stalls = 0

    while True:
        try:
            current_count = sum(request_counter.counts.values())
            current_time = time.time()

            # Check if requests are being made
            if current_count > last_request_count:
                # Activity detected, reset stall counter
                consecutive_stalls = 0
                last_active_time = current_time
                last_request_count = current_count
                logger.debug(
                    f"Processing active: {current_count - last_request_count} new requests"
                )
            else:
                # No new requests detected
                elapsed = current_time - last_active_time
                if elapsed > 180:  # 3 minutes with no activity
                    consecutive_stalls += 1
                    logger.warning(
                        f"Processing appears stalled for {elapsed:.1f} seconds (stall #{consecutive_stalls})"
                    )

                    # After multiple consecutive stalls, try recovery actions
                    if consecutive_stalls >= 2:
                        logger.error(
                            f"Processing stalled for {consecutive_stalls} consecutive checks, attempting recovery"
                        )

                        try:
                            # Force garbage collection
                            logger.info("Forcing garbage collection")
                            gc.collect(generation=2)

                            import urllib3
                            import requests
                            from requests.adapters import HTTPAdapter

                            logger.info("Clearing connection pools")
                            # Clear connection pools
                            for pool in list(urllib3.PoolManager.pools.values()):
                                pool.clear()

                            # Refresh Earth Engine session
                            logger.info("Refreshing Earth Engine session")
                            ee.Reset()
                            time.sleep(2)
                            ee.Initialize(
                                project="wealth-satellite-forecasting",
                                opt_url="https://earthengine-highvolume.googleapis.com",
                            )
                            setup_request_counting()

                        except Exception as e:
                            logger.error(f"Error during recovery: {e}")
                            logger.exception("Recovery error details:")

                        # Reset counters after recovery attempt
                        last_active_time = current_time
                        consecutive_stalls = 0

        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")

        time.sleep(45)


def cleanup_processing_files(output_dir, country_name, year):
    """
    Clean up any leftover processing files from previous runs.

    Args:
        output_dir: Base output directory
        country_name: Name of the country
        year: Year to process
    """
    logger.info(f"Cleaning up processing files for {country_name}, year {year}")

    # Find all cell directories
    year_dir = output_dir / country_name / str(year)
    if not year_dir.exists():
        return

    cell_dirs = [
        d for d in year_dir.iterdir() if d.is_dir() and d.name.startswith("cell_")
    ]

    # Check each cell directory for processing files
    cleaned_count = 0
    for cell_dir in cell_dirs:
        processing_file = cell_dir / "processing.json"
        if processing_file.exists():
            try:
                # Read the file to log when processing started
                with open(processing_file, "r") as f:
                    content = f.read()
                logger.info(
                    f"Removing stale processing file in {cell_dir.name}: {content}"
                )

                # Delete the file
                processing_file.unlink()
                cleaned_count += 1
            except Exception as e:
                logger.warning(
                    f"Error cleaning up processing file {processing_file}: {e}"
                )

        # Also check for .processing files
        processing_marker = cell_dir / ".processing"
        if processing_marker.exists():
            try:
                processing_marker.unlink()
                cleaned_count += 1
            except Exception as e:
                logger.warning(
                    f"Error cleaning up .processing file in {cell_dir.name}: {e}"
                )

    logger.info(
        f"Cleaned up {cleaned_count} processing files for {country_name}, year {year}"
    )


def measure_memory_usage(before=True, cell_id=None):
    """Measure memory usage before/after processing a cell."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)

    if before:
        thread_local_data = threading.local()
        thread_local_data.start_memory = memory_mb
        logger.debug(f"Memory before processing cell {cell_id}: {memory_mb:.1f} MB")
    else:
        if hasattr(threading.local(), "start_memory"):
            memory_diff = memory_mb - threading.local().start_memory
            logger.debug(
                f"Memory after processing cell {cell_id}: {memory_mb:.1f} MB (diff: {memory_diff:.1f} MB)"
            )
            return memory_diff

    return 0


def estimate_memory_per_worker(sample_size=3):
    """
    Estimate memory usage per worker by processing a few sample cells.

    Args:
        sample_size: Number of cells to sample

    Returns:
        float: Estimated memory usage per worker in GB
    """
    # Record starting memory
    initial_memory = psutil.Process().memory_info().rss / (1024**3)

    # Store memory measurements
    memory_increases = []

    # Process a few sample cells with a single worker
    for i in range(sample_size):
        # Force garbage collection before measurement
        gc.collect()

        # Measure memory before
        measure_memory_usage(before=True, cell_id=f"sample_{i}")

        # Process a sample cell
        try:
            sample_result = process_sample_cell()
            memory_diff_mb = measure_memory_usage(before=False, cell_id=f"sample_{i}")
            memory_diff_gb = memory_diff_mb / 1024

            if memory_diff_gb > 0:
                memory_increases.append(memory_diff_gb)

            # Clean up
            del sample_result
            gc.collect()

        except Exception as e:
            logger.warning(f"Error during memory estimation: {e}")

    # Calculate average with safety factor
    if memory_increases:
        avg_increase = sum(memory_increases) / len(memory_increases)
        estimated_memory = avg_increase * 1.2
    else:
        estimated_memory = 0.3  # Default

    logger.info(f"Estimated memory per worker: {estimated_memory:.2f} GB")
    return max(0.1, estimated_memory)


def calculate_optimal_workers(config):
    """
    Calculate the optimal number of worker threads based on system resources and EE limits.
    """
    import psutil

    # Get configured max workers
    configured_max = config.get("max_workers", None)
    if configured_max is not None and configured_max > 0:
        return configured_max

    # Check if using high-volume endpoint
    use_high_volume = config.get("use_high_volume_endpoint", True)

    # Earth Engine rate limits
    ee_rate_limit = 6000 / 60
    max_concurrent = 40 if not use_high_volume else 80

    # Calculate request-based limit
    avg_requests = request_counter.get_average()
    requests_per_cell = max(avg_requests, 8) if avg_requests > 0 else 8
    rate_multiplier = 3.0 if use_high_volume else 2.5
    rate_limited_workers = int((ee_rate_limit / requests_per_cell) * rate_multiplier)
    if use_high_volume:
        rate_limited_workers = min(rate_limited_workers, 35)

    # CPU-based limit - more aggressive for I/O bound workloads
    cpu_count = psutil.cpu_count(logical=True)

    # Check current CPU usage
    current_cpu_percent = psutil.cpu_percent(interval=0.5)

    # If CPU usage is low, we can be more aggressive with worker count
    if current_cpu_percent < 50:
        # For I/O bound workloads, we can use many more workers than CPU cores
        cpu_based_limit = max(cpu_count * 4, 8)  # Much more aggressive
        logger.info(
            f"Low CPU usage detected ({current_cpu_percent}%), increasing worker limit"
        )
    else:
        # Standard calculation for higher CPU usage
        cpu_based_limit = max(cpu_count * 2, 4)

    # Memory-based limit
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    memory_per_worker = config.get("memory_per_worker_gb", 0.7)
    memory_based_limit = max(int(available_memory_gb / memory_per_worker), 4)

    # Network-based limit
    network_based_limit = max_concurrent

    # Calculate optimal limit with priority on rate limit
    if (
        rate_limited_workers < memory_based_limit
        and rate_limited_workers < cpu_based_limit
    ):
        # Rate limit is the bottleneck
        optimal_limit = min(network_based_limit, rate_limited_workers)
    elif memory_based_limit < cpu_based_limit:
        # Memory is the bottleneck
        optimal_limit = min(network_based_limit, memory_based_limit)
    else:
        # CPU would be the bottleneck, but we're I/O bound, so be more aggressive
        optimal_limit = min(
            network_based_limit, int(cpu_based_limit * 1.5)  # Increase by 50%
        )

    logger.info(
        f"Calculated worker limits: CPU={cpu_based_limit} (usage: {current_cpu_percent}%), "
        f"Memory={memory_based_limit}, Network={network_based_limit}, "
        f"EE Rate={rate_limited_workers} (based on {requests_per_cell:.1f} requests/cell)"
    )
    logger.info(
        f"Using {optimal_limit} workers with {'high-volume' if use_high_volume else 'standard'} endpoint"
    )

    return optimal_limit


def calculate_optimal_cache_size():
    """
    Calculate optimal LRU cache size based on available memory and measured collection size.

    Returns:
        int: Optimal cache size
    """
    import psutil

    # Get available memory in GB
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Measure actual memory per cached collection
    memory_per_cached_item_mb = measure_collection_memory_usage()

    # Calculate how many items we can cache with 15% of available memory
    memory_for_cache_gb = available_memory_gb * 0.15  # Use 15% of available memory
    memory_for_cache_mb = memory_for_cache_gb * 1024

    # Calculate cache size
    cache_size = int(memory_for_cache_mb / memory_per_cached_item_mb)

    # Set reasonable bounds
    cache_size = max(50, min(cache_size, 2000))  # Between 50 and 2000 items

    logger.info(
        f"Optimal LRU cache size: {cache_size} items "
        + f"(based on {memory_per_cached_item_mb:.2f} MB per item and "
        + f"{memory_for_cache_gb:.2f} GB allocated for cache)"
    )

    return cache_size


def measure_collection_memory_usage(sample_size=5):
    """
    Measure the actual memory usage of cached Earth Engine collections.

    Args:
        sample_size: Number of collections to sample

    Returns:
        float: Average memory usage per collection in MB
    """
    import gc
    import psutil
    import random
    import sys

    logger.info("Measuring memory usage of cached Earth Engine collections...")

    # Force garbage collection before starting
    gc.collect()

    # Initial memory usage
    initial_memory = psutil.Process().memory_info().rss

    # Create a list to hold references to collections
    collections = []

    # Sample different areas and time periods
    try:
        # Create some sample parameters
        sample_bounds = [
            [0, 0, 1, 1],  # Small area near origin
            [10, 10, 11, 11],  # Another small area
            [30, 30, 31, 31],  # Yet another area
            [-120, 30, -119, 31],  # Area in North America
            [20, 40, 21, 41],  # Area in Europe/Asia
        ]

        sample_dates = [
            ("2020-01-01", "2020-01-31"),
            ("2020-06-01", "2020-06-30"),
            ("2021-01-01", "2021-01-31"),
            ("2021-06-01", "2021-06-30"),
            ("2022-01-01", "2022-01-31"),
        ]

        # Take memory measurements after each collection
        memory_increases = []

        # Sample collections
        for i in range(min(sample_size, len(sample_bounds))):
            # Force garbage collection
            gc.collect()

            # Measure memory before
            before_memory = psutil.Process().memory_info().rss

            # Create a geometry
            bounds = sample_bounds[i % len(sample_bounds)]
            start_date, end_date = sample_dates[i % len(sample_dates)]

            # Create Earth Engine geometry
            ee_geometry = ee.Geometry.Rectangle(bounds)

            # Get VIIRS collection
            collection = (
                ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
                .filterDate(start_date, end_date)
                .filterBounds(ee_geometry)
                .select(["avg_rad"])
            )

            # Force computation by getting info
            _ = collection.size().getInfo()

            # Keep reference to prevent garbage collection
            collections.append(collection)

            # Measure memory after
            after_memory = psutil.Process().memory_info().rss

            # Calculate increase in MB
            increase_mb = (after_memory - before_memory) / (1024 * 1024)

            # Only count positive increases
            if increase_mb > 0:
                memory_increases.append(increase_mb)
                logger.debug(f"Collection {i+1} memory usage: {increase_mb:.2f} MB")

        # Calculate average with safety factor
        if memory_increases:
            avg_increase = sum(memory_increases) / len(memory_increases)
            # Add 50% safety margin
            estimated_memory = avg_increase * 1.5
        else:
            # Default if measurement fails
            estimated_memory = 5  # 5MB default

        logger.info(
            f"Measured average memory per cached collection: {estimated_memory:.2f} MB"
        )
        return max(1, estimated_memory)  # At least 1MB

    except Exception as e:
        logger.error(f"Error measuring collection memory: {e}")
        return 5  # Default to 5MB if measurement fails

    finally:
        # Clean up
        collections.clear()
        gc.collect()


def process_sample_cell():
    """
    Process a sample cell to estimate memory usage.
    Creates a small Earth Engine collection and performs typical operations.

    Returns:
        dict: Sample results
    """
    import random

    # Create a small random area
    x = random.uniform(-180, 180)
    y = random.uniform(-60, 60)

    # Create a small geometry
    geometry = ee.Geometry.Rectangle([x, y, x + 0.1, y + 0.1])

    # Get a small VIIRS collection
    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate("2020-01-01", "2020-01-15")
        .filterBounds(geometry)
        .limit(5)
    )

    # Perform typical operations
    count = collection.size().getInfo()

    if count > 0:
        # Get a composite
        composite = collection.select(["avg_rad"]).median()

        # Get some stats
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=500, maxPixels=1e6
        ).getInfo()

        return {"count": count, "stats": stats}
    else:
        # Try another area if no images found
        return {"count": 0, "stats": {}}


def save_value_statistics(output_dir: Path, country_name: str, year: int):
    """
    Save value statistics to a JSON file for future reference.

    Args:
        output_dir: Base output directory
        country_name: Name of the country
        year: Year of the data
    """
    stats = value_tracker.get_statistics()

    # Create stats directory
    stats_dir = output_dir / country_name / str(year) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    stats["timestamp"] = datetime.now().isoformat()
    stats["country"] = country_name
    stats["year"] = year

    # Save to file
    stats_file = stats_dir / "value_statistics.json"
    with open(stats_file, "w") as f:
        import json

        json.dump(stats, f, indent=2)

    logger.info(f"Saved value statistics to {stats_file}")
