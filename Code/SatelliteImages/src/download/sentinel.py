###TODO: create overall stich
###TODO: add validation
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
    process_and_save_bands,
    cleanup_original_files,
)
import json
import traceback
import random
import queue
from queue import Queue, Empty

logger = logging.getLogger("image_processing")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush

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
    Includes timeout handling and retry logic.
    """
    import ee

    # Get project ID
    project_id = config.get("project_id", "wealth-satellite-forecasting")

    # Check if high-volume endpoint should be used
    use_high_volume = config.get("use_high_volume_endpoint", True)

    # Add timeout handling
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


def download_sentinel_for_country_year(
    config: Dict[str, Any], country_name: str, year: int, grid_gdf: gpd.GeoDataFrame
) -> None:
    """
    Download Sentinel-2 data for all cells of a specific country-year pair using optimized parallel processing.
    """
    # Filter out cells that have already been processed
    # Get the output directory
    output_dir = get_results_dir() / "Images" / "Sentinel"
    cells_to_process = []
    for idx, cell in grid_gdf.iterrows():
        cell_id = cell["cell_id"]
        cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
        metadata_file = cell_dir / "metadata.json"

        if not metadata_file.exists():
            cells_to_process.append((idx, cell))

    if not cells_to_process:
        logger.info(
            f"All Sentinel-2 cells for {country_name}, year {year} have already been processed"
        )
        return

    # Initialize failure logger
    failure_logger = FailureLogger(output_dir, country_name, year)

    try:
        # Start the monitoring threads

        progress_queue = Queue()
        progress_thread_stop = threading.Event()
        progress_thread = threading.Thread(
            target=progress_updater_thread,
            args=(
                progress_queue,
                len(cells_to_process),
                f"{country_name} {year}",
                progress_thread_stop,
            ),
            daemon=True,
        )
        progress_thread.start()
        monitor_thread = threading.Thread(target=monitor_request_rate, daemon=True)
        monitor_thread.start()

        recovery_thread = threading.Thread(
            target=monitor_and_recover_processing, daemon=True
        )
        recovery_thread.start()

        # Initialize Earth Engine with appropriate endpoint
        initialize_earth_engine(config)
        early_year = False
        if year < 2017:
            early_year = True

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

        # Get Sentinel-2 configuration
        sentinel_config = config.get("sentinel", {})
        bands = sentinel_config.get("bands", ["B2", "B3", "B4", "B8"])
        cloud_threshold = sentinel_config.get("cloud_threshold", 20)
        composite_method = sentinel_config.get("composite_method", "median")

        # Dynamically calculate optimal parameters
        if "memory_per_worker_gb" not in config:
            config["memory_per_worker_gb"] = estimate_memory_per_worker(early_year)

        # Calculate optimal number of workers
        max_workers = calculate_optimal_workers(config)

        # Set optimal LRU cache size
        optimal_cache_size = calculate_optimal_cache_size(early_year=early_year)
        update_lru_cache_size(optimal_cache_size)

        logger.info(
            f"Processing {len(grid_gdf)} cells for {country_name}, year {year} "
            f"using up to {max_workers} workers with "
            f"{'high-volume' if use_high_volume else 'standard'} endpoint"
        )

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

        # Variables to track request count for session refresh
        last_request_count = sum(request_counter.counts.values())
        last_active_time = time.time()

        # Process in batches to avoid overwhelming memory
        for batch_start in range(0, len(cells_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(cells_to_process))
            current_batch = cells_to_process[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//batch_size + 1}/{(len(cells_to_process)-1)//batch_size + 1} "
                f"({len(current_batch)} cells)"
            )

            # Process batch with retry mechanism (simplified)
            batch_success = False
            for batch_attempt in range(5):
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
                        batch_timeout = time.time() + (
                            360 * len(current_batch) / max_workers
                        )
                        # Submit all tasks
                        future_to_cell = {
                            executor.submit(
                                process_sentinel_cell_optimized,
                                idx,
                                cell,
                                grid_gdf,
                                country_name,
                                year,
                                bands,
                                cloud_threshold,
                                composite_method,
                                country_crs,
                                output_dir,
                                session,
                                early_year,
                                failure_logger,
                                progress_queue,
                            ): (idx, cell)
                            for idx, cell in current_batch
                        }

                        # Process completed tasks as they finish
                        for future in concurrent.futures.as_completed(future_to_cell):
                            if time.time() > batch_timeout:
                                logger.warning(
                                    "Batch timeout reached, cancelling remaining tasks"
                                )
                                for f in future_to_cell:
                                    if not f.done():
                                        f.cancel()
                                break
                            try:
                                cell_id, success = future.result(
                                    timeout=360  # Increased from 300 to 360 seconds
                                )
                                progress_queue.put(1)

                                if not success:
                                    logger.warning(f"Failed to process cell {cell_id}")
                            except concurrent.futures.TimeoutError:
                                idx, cell = future_to_cell[future]
                                cell_id = cell["cell_id"]
                                error_msg = f"Timeout while processing cell {cell_id}"
                                logger.error(error_msg)
                                # Log the timeout failure
                                failure_logger.log_failure(
                                    cell_id,
                                    "timeout_error",
                                    error_msg,
                                    {"batch_attempt": batch_attempt},
                                )
                                progress_queue.put(1)

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
                                progress_queue.put(1)

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
                for _ in range(len(current_batch)):
                    progress_queue.put(1)

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
        progress_thread_stop.set()
        progress_thread.join(timeout=5)
        # Log failure summary at the end
        failure_summary = failure_logger.get_failure_summary()
        logger.info(f"Failure summary: {failure_summary}")
        logger.info(f"Completed processing for {country_name}, year {year}")

    except Exception as e:
        error_msg = f"Critical error in download_sentinel_for_country_year: {str(e)}"
        logger.error(error_msg)
        logger.exception("Detailed error:")
        # Log the global failure
        failure_logger.log_failure(
            "global", "critical_error", str(e), {"country": country_name, "year": year}
        )
        if "progress_thread_stop" in locals():
            progress_thread_stop.set()
            if "progress_thread" in locals():
                progress_thread.join(timeout=1)
        # Re-raise to allow higher-level handling
        raise


def create_optimized_session(max_workers=None, use_high_volume=True):
    """
    Create an optimized requests session with connection pooling and tracking.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import psutil

    if "global_session" in globals() and "global_session" in globals():
        try:
            globals()["global_session"].close()
        except:
            pass

    # Track session creation for debugging
    logger.info(f"Creating new HTTP session (high_volume={use_high_volume})")

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

    # More conservative pool sizes to prevent exhaustion
    pool_connections = min(max(30, max_workers), 100)
    pool_maxsize = min(max(60, max_workers * 3), 200)

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

    # Add session creation time for tracking staleness
    session._creation_time = time.time()
    session._expiration_time = time.time() + 1800  # 30 minutes
    globals()["global_session"] = session
    return session


@dynamic_lru_cache
def get_sentinel_collection_cached(
    bounds_key, start_date, end_date, cloud_threshold, bands_key, early_year
):
    """
    Cached version of get_sentinel_collection.

    Args:
        bounds_key: String representation of bounds
        start_date: Start date
        end_date: End date
        cloud_threshold: Cloud threshold
        bands_key: String representation of bands
        early_year: Whether to use early year collection

    Returns:
        ee.ImageCollection
    """
    # Convert string parameters back to original format
    bounds = [float(x) for x in bounds_key.split("_")]
    bands = bands_key.split("_") if "_" in bands_key else bands_key

    # Create a rectangle geometry
    ee_geometry = ee.Geometry.Rectangle(bounds)

    # Get the collection
    if early_year:
        collection = ee.ImageCollection("NASA/HLS/HLSS30/v002")

        # Filter by date and location
        collection = collection.filterDate(start_date, end_date)
        collection = collection.filterBounds(ee_geometry)

        # Filter by cloud cover - using CLOUD_COVERAGE for early years
        collection = collection.filter(ee.Filter.lt("CLOUD_COVERAGE", cloud_threshold))

        # Define cloud masking function for early years using Fmask
        def mask_hls_clouds(image):
            fmask = image.select("Fmask")
            # Mask out pixels where:
            # Bit 1 (Cloud) = 1 OR
            # Bit 3 (Cloud shadow) = 1
            cloud_bit = 1 << 1  # Bit 1 for cloud
            shadow_bit = 1 << 3  # Bit 3 for cloud shadow
            mask = (
                fmask.bitwiseAnd(cloud_bit)
                .eq(0)
                .And(fmask.bitwiseAnd(shadow_bit).eq(0))
            )
            return image.updateMask(mask)

        # Apply cloud masking for early years
        collection = collection.map(mask_hls_clouds)
    else:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

        # Filter by date and location
        collection = collection.filterDate(start_date, end_date)
        collection = collection.filterBounds(ee_geometry)

        # Filter by cloud cover
        collection = collection.filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)
        )

        # Define cloud masking function for Sentinel-2
        def mask_s2_clouds(image):
            scl = image.select("SCL")
            mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return image.updateMask(mask)

        # Apply cloud masking
        collection = collection.map(mask_s2_clouds)

    # Select bands if specified
    if bands and bands != "None":
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
def get_sentinel_collection_optimized(
    grid_cell: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    cloud_threshold: int = 20,
    bands: Optional[List[str]] = None,
    early_year: bool = False,
) -> ee.ImageCollection:
    """
    Optimized version of get_sentinel_collection that uses caching.
    """
    # Convert grid cell to WGS84
    grid_cell_wgs84 = grid_cell.to_crs("EPSG:4326")

    # Get bounds
    bounds = grid_cell_wgs84.total_bounds
    bounds_key = "_".join(str(x) for x in bounds)

    # Convert bands to a cache-friendly format
    bands_key = "_".join(sorted(bands)) if isinstance(bands, list) else "None"

    # Use the cached function
    return get_sentinel_collection_cached(
        bounds_key, start_date, end_date, cloud_threshold, bands_key, early_year
    )


def process_sentinel_cell_optimized(
    idx,
    cell,
    grid_gdf,
    country_name,
    year,
    bands,
    cloud_threshold,
    composite_method,
    target_crs,
    output_dir,
    session,
    early_year,
    failure_logger=None,
    progress_queue=None,
):
    """
    Optimized version of process_sentinel_cell with failure logging.

    Args:
        idx: Index of the cell in the grid
        cell: The cell data
        grid_gdf: Full grid GeoDataFrame
        country_name: Name of the country
        year: Year to process
        bands: List of bands to download
        cloud_threshold: Maximum cloud cover percentage
        composite_method: Method for compositing images
        target_crs: Target CRS
        output_dir: Output directory
        session: Shared HTTP session
        early_year: Whether this is an early year (pre-2017)
        failure_logger: Logger for recording failures
        progress_queue: Queue for progress updates

    Returns:
        Tuple of (cell_id, success_flag)
    """
    cell_id = cell["cell_id"]
    cell_gdf = grid_gdf.iloc[[idx]]
    placeholder_file = None
    retry = True
    trying_early_year = early_year
    try:
        while retry:
            if trying_early_year:
                retry = False
            if (
                hasattr(session, "_creation_time")
                and time.time() - session._creation_time > 1800
            ):  # 30 minutes
                logger.info(
                    f"Session for cell {cell_id} is stale, creating fresh session"
                )
                session = create_optimized_session(max_workers=20, use_high_volume=True)
            measure_memory_usage(before=True, cell_id=cell_id)
            logger.info(f"Processing cell {cell_id} for {country_name}, year {year}")

            # Create output directory early to mark as "in progress"
            cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
            cell_dir.mkdir(parents=True, exist_ok=True)

            # Create a placeholder file to indicate processing has started
            placeholder_file = cell_dir / ".processing"
            with open(placeholder_file, "w") as f:
                f.write(f"Started: {datetime.now().isoformat()}")
            band_arrays, missing = download_sentinel_data_optimized(
                grid_cell=cell_gdf,
                year=year,
                bands=bands,
                cloud_threshold=cloud_threshold,
                composite_method=composite_method,
                target_crs=target_crs,
                session=session,
                early_year=trying_early_year,
                cell_id=cell_id,
                failure_logger=failure_logger,  # Pass the failure logger
            )

            if not band_arrays or missing:
                error_msg = f"No or not enough data downloaded for {country_name} cell {cell_id} in {year}"
                logger.warning(error_msg)
                if not trying_early_year:
                    logger.info(f"Retrying with other dataset")
                    trying_early_year = True
                    continue

                # Log the failure
                if failure_logger:
                    failure_logger.log_failure(
                        cell_id,
                        "no_data_error",
                        error_msg,
                        {"bands": bands, "cloud_threshold": cloud_threshold},
                    )
                # Clean up placeholder
                if placeholder_file and placeholder_file.exists():
                    placeholder_file.unlink()
                return cell_id, False

            try:
                save_band_arrays(
                    band_arrays=band_arrays,
                    output_dir=output_dir,
                    country_name=country_name,
                    cell_id=cell_id,
                    year=year,
                    grid_cell=cell_gdf,
                    target_crs=target_crs,
                    early_year=trying_early_year,
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
            try:
                save_cell_metadata(
                    country_name=country_name,
                    cell_id=cell_id,
                    year=year,
                    bands=bands,
                    cell_gdf=cell_gdf,
                    output_dir=output_dir,
                    composite_method=composite_method,
                    cloud_threshold=cloud_threshold,
                    band_arrays=band_arrays,
                    early_year=trying_early_year,
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
            logger.debug(f"Memory used for cell {cell_id}: {memory_diff:.4f} MB")

            del band_arrays
            gc.collect()
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


def download_sentinel_data_optimized(
    grid_cell: gpd.GeoDataFrame,
    year: int,
    cell_id,
    bands: List[str],
    cloud_threshold: int = 20,
    composite_method: str = "median",
    images_per_month: int = 5,
    total_target: int = 60,
    target_crs: str = "EPSG:32628",
    session=None,
    early_year: bool = False,
    failure_logger=None,
) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Optimized version of download_sentinel_data with parallel month processing.
    Exits immediately if any month processing fails with an error.
    """
    try:
        # Track if any month processing has failed
        month_failure_occurred = False

        # Get monthly date ranges for the year
        date_ranges = get_monthly_date_ranges(year)
        month_workers = min(6, max(3, psutil.cpu_count(logical=False)))

        # Process months in parallel instead of sequentially
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=month_workers
        ) as month_executor:
            # Create a function to process a single month
            def process_single_month(month_data, max_retries=6):
                """
                Process a single month of Sentinel data with retry logic.

                Args:
                    month_data: Tuple containing (month_idx, (start_date, end_date))
                    max_retries: Maximum number of retry attempts

                Returns:
                    Tuple containing (month_idx, count, month_selection, month_selection_count, error_flag)
                """
                month_idx, (start_date, end_date) = month_data
                retry_delay = 2

                for attempt in range(max_retries):
                    jitter = random.uniform(0.8, 1.2)
                    actual_delay = retry_delay * jitter
                    try:
                        # Add a small jitter to avoid synchronized requests
                        time.sleep(random.uniform(0.1, 0.5))

                        # Get collection for this month using optimized function
                        collection = get_sentinel_collection_optimized(
                            grid_cell=grid_cell,
                            start_date=start_date,
                            end_date=end_date,
                            cloud_threshold=cloud_threshold,
                            bands=bands,
                            early_year=early_year,
                        )

                        # Get the count of images for this month with timeout protection
                        try:
                            # Increase timeout for each retry attempt
                            timeout = min(30 + attempt * 15, 120)
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=1
                            ) as executor:
                                future = executor.submit(collection.size().getInfo)
                                count = future.result(timeout=timeout)
                        except concurrent.futures.TimeoutError:
                            logger.warning(
                                f"Timeout getting collection size for month {month_idx+1}, attempt {attempt+1}"
                            )
                            if attempt < max_retries - 1:

                                logger.info(f"Retrying in {actual_delay:.1f}s")
                                time.sleep(actual_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                # Last attempt, return failure
                                return (month_idx, 0, None, 0, True)

                        # Select top images by cloud cover if available
                        if count > 0:
                            try:
                                # Sort by cloud cover (ascending)
                                if early_year:
                                    sorted_collection = collection.sort(
                                        "CLOUD_COVERAGE"
                                    )
                                else:
                                    sorted_collection = collection.sort(
                                        "CLOUDY_PIXEL_PERCENTAGE"
                                    )

                                # Take up to images_per_month images
                                month_selection = sorted_collection.limit(
                                    images_per_month
                                )
                                month_selection_count = min(count, images_per_month)
                                return (
                                    month_idx,
                                    count,
                                    month_selection,
                                    month_selection_count,
                                    False,  # No error occurred
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error sorting/limiting collection for month {month_idx+1}, attempt {attempt+1}: {e}"
                                )
                                if attempt < max_retries - 1:
                                    logger.info(f"Retrying in {actual_delay:.1f}s")
                                    time.sleep(actual_delay)
                                    retry_delay *= 2
                                    continue
                                else:
                                    # Last attempt, return failure
                                    if failure_logger:
                                        failure_logger.log_failure(
                                            cell_id,
                                            "month_sorting_error",
                                            str(e),
                                            {
                                                "month_idx": month_idx,
                                                "attempts": attempt + 1,
                                            },
                                        )
                                    return (month_idx, 0, None, 0, True)
                        else:
                            # No images but not an error
                            return (month_idx, 0, None, 0, False)

                    except Exception as e:
                        error_msg = f"Error processing month {month_idx+1}, attempt {attempt+1}/{max_retries}: {e}"
                        logger.warning(error_msg)

                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {actual_delay:.1f}s")
                            time.sleep(actual_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            # Last attempt failed
                            if failure_logger:
                                failure_logger.log_failure(
                                    cell_id,
                                    "month_processing_error",
                                    str(e),
                                    {
                                        "month_idx": month_idx,
                                        "start_date": start_date,
                                        "end_date": end_date,
                                        "attempts": attempt + 1,
                                    },
                                )
                            logger.error(
                                f"Error processing month {month_idx+1} after {max_retries} attempts: {e}"
                            )

            # Submit all months for parallel processing
            month_futures = {}
            for month_idx, date_range in enumerate(date_ranges):
                time.sleep(0.2)
                future = month_executor.submit(
                    process_single_month, (month_idx, date_range)
                )
                month_futures[future] = month_idx
                # Add a small delay between submissions to smooth out request rate
                time.sleep(0.1)

            # Process results as they complete
            monthly_counts = [0] * 12  # Initialize count array for all months
            selected_collections = []
            total_selected = 0

            for future in concurrent.futures.as_completed(month_futures):
                try:
                    (
                        month_idx,
                        count,
                        month_selection,
                        month_selection_count,
                        error_occurred,
                    ) = future.result()

                    # If any month processing had an error, mark for early exit
                    if error_occurred:
                        month_failure_occurred = True
                        # Continue to collect all results but we'll exit after
                    monthly_counts[month_idx] = count

                    if month_selection is not None:
                        selected_collections.append(month_selection)
                        total_selected += month_selection_count
                except Exception as e:
                    error_msg = f"Error processing month result: {e}"
                    logger.error(error_msg)
                    if failure_logger:
                        failure_logger.log_failure(
                            cell_id, "month_result_error", str(e), {}
                        )
                    month_failure_occurred = True

        # If any month processing failed, log it and exit early
        if month_failure_occurred:
            error_msg = (
                f"Exiting cell {cell_id} processing due to month processing failures"
            )
            logger.warning(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id,
                    "cell_aborted_due_to_month_failures",
                    error_msg,
                    {"year": year},
                )
            return {}, False
        # Continue only if there were no month processing failures
        # If we have fewer than total_target images, get more from months with extras
        if total_selected < total_target:
            additional_needed = total_target - total_selected

            # Find months with extra images
            extra_images = []
            for month_idx, count in enumerate(monthly_counts):
                if count > images_per_month:
                    extras = count - images_per_month
                    extra_images.append((month_idx, extras))

            # Sort months by number of extra images (descending)
            extra_images.sort(key=lambda x: x[1], reverse=True)

            # Get additional images from months with extras
            additional_collections = []
            additional_count = 0

            for month_idx, extras in extra_images:
                if additional_count >= additional_needed:
                    break

                month_num = month_idx + 1

                try:
                    # Get collection for this month
                    collection = get_sentinel_collection_optimized(
                        grid_cell=grid_cell,
                        start_date=date_ranges[month_idx][0],
                        end_date=date_ranges[month_idx][1],
                        cloud_threshold=cloud_threshold,
                        bands=bands,
                        early_year=early_year,
                    )

                    # Sort by cloud cover (ascending)
                    if early_year:
                        sorted_collection = collection.sort("CLOUD_COVERAGE")
                    else:
                        sorted_collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE")

                    # Skip the first images_per_month images (already selected)
                    # and take up to the number needed
                    to_take = min(extras, additional_needed - additional_count)

                    if to_take > 0:
                        count = collection.size().getInfo()
                        additional = sorted_collection.toList(count).slice(
                            images_per_month, images_per_month + to_take
                        )
                        additional_collection = ee.ImageCollection(additional)
                        additional_collections.append(additional_collection)
                        additional_count += to_take
                except Exception as e:
                    error_msg = (
                        f"Error getting additional images for month {month_idx+1}: {e}"
                    )
                    # Exit early for additional image failures too
                    error_msg = f"Exiting cell {cell_id} processing due to additional image collection failures"
                    logger.warning(error_msg)
                    if failure_logger:
                        failure_logger.log_failure(
                            cell_id,
                            "cell_aborted_due_to_additional_image_failures",
                            error_msg,
                            {"year": year, "month_idx": month_idx},
                        )
                    return {}, False

            # Add additional collections to selected collections
            selected_collections.extend(additional_collections)

        if not selected_collections or len(selected_collections) == 0:
            total_images = 0
        else:
            merged_collection = selected_collections[0]
            for collection in selected_collections[1:]:
                merged_collection = merged_collection.merge(collection)
            # Get the total number of images
            total_images = merged_collection.size().getInfo()

        logger.debug(f"{cell_id} check of total images: {total_images}")
        if not selected_collections or total_images < 10:
            logger.warning(f"No images selected for {year}")
            if failure_logger and early_year:
                failure_logger.log_failure(
                    cell_id,
                    "no_images_error",
                    f"No images selected for {year}",
                    {"monthly_counts": monthly_counts},
                )
            return None, True

        merged_collection = selected_collections[0]
        for collection in selected_collections[1:]:
            merged_collection = merged_collection.merge(collection)

        # Get the total number of images
        total_images = merged_collection.size().getInfo()
        logger.info(f"Final image count for cell: {cell_id} {year}: {total_images}")

        # Convert grid cell to WGS84 for Earth Engine region definition
        grid_cell_wgs84 = grid_cell.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = grid_cell_wgs84.total_bounds

        # Create a region in WGS84
        region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

        # Also get the bounds in UTM for calculating dimensions
        grid_cell_utm = grid_cell.to_crs(target_crs)
        utm_minx, utm_miny, utm_maxx, utm_maxy = grid_cell_utm.total_bounds

        # Calculate width and height in meters
        width_meters = utm_maxx - utm_minx
        height_meters = utm_maxy - utm_miny

        # Create dictionaries to store the band arrays
        band_arrays = {}

        # Process bands in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(bands) * 2, 20)
        ) as executor:
            # Submit download tasks for each band
            future_to_band = {}

            for band in bands:
                future = executor.submit(
                    download_single_band,
                    merged_collection=merged_collection,
                    band=band,
                    composite_method=composite_method,
                    region=region,
                    width_meters=width_meters,
                    height_meters=height_meters,
                    target_crs=target_crs,
                    session=session,
                    early_year=early_year,
                    failure_logger=failure_logger,
                    cell_id=cell_id,
                )
                future_to_band[future] = band

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_band):
                band = future_to_band[future]
                try:
                    band_array = future.result()
                    if band_array is not None:
                        band_arrays[band] = band_array
                    else:
                        if failure_logger:
                            failure_logger.log_failure(
                                cell_id,
                                "band_download_failed",
                                f"Band {band} download returned None",
                                {"band": band},
                            )
                except Exception as e:
                    error_msg = f"Error downloading band {band}: {e}"
                    logger.error(error_msg)
                    if failure_logger:
                        failure_logger.log_failure(
                            cell_id, "band_future_error", str(e), {"band": band}
                        )

        # If no bands were successfully downloaded, log failure and return empty
        if not band_arrays:
            error_msg = f"All bands failed to download for cell {cell_id}"
            logger.error(error_msg)
            if failure_logger:
                failure_logger.log_failure(
                    cell_id, "all_bands_failed", error_msg, {"bands": bands}
                )
            return {}, False

        return band_arrays, False

    except Exception as e:
        error_msg = f"Error in download_sentinel_data_optimized: {e}"
        logger.error(error_msg)
        logger.exception("Detailed error:")
        if failure_logger:
            failure_logger.log_failure(
                cell_id, "download_data_error", str(e), {"year": year, "bands": bands}
            )
        return {}, False


def download_single_band(
    merged_collection,
    band,
    composite_method,
    region,
    width_meters,
    height_meters,
    target_crs,
    session=None,
    early_year: bool = False,
    failure_logger=None,
    cell_id=None,
    max_retries=8,
):
    """
    Download a single band from Earth Engine with improved error handling.

    Args:
        merged_collection: The merged image collection
        band: The band to download
        composite_method: Method for compositing
        region: Earth Engine region geometry
        width_meters: Width in meters
        height_meters: Height in meters
        target_crs: Target CRS
        session: HTTP session to use
        early_year: Whether this is an early year (pre-2017)
        failure_logger: Logger for recording failures
        cell_id: Cell ID for failure logging

    Returns:
        Numpy array containing the band data
    """
    import tempfile
    import requests

    # Get the native resolution for this band
    if early_year:
        scale = 30
    else:
        scale = BAND_RESOLUTION.get(band, 10)  # Default to 10m if unknown

    # Calculate expected dimensions in pixels
    width_pixels = int(width_meters / scale)
    height_pixels = int(height_meters / scale)

    logger.debug(
        f"Processing band {band} at {scale}m resolution, expected dimensions: {width_pixels}x{height_pixels}"
    )

    if session is None:
        session = requests
    retry_delay = 2
    tmp_path = None
    url = None
    for attempt in range(max_retries):
        jitter = random.uniform(0.8, 1.2)
        actual_delay = retry_delay * jitter
        try:
            # Create the composite
            if composite_method == "median":
                band_composite = merged_collection.select(band).median()
            elif composite_method == "mean":
                band_composite = merged_collection.select(band).mean()
            else:
                band_composite = merged_collection.select(band).median()

            # Get download URL with explicit dimensions
            url = band_composite.getDownloadURL(
                {
                    "region": region,  # WGS84 region
                    "dimensions": f"{width_pixels}x{height_pixels}",  # Explicit pixel dimensions
                    "format": "GEO_TIFF",
                    "crs": target_crs,  # Output in UTM
                }
            )
            logger.debug(f"Created download URL for band {band}, attempt {attempt+1}")

            timeout = min(30 + attempt * 60, 150)
            logger.debug(
                f"Download attempt {attempt+1}/{max_retries} for band {band}, timeout={timeout}s"
            )
            response = session.get(url, timeout=timeout)
            if response.status_code != 200:
                error_msg = (
                    f"Failed to download band {band}: HTTP {response.status_code}"
                )
                logger.warning(error_msg)

                if attempt < max_retries - 1:
                    # If we got a 403 or 401, the URL might be expired - force regeneration
                    if response.status_code in (401, 403, 404):
                        url = None  # Force URL regeneration on next attempt

                    logger.warning(f"Retrying in {actual_delay:.1f}s")
                    time.sleep(actual_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Max retries reached for band {band}")
                    if failure_logger and cell_id:
                        failure_logger.log_failure(
                            cell_id,
                            "band_download_http_error",
                            error_msg,
                            {
                                "band": band,
                                "attempt": attempt + 1,
                                "status_code": response.status_code,
                            },
                        )
                    return None
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            # Read the GeoTIFF with rasterio
            with rasterio.open(tmp_path) as src:
                band_array = src.read(1)
                logger.info(f"Downloaded band {band} with shape {band_array.shape}")

                # Verify we got reasonable data
                if band_array.shape[0] < 5 or band_array.shape[1] < 5:
                    error_msg = f"Band {band} has unexpectedly low resolution: {band_array.shape}"
                    logger.error(error_msg)

                    if failure_logger and cell_id:
                        failure_logger.log_failure(
                            cell_id,
                            "band_resolution_error",
                            error_msg,
                            {"band": band, "shape": band_array.shape},
                        )

                    os.unlink(tmp_path)
                    tmp_path = None

                    # Force URL regeneration and retry if not the last attempt
                    if attempt < max_retries - 1:
                        url = None
                        logger.warning(f"Retrying with new URL in {actual_delay:.1f}s")
                        time.sleep(actual_delay)
                        retry_delay *= 2
                        continue
                    return None

            # Remove the temporary file
            if tmp_path:
                os.unlink(tmp_path)
                tmp_path = None

            # Success! Return the band array
            return band_array

        except Exception as e:
            error_msg = (
                f"Error with band {band}, attempt {attempt+1}/{max_retries}: {str(e)}"
            )
            logger.warning(error_msg)

            # Clean up temporary file if it exists
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    tmp_path = None
                except:
                    pass

            # For most errors, we should try regenerating the URL
            url = None

            if attempt < max_retries - 1:
                logger.info(f"Will retry in {actual_delay:.1f}s")
                time.sleep(actual_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(
                    f"Failed to process band {band} after {max_retries} attempts"
                )
                if failure_logger and cell_id:
                    failure_logger.log_failure(
                        cell_id,
                        "band_processing_error",
                        str(e),
                        {"band": band, "attempt": attempt + 1},
                    )
                return None

    # We should never reach here, but just in case
    return None


def calculate_optimal_workers(config):
    """
    Calculate the optimal number of worker threads based on system resources and EE limits,
    with special handling for I/O-bound workloads.
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

    # Memory-based limit - improved calculation
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    usable_memory_percentage = 0.9
    usable_memory_gb = min(
        total_memory_gb * usable_memory_percentage,
        available_memory_gb * 0.95,
    )
    default_memory_per_worker = max(0.5, min(1.5, total_memory_gb / 20))
    memory_per_worker = config.get("memory_per_worker_gb", default_memory_per_worker)
    process_overhead_gb = max(1.0, total_memory_gb * 0.05)
    worker_memory_gb = max(0.1, usable_memory_gb - process_overhead_gb)
    memory_based_limit = max(int(worker_memory_gb / memory_per_worker), 12)

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


def cleanup_processing_files(output_dir, country_name, year):
    """
    Clean up any leftover processing.json files from previous runs.

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

        # Also check for .processing files from the new code
        processing_marker = cell_dir / ".processing"
        if processing_marker.exists():
            try:
                processing_marker.unlink()
                cleaned_count += 1
            except Exception as e:
                logger.warning(
                    f"Error cleaning up .processing file in {cell_dir.name}: {e}"
                )

    logger.debug(
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
        thread_name = threading.current_thread().name
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


def estimate_memory_per_worker(sample_size=3, early_year=False):
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

        # Process a sample cell (simplified version)
        try:
            sample_result = process_sample_cell(early_year=early_year)
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
        estimated_memory = 0.3  # Reduced default

    logger.info(f"Estimated memory per worker: {estimated_memory:.2f} GB")
    return max(0.1, estimated_memory)


def calculate_optimal_cache_size(early_year):
    """
    Calculate optimal LRU cache size based on available memory and measured collection size.

    Returns:
        int: Optimal cache size
    """
    import psutil

    # Get available memory in GB
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Measure actual memory per cached collection
    memory_per_cached_item_mb = measure_collection_memory_usage(early_year=early_year)

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


# Global variable to store the LRU cache size
_OPTIMAL_CACHE_SIZE = 200


def update_lru_cache_size(new_size):
    """Update the global LRU cache size."""
    global _OPTIMAL_CACHE_SIZE
    _OPTIMAL_CACHE_SIZE = new_size
    logger.info(f"Updated LRU cache size to {new_size}")


def measure_collection_memory_usage(sample_size=5, early_year=False):
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

    # Create a list to hold references to collections (so they don't get garbage collected)
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

        sample_bands = [
            ["B2", "B3", "B4"],
            ["B5", "B6", "B7"],
            ["B8", "B8A"],
            ["B11", "B12"],
            ["B2", "B3", "B4", "B8"],
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
            bands = sample_bands[i % len(sample_bands)]

            # Create Earth Engine geometry
            ee_geometry = ee.Geometry.Rectangle(bounds)
            if early_year:
                # Get the collection
                collection = (
                    ee.ImageCollection("NASA/HLS/HLSS30/v002")
                    .filterDate(start_date, end_date)
                    .filterBounds(ee_geometry)
                    .select(bands)
                )
            else:
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterDate(start_date, end_date)
                    .filterBounds(ee_geometry)
                    .select(bands)
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


def process_sample_cell(early_year=False):
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

    # Get a small collection
    if early_year:
        collection = (
            ee.ImageCollection("NASA/HLS/HLSS30/v002")
            .filterDate("2020-01-01", "2020-01-15")
            .filterBounds(geometry)
            .limit(5)
        )
    else:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate("2020-01-01", "2020-01-15")
            .filterBounds(geometry)
            .limit(5)
        )

    # Perform typical operations
    count = collection.size().getInfo()

    if count > 0:
        # Get a composite
        composite = collection.select(["B2", "B3", "B4"]).median()

        # Get some stats
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=100, maxPixels=1e6
        ).getInfo()

        return {"count": count, "stats": stats}
    else:
        # Try another area if no images found
        return {"count": 0, "stats": {}}


def get_monthly_date_ranges(year: int) -> List[Tuple[str, str]]:
    """
    Generate start and end date strings for each month in the given year.

    Args:
        year: The year to generate date ranges for

    Returns:
        List of (start_date, end_date) tuples as strings in 'YYYY-MM-DD' format
    """
    date_ranges = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(year, month)
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{days_in_month:02d}"
        date_ranges.append((start_date, end_date))
    return date_ranges


def save_band_arrays(
    band_arrays: Dict[str, np.ndarray],
    output_dir: Path,
    country_name: str,
    cell_id: int,
    year: int,
    grid_cell: gpd.GeoDataFrame,
    target_crs: str,
    early_year: bool,
) -> None:
    """
    Process and save band arrays using the optimized format.

    Args:
        band_arrays: Dictionary mapping band names to numpy arrays
        output_dir: Directory to save the files
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data
        grid_cell: The GeoDataFrame containing the cell geometry and CRS
        target_crs: Target CRS to convert to before saving
        early_year: Whether this is an early year (pre-2017)
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

        # Get the resolution for this band
        if early_year:
            resolution = 30
        else:
            resolution = BAND_RESOLUTION.get(band_name, 10)

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
    npz_path = process_and_save_bands(
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

    logger.info(f"Processed and saved optimized data for cell {cell_id}")


def save_cell_metadata(
    country_name: str,
    cell_id: int,
    year: int,
    bands: List[str],
    cell_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    composite_method: str,
    cloud_threshold: int,
    band_arrays: Dict[str, np.ndarray],
    early_year: bool,
) -> None:
    """
    Save comprehensive metadata for a processed cell.

    Args:
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data
        bands: List of bands that were processed
        cell_gdf: GeoDataFrame containing the cell geometry
        output_dir: Base output directory
        composite_method: Method used for compositing
        cloud_threshold: Cloud threshold used for filtering
        band_arrays: Dictionary of band arrays (for shape information)
        early_year: Whether this is an early year (pre-2017)
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

    # Categorize the arrays
    indices = [key for key in processed_data.keys() if key in ["ndvi", "built_up"]]
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
        import json

        json.dump(metadata, f, indent=2)

    logger.debug(f"Saved comprehensive metadata to {metadata_file}")


# Monkey patch ee.data methods to count requests
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
    Enhanced monitoring function with complete restart capability when stalled.
    """
    last_request_count = 0
    last_active_time = time.time()
    consecutive_stalls = 0
    recovery_attempts = 0

    # Keep track of when workers were last restarted
    last_worker_restart = time.time()

    while True:
        try:
            current_count = sum(request_counter.counts.values())
            current_time = time.time()

            # Monitor memory usage for leaks
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / (1024 * 1024)

            # Check if requests are being made
            if current_count > last_request_count:
                # Activity detected, reset stall counter
                consecutive_stalls = 0
                recovery_attempts = 0  # Reset recovery attempts on successful activity
                last_active_time = current_time
                last_request_count = current_count
                logger.debug(
                    f"Processing active: {current_count - last_request_count} new requests, "
                    f"Memory: {current_memory:.1f} MB"
                )
            else:
                # No new requests detected
                elapsed = current_time - last_active_time
                if elapsed > 120:  # 2 minutes of inactivity
                    consecutive_stalls += 1
                    logger.warning(
                        f"Processing appears stalled for {elapsed:.1f} seconds (stall #{consecutive_stalls}), "
                        f"Memory: {current_memory:.1f} MB"
                    )

                    # After multiple consecutive stalls, try recovery actions
                    if consecutive_stalls >= 2:
                        recovery_attempts += 1
                        logger.error(
                            f"Processing stalled for {consecutive_stalls} consecutive checks, "
                            f"attempting recovery (attempt #{recovery_attempts})"
                        )

                        try:
                            # Escalating recovery actions based on number of attempts
                            if recovery_attempts == 1:
                                # First attempt: basic recovery
                                perform_basic_recovery()
                            elif recovery_attempts == 2:
                                # Second attempt: intermediate recovery
                                perform_intermediate_recovery()
                            else:
                                # Third+ attempt: COMPLETE RESTART
                                logger.critical(
                                    "CRITICAL STALL DETECTED. Initiating full process restart!"
                                )

                                # Save command line arguments and restart the process
                                python = sys.executable
                                os.execl(python, python, *sys.argv)
                                # This will completely restart the current Python process
                                # No code after this point will execute in the current process

                        except Exception as e:
                            logger.error(f"Error during recovery: {e}")
                            logger.exception("Recovery error details:")

                        # Reset stall counter but not recovery attempts
                        last_active_time = current_time
                        consecutive_stalls = 0

            # Periodic full reset regardless of stalls (every 2 hours)
            if current_time - last_worker_restart > 7200:  # 2 hours
                logger.info("Performing scheduled preventative recovery")
                perform_intermediate_recovery()
                last_worker_restart = current_time

        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")

        time.sleep(30)  # Check every 30 seconds


def perform_basic_recovery():
    """Basic recovery actions for first stall detection."""
    logger.info("Performing basic recovery actions")

    # Force garbage collection
    gc.collect(generation=2)

    # Clear function caches
    if hasattr(get_sentinel_collection_cached, "cache_clear"):
        get_sentinel_collection_cached.cache_clear()

    # Check for and kill any zombie threads
    check_for_zombie_threads(timeout_seconds=300)  # 5 minutes


def perform_intermediate_recovery():
    """Intermediate recovery for repeated stalls."""
    logger.info("Performing intermediate recovery actions")

    # Do basic recovery first
    perform_basic_recovery()

    # Refresh Earth Engine session more aggressively
    try:
        logger.info("Refreshing Earth Engine session")
        ee.Reset()
        time.sleep(3)

        ee.Initialize(
            project="wealth-satellite-forecasting",
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
        setup_request_counting()
    except Exception as e:
        logger.error(f"Error refreshing Earth Engine: {e}")

    # Create new global HTTP session
    try:
        logger.info("Creating new HTTP session with fresh connection pool")
        global_session = create_optimized_session(max_workers=20, use_high_volume=True)

        # Test the new session
        test_response = global_session.get(
            "https://earthengine.googleapis.com", timeout=10
        )
        logger.info(f"New session test response: {test_response.status_code}")
    except Exception as e:
        logger.error(f"Error creating new session: {e}")


def check_for_zombie_threads(timeout_seconds=300):
    """Check for and log any threads that have been running too long."""
    current_time = time.time()
    for thread in threading.enumerate():
        # Skip daemon threads and main thread
        if thread.daemon or thread.name == "MainThread":
            continue

        # Check if thread has thread-local start time
        if (
            hasattr(thread, "_start_time")
            and current_time - thread._start_time > timeout_seconds
        ):
            logger.warning(
                f"Potential zombie thread detected: {thread.name}, "
                f"running for {current_time - thread._start_time:.1f} seconds"
            )


def progress_updater_thread(queue, total, desc, stop_event):
    """
    Thread function to handle progress bar updates from a queue at a consistent rate.

    Args:
        queue: Queue containing progress update increments
        total: Total number of items to process
        desc: Description for the progress bar
        stop_event: Event to signal when the thread should stop
    """
    # Update frequency in seconds
    update_interval = 45

    with tqdm(total=total, desc=desc) as pbar:
        accumulated_progress = 0
        last_update_time = time.time()

        while not stop_event.is_set() or not queue.empty():
            # Process all available items in the queue without blocking
            items_processed = 0

            # Process queue items without blocking
            while True:
                try:
                    # Non-blocking queue get
                    increment = queue.get_nowait()
                    accumulated_progress += increment
                    items_processed += 1
                    queue.task_done()
                except Empty:
                    # No more items in queue
                    break
                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")
                    break

            current_time = time.time()
            time_since_update = current_time - last_update_time

            # Update the progress bar at fixed intervals
            if time_since_update >= update_interval and accumulated_progress > 0:
                pbar.update(accumulated_progress)
                accumulated_progress = 0
                last_update_time = current_time

            # If no items were processed, sleep a bit to avoid CPU spinning
            if items_processed == 0:
                time.sleep(min(0.1, update_interval / 5))

        # Final update to ensure we don't miss any progress
        if accumulated_progress > 0:
            pbar.update(accumulated_progress)


def count_total_images(collections):
    """
    Count the total number of images across all Earth Engine ImageCollections.

    Args:
        collections: List of ee.ImageCollection objects

    Returns:
        int: Total number of images
    """
    total_count = 0

    for collection in collections:
        try:
            # Get the size of each collection
            if isinstance(collection, ee.imagecollection.ImageCollection):
                # Use getInfo to get the actual count from the server
                collection_size = collection.size().getInfo()
                total_count += collection_size
            elif isinstance(collection, list):
                # If it's a list, recursively count its contents
                total_count += count_total_images(collection)
        except Exception as e:
            logger.warning(f"Error counting images in collection: {e}")
            # Continue with next collection rather than failing completely
            continue

    return total_count
