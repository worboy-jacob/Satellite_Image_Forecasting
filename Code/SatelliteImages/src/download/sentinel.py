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
    save_metadata,
    cleanup_original_files,
)

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


def download_sentinel_for_country_year(
    config: Dict[str, Any], country_name: str, year: int, grid_gdf: gpd.GeoDataFrame
) -> None:
    """
    Download Sentinel-2 data for all cells of a specific country-year pair using optimized parallel processing.
    """
    # Initialize Earth Engine with appropriate endpoint
    initialize_earth_engine(config)
    early_year = False
    if year < 2017:
        early_year = True
    # Check if using high-volume endpoint
    use_high_volume = config.get("use_high_volume_endpoint", True)

    # Get the output directory
    output_dir = get_results_dir() / "Images" / "Sentinel"

    # Clean up any leftover processing files
    cleanup_processing_files(output_dir, country_name, year)

    # Get country CRS from config
    country_crs = None
    for country in config.get("countries", []):
        if country.get("name") == country_name:
            country_crs = country.get("crs")
            break

    if not country_crs:
        logger.error(f"No CRS defined for country {country_name}")
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

    # Use 70% of available memory, divided by memory per cell
    memory_based_batch_size = max(
        10, int((available_memory_gb * 0.8) / memory_per_cell_gb)
    )

    # Cap batch size at a reasonable maximum
    batch_size = min(memory_based_batch_size, 500, len(cells_to_process))

    logger.info(
        f"Processing cells in batches of {batch_size} (memory-based calculation)"
    )
    with tqdm(
        total=len(cells_to_process),
        desc=f"Processing {country_name} {year}",
    ) as pbar:
        # Process in batches to avoid overwhelming memory
        for batch_start in range(0, len(cells_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(cells_to_process))
            current_batch = cells_to_process[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//batch_size + 1}/{(len(cells_to_process)-1)//batch_size + 1} "
                f"({len(current_batch)} cells)"
            )

            # Use a ThreadPoolExecutor for better control over concurrency
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
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
                    ): (idx, cell)
                    for idx, cell in current_batch
                }

                # Process completed tasks as they finish
                for future in concurrent.futures.as_completed(future_to_cell):
                    cell_id, success = future.result()
                    pbar.update(1)

                    if not success:
                        logger.warning(f"Failed to process cell {cell_id}")

            # Force garbage collection between batches
            gc.collect()

    logger.info(f"Completed processing for {country_name}, year {year}")


def create_optimized_session(max_workers=None, use_high_volume=True):
    """
    Create an optimized requests session with connection pooling based on system capabilities.

    Args:
        max_workers: Number of workers that will use this session
        use_high_volume: Whether the high-volume endpoint is being used

    Returns:
        requests.Session: Optimized session
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import psutil

    # Create a session
    session = requests.Session()

    # Configure retry strategy based on endpoint
    if use_high_volume:
        # More retries and longer backoff for high-volume endpoint
        retry_strategy = Retry(
            total=8,  # More retries
            backoff_factor=1.0,  # Longer backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        # Longer timeouts for high-volume endpoint
        timeout = (30, 300)  # 30s connect, 5min read
    else:
        # Standard endpoint settings
        retry_strategy = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        timeout = (15, 180)  # 15s connect, 3min read

    # Calculate optimal connection pool size
    if max_workers is None:
        # Estimate based on CPU count
        cpu_count = psutil.cpu_count(logical=True)
        max_workers = cpu_count * 2

    # Each worker might have multiple concurrent requests
    connections_per_worker = 4 if use_high_volume else 3

    # Calculate pool sizes with a minimum and maximum
    pool_connections = min(max(20, max_workers), 150 if use_high_volume else 100)
    pool_maxsize = min(
        max(50, max_workers * connections_per_worker), 300 if use_high_volume else 200
    )

    logger.info(
        f"HTTP connection pool: connections={pool_connections}, max_size={pool_maxsize}"
    )

    # Configure connection pooling
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
    else:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    # Filter by date and location
    collection = collection.filterDate(start_date, end_date)
    collection = collection.filterBounds(ee_geometry)

    # Filter by cloud cover
    collection = collection.filter(
        ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)
    )

    def mask_s2_clouds(image):
        scl = image.select("SCL")
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(mask)

    # Apply cloud masking
    collection = collection.map(mask_s2_clouds)

    # Select bands if specified
    if bands and bands != "None":
        collection = collection.select(bands)

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
):
    """
    Optimized version of process_sentinel_cell.

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

    Returns:
        Tuple of (cell_id, success_flag)
    """
    cell_id = cell["cell_id"]
    cell_gdf = grid_gdf.iloc[[idx]]

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
        band_arrays = download_sentinel_data_optimized(
            grid_cell=cell_gdf,
            year=year,
            bands=bands,
            cloud_threshold=cloud_threshold,
            composite_method=composite_method,
            target_crs=target_crs,
            session=session,
            early_year=early_year,
        )

        if not band_arrays:
            logger.warning(
                f"No data downloaded for {country_name} cell {cell_id} in {year}"
            )
            # Clean up placeholder
            if placeholder_file.exists():
                placeholder_file.unlink()
            return cell_id, False

        # Save the data
        save_band_arrays(
            band_arrays=band_arrays,
            output_dir=output_dir,
            country_name=country_name,
            cell_id=cell_id,
            year=year,
            grid_cell=cell_gdf,
            target_crs=target_crs,
            early_year=early_year,
        )

        # Save metadata
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
            early_year=early_year,
        )

        # Remove placeholder file
        if placeholder_file.exists():
            placeholder_file.unlink()

        logger.info(
            f"Successfully processed cell {cell_id} for {country_name} in {year}"
        )
        memory_diff = measure_memory_usage(before=False, cell_id=cell_id)
        logger.info(f"Memory used for cell {cell_id}: {memory_diff:.4f} MB")
        return cell_id, True

    except Exception as e:
        logger.error(f"Error processing cell {cell_id}: {str(e)}")
        logger.exception(f"Detailed error for cell {cell_id}:")
        return cell_id, False


def download_sentinel_data_optimized(
    grid_cell: gpd.GeoDataFrame,
    year: int,
    bands: List[str],
    cloud_threshold: int = 20,
    composite_method: str = "median",
    images_per_month: int = 5,
    total_target: int = 60,
    target_crs: str = "EPSG:32628",
    session=None,
    early_year: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Optimized version of download_sentinel_data.

    Args:
        grid_cell: GeoDataFrame containing a single grid cell
        year: Year to download data for
        bands: List of bands to download
        cloud_threshold: Maximum cloud cover percentage
        composite_method: Method for compositing images
        images_per_month: Target number of images per month
        total_target: Overall target number of images
        target_crs: Target CRS
        session: HTTP session to use for downloads

    Returns:
        Dictionary mapping band names to numpy arrays
    """
    try:
        # Get monthly date ranges for the year
        date_ranges = get_monthly_date_ranges(year)

        # Collect images from each month - use optimized collection retrieval
        monthly_counts = []
        selected_collections = []
        total_selected = 0

        # First pass: get counts and select top images for each month
        for month_idx, (start_date, end_date) in enumerate(date_ranges):
            month_num = month_idx + 1

            # Get collection for this month using optimized function
            collection = get_sentinel_collection_optimized(
                grid_cell=grid_cell,
                start_date=start_date,
                end_date=end_date,
                cloud_threshold=cloud_threshold,
                bands=bands,
                early_year=early_year,
            )

            # Get the count of images for this month
            count = collection.size().getInfo()
            monthly_counts.append(count)

            # Select top images by cloud cover
            if count > 0:
                # Sort by cloud cover (ascending)
                sorted_collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE")

                # Take up to images_per_month images
                month_selection = sorted_collection.limit(images_per_month)
                month_selection_count = min(count, images_per_month)

                selected_collections.append(month_selection)
                total_selected += month_selection_count

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
                sorted_collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE")

                # Skip the first images_per_month images (already selected)
                # and take up to the number needed
                to_take = min(extras, additional_needed - additional_count)

                if to_take > 0:
                    additional = sorted_collection.toList(count).slice(
                        images_per_month, images_per_month + to_take
                    )
                    additional_collection = ee.ImageCollection(additional)
                    additional_collections.append(additional_collection)
                    additional_count += to_take

            # Add additional collections to selected collections
            selected_collections.extend(additional_collections)

        # Merge all selected collections
        if not selected_collections:
            logger.warning(f"No images selected for {year}")
            return None

        merged_collection = selected_collections[0]
        for collection in selected_collections[1:]:
            merged_collection = merged_collection.merge(collection)

        # Get the total number of images
        total_images = merged_collection.size().getInfo()
        logger.info(f"Final image count for {year}: {total_images}")

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
            max_workers=min(len(bands), 9)
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
                )
                future_to_band[future] = band

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_band):
                band = future_to_band[future]
                try:
                    band_array = future.result()
                    if band_array is not None:
                        band_arrays[band] = band_array
                except Exception as e:
                    logger.error(f"Error downloading band {band}: {e}")

        return band_arrays

    except Exception as e:
        logger.error(f"Error in download_sentinel_data_optimized: {e}")
        logger.exception("Detailed error:")
        return {}


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
):
    """
    Download a single band from Earth Engine.

    Args:
        merged_collection: The merged image collection
        band: The band to download
        composite_method: Method for compositing
        region: Earth Engine region geometry
        width_meters: Width in meters
        height_meters: Height in meters
        target_crs: Target CRS
        session: HTTP session to use

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

    # Use the provided session or create a new one
    if session is None:
        session = requests

    # Download with retries
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=300)  # 5-minute timeout

            if response.status_code != 200:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to download band {band}: HTTP {response.status_code}, retrying in {retry_delay}s"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(
                        f"Failed to download band {band}: HTTP {response.status_code}"
                    )
                    return None

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Read the GeoTIFF with rasterio
            with rasterio.open(tmp_path) as src:
                band_array = src.read(1)
                logger.info(f"Downloaded band {band} with shape {band_array.shape}")

                # Verify we got the expected resolution
                if band_array.shape[0] < 10 or band_array.shape[1] < 10:
                    logger.error(
                        f"Band {band} has unexpectedly low resolution: {band_array.shape}"
                    )
                    os.unlink(tmp_path)
                    return None

            # Remove the temporary file
            os.unlink(tmp_path)

            return band_array

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Error downloading band {band}, attempt {attempt+1}/{max_retries}: {e}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(
                    f"Failed to download band {band} after {max_retries} attempts: {e}"
                )
                return None

    return None


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

    # Standard endpoint
    ee_rate_limit = 6000 / 60
    max_concurrent = 40

    # Use measured requests per cell if available, otherwise use a conservative estimate
    avg_requests = request_counter.get_average()
    requests_per_cell = max(avg_requests, 5) if avg_requests > 0 else 20

    # Calculate rate-limited workers
    rate_limited_workers = int(ee_rate_limit / requests_per_cell)

    # CPU-based limit
    cpu_count = psutil.cpu_count(logical=True)
    cpu_based_limit = max(cpu_count * 2, 4)  # At least 4 workers

    # Memory-based limit
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    memory_per_worker = config.get("memory_per_worker_gb", 1.0)
    memory_based_limit = max(int(available_memory_gb / memory_per_worker), 4)

    # Network-based limit
    network_based_limit = max_concurrent  # Based on endpoint

    # Take the minimum of all limits
    optimal_limit = min(
        cpu_based_limit, memory_based_limit, network_based_limit, rate_limited_workers
    )

    logger.info(
        f"Calculated worker limits: CPU={cpu_based_limit}, Memory={memory_based_limit}, "
        f"Network={network_based_limit}, EE Rate={rate_limited_workers} "
        f"(based on {requests_per_cell:.1f} requests/cell)"
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
            # Use a small, representative workload
            # This is a placeholder - in practice, you'd process an actual cell
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
        # Add 50% safety margin
        estimated_memory = avg_increase * 1.5
    else:
        # Default if measurement fails
        estimated_memory = 0.5  # 500MB default

    logger.info(f"Estimated memory per worker: {estimated_memory:.2f} GB")
    return max(0.1, estimated_memory)  # At least 100MB


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
    Save metadata for a processed cell.

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
    if npz_path.exists():
        # Load processed data to get array information
        with np.load(npz_path) as data:
            processed_data = {key: data[key] for key in data.files}

        # Save detailed metadata about processed data
        save_metadata(
            npz_path=npz_path,
            country_name=country_name,
            cell_id=cell_id,
            year=year,
            processed_data=processed_data,
        )

    # Get cell centroid in WGS84 for coordinates
    cell_wgs84 = cell_gdf.to_crs("EPSG:4326")
    centroid = cell_wgs84.geometry.iloc[0].centroid

    # Get cell bounds in original CRS
    bounds = cell_gdf.total_bounds

    # Create metadata dictionary
    metadata = {
        "country": country_name,
        "cell_id": int(cell_id),
        "year": year,
        "processed_date": datetime.now().isoformat(),
        "bands": bands,
        "available_bands": list(band_arrays.keys()),
        "composite_method": composite_method,
        "cloud_threshold": cloud_threshold,
        "coordinates": {"latitude": centroid.y, "longitude": centroid.x},
        "bounds": {
            "minx": float(bounds[0]),
            "miny": float(bounds[1]),
            "maxx": float(bounds[2]),
            "maxy": float(bounds[3]),
        },
        "crs": cell_gdf.crs.to_string(),
        "band_shapes": {band: band_arrays[band].shape for band in band_arrays},
        "band_resolutions": {
            band: 30 if early_year else BAND_RESOLUTION.get(band, 10)
            for band in band_arrays
        },
        "processed_data_path": str(npz_path) if npz_path.exists() else None,
    }

    # Save metadata to JSON file
    metadata_file = cell_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        import json

        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")


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
