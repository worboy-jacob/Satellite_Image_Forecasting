"""
Image processing utilities for satellite imagery handling.

Provides functions for resampling, normalization, index calculation, and data
compression to prepare satellite imagery for machine learning applications.
"""

import numpy as np
import logging
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import gc

logger = logging.getLogger("image_processing")


def resample_to_256x256(image: np.ndarray, method: str = "bicubic") -> np.ndarray:
    """
    Resample an image to 256x256 pixels using the specified method.

    Args:
        image: Input image array
        method: Resampling method ('bicubic' for downsampling, 'lanczos' for upsampling)

    Returns:
        Resampled image as numpy array
    """
    if image is None:
        return None

    # Check if already 256x256
    if image.shape == (256, 256):
        return image

    import cv2

    # Select appropriate interpolation method based on input size
    if image.shape[0] > 256 or image.shape[1] > 256:
        # Downsampling - use bicubic
        interpolation = cv2.INTER_CUBIC
    else:
        # Upsampling - use lanczos
        interpolation = cv2.INTER_LANCZOS4

    # Resample the image
    resampled = cv2.resize(image, (256, 256), interpolation=interpolation)

    return resampled


def normalize_and_quantize(
    image: np.ndarray, band_type: str, data_type: str = "uint8"
) -> np.ndarray:
    """
    Normalize image using fixed scaling based on expected value ranges, then quantize.

    Uses predefined value ranges appropriate for each band type to ensure
    consistent scaling across different cells and regions.

    Args:
        image: Input image array
        band_type: Type of band ('visible', 'nir', 'swir', 'ndvi', 'built_up')
        data_type: Target data type ('uint8' or 'uint16')

    Returns:
        Normalized and quantized image
    """
    if image is None:
        return None

    # Handle NaN and Inf values
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Define scaling parameters based on band type
    scaling_params = {
        "visible": (0, 3600),  # 30% reflectance with 20% safety factor
        "nir": (0, 6000),  # 50% reflectance with 20% safety factor
        "swir": (0, 4800),  # 40% reflectance with 20% safety factor
        "ndvi": (-1, 1),  # Already normalized
        "built_up": (-1, 1),  # Already normalized
    }

    if band_type in scaling_params:
        min_val, max_val = scaling_params[band_type]

        # Special handling for indices
        if band_type in ["ndvi", "built_up"]:
            image = np.clip(image, min_val, max_val)
            if data_type == "uint8":
                return ((image + 1) * 127.5).astype(np.uint8)
            else:
                return ((image + 1) * 32767.5).astype(np.uint16)
    else:
        logger.warning(f"Unknown band type '{band_type}', using data min/max")
        min_val = np.min(image)
        max_val = np.max(image)

    # Clip to valid range for raw bands
    image = np.clip(image, min_val, max_val)

    # Normalize to 0-1 range
    normalized = (
        (image - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(image)
    )

    # Scale and convert to target data type
    if data_type == "uint8":
        return (normalized * 255).astype(np.uint8)
    elif data_type == "uint16":
        return (normalized * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).

    Args:
        nir: Near-infrared band (B8)
        red: Red band (B4)

    Returns:
        NDVI array
    """
    if nir is None or red is None:
        return None

    # Create masked calculation to prevent division by zero
    denominator = nir + red
    ndvi = np.zeros(nir.shape, dtype=np.float32)
    valid = denominator != 0
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]

    # NDVI ranges from -1 to 1
    return np.clip(ndvi, -1, 1)


def calculate_built_up_index(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate the Built-up Index (BUI) using SWIR1 and NIR bands.

    Args:
        swir1: Short-wave infrared band (B11)
        nir: Near-infrared band (B8)

    Returns:
        Built-up index array
    """
    if swir1 is None or nir is None:
        return None

    # Handle division by zero
    denominator = swir1 + nir
    bui = np.zeros(swir1.shape, dtype=np.float32)
    valid = denominator != 0
    bui[valid] = (swir1[valid] - nir[valid]) / denominator[valid]

    # BUI typically ranges from -1 to 1
    return np.clip(bui, -1, 1)


def create_rgb_composite(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray
) -> np.ndarray:
    """
    Create an RGB composite from individual bands.

    Args:
        red: Red band (B4)
        green: Green band (B3)
        blue: Blue band (B2)

    Returns:
        RGB composite array with shape (256, 256, 3)
    """
    if red is None or green is None or blue is None:
        return None

    # Stack the bands
    rgb = np.stack([red, green, blue], axis=-1)

    return rgb


def process_and_save_bands(
    band_arrays: Dict[str, np.ndarray],
    output_dir: Path,
    country_name: str,
    cell_id: int,
    year: int,
) -> Path:
    """
    Process Sentinel bands, create composites and indices, and save as compressed NPZ file.

    Handles resampling to a uniform size, normalization, and calculation of derived
    products like vegetation indices and RGB composites.

    Args:
        band_arrays: Dictionary of band arrays
        output_dir: Base output directory
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data

    Returns:
        Path to the saved NPZ file
    """
    # Create output directory
    cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store processed arrays
    processed_data = {}

    # First, resample all bands to 256x256
    processed_bands = {}
    for band_name, band_array in band_arrays.items():
        if band_array is not None:
            processed_bands[band_name] = resample_to_256x256(band_array)

    # Check for required bands
    required_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    missing_bands = [band for band in required_bands if band not in processed_bands]

    if missing_bands:
        logger.warning(f"Missing bands for cell {cell_id}: {missing_bands}")

    # Process RGB composite
    if all(band in processed_bands for band in ["B4", "B3", "B2"]):
        rgb = create_rgb_composite(
            processed_bands["B4"], processed_bands["B3"], processed_bands["B2"]
        )
        if rgb is not None:
            # Process each channel separately
            channels = []
            for i, band_type in enumerate(["visible", "visible", "visible"]):
                channel = normalize_and_quantize(rgb[:, :, i], band_type, "uint8")
                channels.append(channel)

            # Recombine channels
            processed_data["rgb"] = np.stack(channels, axis=-1)

    # Process indices
    if all(band in processed_bands for band in ["B8", "B4"]):
        ndvi = calculate_ndvi(processed_bands["B8"], processed_bands["B4"])
        if ndvi is not None:
            processed_data["ndvi"] = normalize_and_quantize(ndvi, "ndvi", "uint8")

    if all(band in processed_bands for band in ["B11", "B8"]):
        bui = calculate_built_up_index(processed_bands["B11"], processed_bands["B8"])
        if bui is not None:
            processed_data["built_up"] = normalize_and_quantize(
                bui, "built_up", "uint8"
            )

    # Process individual bands to keep
    bands_to_keep = {"B8": "nir", "B11": "swir1", "B12": "swir2"}
    band_type_mapping = {"B8": "nir", "B11": "swir", "B12": "swir"}

    for band_name, output_name in bands_to_keep.items():
        if band_name in processed_bands and processed_bands[band_name] is not None:
            band_type = band_type_mapping.get(band_name, "unknown")
            processed_data[output_name] = normalize_and_quantize(
                processed_bands[band_name], band_type, "uint16"
            )

    # Save results
    if processed_data:
        npz_path = cell_dir / "processed_data.npz"
        np.savez_compressed(npz_path, **processed_data)
        logger.debug(f"Saved processed Sentinel-2 data to {npz_path}")
        return npz_path
    else:
        logger.warning(f"No processed data to save for cell {cell_id}")
        return None


def cleanup_original_files(
    output_dir: Path, country_name: str, cell_id: int, year: int
) -> None:
    """
    Clean up original band files after processing.

    Args:
        output_dir: Base output directory (should be cell directory)
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data
    """
    # This should be the cell directory already
    cell_dir = output_dir
    original_dir = cell_dir / "original"

    # Check if the directory exists
    if not original_dir.exists():
        logger.debug(f"No original directory found for cell {cell_id}")
        return

    # Find all .tif files
    tif_files = list(original_dir.glob("*.tif"))

    if not tif_files:
        logger.debug(f"No TIF files found in {original_dir}")
        return

    # Delete them
    deleted_count = 0
    for file in tif_files:
        try:
            file.unlink()
            deleted_count += 1
            logger.debug(f"Deleted original file: {file}")
        except Exception as e:
            logger.warning(f"Failed to delete {file}: {e}")

    # Try to remove the empty directory
    try:
        if not list(original_dir.iterdir()):
            original_dir.rmdir()
            logger.debug(f"Removed empty original directory for cell {cell_id}")
    except Exception as e:
        logger.warning(f"Could not remove original directory: {e}")

    logger.debug(f"Cleaned up {deleted_count} original files for cell {cell_id}")


def calculate_nightlight_gradient(nightlights: np.ndarray) -> np.ndarray:
    """
    Calculate the spatial gradient of nightlight intensity using Sobel operator.
    """
    import cv2

    # Capture initial state for quality control
    debug_unique_before = len(np.unique(nightlights))
    if debug_unique_before < 20:
        logger.warning(
            f"calculate_nightlight_gradient: Input has only {debug_unique_before} unique values!"
        )

    # Handle NaN values
    nightlights = np.nan_to_num(nightlights, nan=0.0)

    # Convert to float32 for better precision
    nightlights_float = nightlights.astype(np.float32)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(nightlights_float, (3, 3), 0)

    # Debug check after blur
    debug_unique_after_blur = len(np.unique(blurred))

    # Calculate gradients using Sobel operator
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient = cv2.magnitude(grad_x, grad_y)

    # Debug check after gradient
    debug_unique_after_grad = len(np.unique(gradient))

    # Log if something unusual happens
    if debug_unique_after_grad < 20 and debug_unique_before > 100:
        logger.warning(
            f"calculate_nightlight_gradient: Unusual reduction in unique values: "
            f"before={debug_unique_before}, after_blur={debug_unique_after_blur}, "
            f"after_grad={debug_unique_after_grad}"
        )

    return gradient


def normalize_array(
    image: np.ndarray,
    min_val: float = 0,
    max_val: float = 1,
    log_transform: bool = False,
    data_type: str = "uint8",
    cell_id: Optional[int] = None,
    debug_name: str = "array",
) -> np.ndarray:
    """
    Normalize array using specified min/max values with optional log transform.

    Generic normalization function that can handle both nightlights and gradients
    with appropriate parameters.

    Args:
        image: Input image array
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling
        log_transform: Whether to apply log(x+1) transform before normalizing
        data_type: Target data type ('uint8' or 'uint16')
        cell_id: Optional cell ID for logging
        debug_name: Name to use in debug messages

    Returns:
        Normalized and quantized array
    """
    if image is None:
        return None

    # Handle NaN and Inf values
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure non-negative values if applying log transform
    if log_transform:
        image = np.maximum(image, 0)
        transformed = np.log1p(image)
    else:
        transformed = image

    # Normalize to 0-1 range
    normalized = (transformed - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)

    # Scale and convert to target data type
    if data_type == "uint8":
        result = (normalized * 255).astype(np.uint8)
    elif data_type == "uint16":
        result = (normalized * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Debug logging for low-variance outputs
    if cell_id is not None:
        final_min = np.min(result)
        final_max = np.max(result)
        final_unique = len(np.unique(result))
        if final_unique < 100:
            logger.info(
                f"Cell {cell_id} - normalize_{debug_name} - Final output: "
                f"min={final_min}, max={final_max}, unique={final_unique}"
            )

    return result


def process_and_save_viirs_bands(
    band_arrays: Dict[str, np.ndarray],
    output_dir: Path,
    country_name: str,
    cell_id: int,
    year: int,
) -> Path:
    """
    Process VIIRS bands and save as compressed NPZ file.

    Args:
        band_arrays: Dictionary of band arrays
        output_dir: Base output directory
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data

    Returns:
        Path to the saved NPZ file
    """
    # Create output directory
    cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store processed arrays
    processed_data = {}

    try:
        # Process nightlights if available
        avg_rad_array = None
        if "avg_rad" in band_arrays and band_arrays["avg_rad"] is not None:
            # Create a copy and clear original reference to manage memory usage
            avg_rad_array = band_arrays["avg_rad"].copy()
            band_arrays["avg_rad"] = None  # Clear original reference

            # Resample to 256x256
            resampled = resample_to_256x256(avg_rad_array)
            del avg_rad_array  # Free memory

            processed_data["nightlights"] = normalize_array(
                resampled,
                min_val=0,
                max_val=5,
                log_transform=True,
                data_type="uint16",
                cell_id=cell_id,
                debug_name="nightlights",
            )
            # Process gradient if available
            gradient_array = None
            if "gradient" in band_arrays and band_arrays["gradient"] is not None:
                # Store a reference and remove from band_arrays
                gradient_array = band_arrays["gradient"].copy()

                band_arrays["gradient"] = None  # Clear original reference

                # Resample to 256x256
                grad_resampled = resample_to_256x256(gradient_array)
                del gradient_array  # Free memory

                # Normalize and quantize using global values
                processed_data["gradient"] = normalize_array(
                    gradient,
                    min_val=0,
                    max_val=200,
                    log_transform=False,
                    data_type="uint16",
                    cell_id=cell_id,
                    debug_name="gradient",
                )
                # Free memory
                del grad_resampled
                gc.collect()

            else:
                # Calculate gradient from the resampled nightlights
                nightlights_float = resampled.astype(np.float32)
                gradient = calculate_nightlight_gradient(nightlights_float)

                # Normalize with global values
                processed_data["gradient"] = normalize_array(
                    gradient,
                    min_val=0,
                    max_val=200,
                    log_transform=False,
                    data_type="uint16",
                    cell_id=cell_id,
                    debug_name="gradient",
                )

                # Free memory
                del gradient
                gc.collect()

            # Clear resampled to free memory
            del resampled
            gc.collect()

        # Save as compressed NPZ file
        if processed_data:

            npz_path = cell_dir / f"processed_data.npz"
            np.savez_compressed(npz_path, **processed_data)
            logger.debug(f"Saved processed VIIRS data to {npz_path}")
            for key in list(processed_data.keys()):
                processed_data[key] = None
            processed_data.clear()

            gc.collect()

            return npz_path
        else:
            logger.warning(f"No processed data to save for cell {cell_id}")
            return None
    except Exception as e:
        logger.error(f"Error in process_and_save_viirs_bands: {e}")
        logger.exception("Detailed error:")
        return None
    finally:
        # Ensure memory is released regardless of processing outcome
        for key in list(band_arrays.keys()):
            band_arrays[key] = None
        if "processed_data" in locals():
            for key in list(processed_data.keys()):
                processed_data[key] = None
            processed_data.clear()
        gc.collect()
