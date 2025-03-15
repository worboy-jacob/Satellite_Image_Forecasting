"""
Image processing module for resampling, compression, and index calculation.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
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

    # Determine if we're upsampling or downsampling
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
    safety_factor = 1.2  # Add 20% to typical max values

    if band_type == "visible":
        # For visible bands (B2, B3, B4)
        min_val = 0
        max_val = 3000 * safety_factor  # 30% reflectance with safety factor
    elif band_type == "nir":
        # For NIR band (B8)
        min_val = 0
        max_val = 5000 * safety_factor  # 50% reflectance with safety factor
    elif band_type == "swir":
        # For SWIR bands (B11, B12)
        min_val = 0
        max_val = 4000 * safety_factor  # 40% reflectance with safety factor
    elif band_type == "ndvi" or band_type == "built_up":
        # Indices are already normalized to [-1, 1]
        # Just clip to ensure they're in range
        image = np.clip(image, -1, 1)

        # Scale from [-1, 1] to [0, 255] for uint8 or [0, 65535] for uint16
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

    # Handle division by zero
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
    Process bands, create composites and indices, and save as compressed NPZ file.

    Args:
        band_arrays: Dictionary of band arrays
        output_dir: Base output directory
        country_name: Name of the country
        cell_id: ID of the grid cell
        year: Year of the data

    Returns:
        Path to the saved NPZ file
    """
    processed_bands = {}
    for band_name, band_array in band_arrays.items():
        if band_array is not None:
            processed_bands[band_name] = resample_to_256x256(band_array)
    # Create output directory
    cell_dir = output_dir / country_name / str(year) / f"cell_{cell_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store processed arrays
    processed_data = {}

    # Check if we have the necessary bands
    required_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    missing_bands = [band for band in required_bands if band not in processed_bands]

    if missing_bands:
        logger.warning(f"Missing bands for cell {cell_id}: {missing_bands}")

    # Process individual bands
    for band_name, band_array in processed_bands.items():
        # Skip if band is None
        if band_array is None:
            continue

        # Resample to 256x256
        resampled = resample_to_256x256(band_array)

        # Store resampled band temporarily
        processed_bands[band_name] = resampled

    # Create RGB composite if possible
    if all(band in processed_bands for band in ["B4", "B3", "B2"]):
        rgb = create_rgb_composite(
            processed_bands["B4"], processed_bands["B3"], processed_bands["B2"]
        )
        # Normalize and quantize RGB to uint8
        if rgb is not None:
            r_normalized = normalize_and_quantize(rgb[:, :, 0], "visible", "uint8")
            g_normalized = normalize_and_quantize(rgb[:, :, 1], "visible", "uint8")
            b_normalized = normalize_and_quantize(rgb[:, :, 2], "visible", "uint8")

            # Recombine
            processed_data["rgb"] = np.stack(
                [r_normalized, g_normalized, b_normalized], axis=-1
            )

    # Calculate NDVI if possible
    if all(band in processed_bands for band in ["B8", "B4"]):
        ndvi = calculate_ndvi(processed_bands["B8"], processed_bands["B4"])
        # Normalize and quantize NDVI to uint8
        if ndvi is not None:
            processed_data["ndvi"] = normalize_and_quantize(ndvi, "ndvi", "uint8")

    # Calculate Built-up Index if possible
    if all(band in processed_bands for band in ["B11", "B8"]):
        bui = calculate_built_up_index(processed_bands["B11"], processed_bands["B8"])
        # Normalize and quantize BUI to uint8
        if bui is not None:
            processed_data["built_up"] = normalize_and_quantize(
                bui, "built_up", "uint8"
            )

    # Process and store individual bands we want to keep
    bands_to_keep = {"B8": "nir", "B11": "swir1", "B12": "swir2"}
    band_type_mapping = {"B8": "nir", "B11": "swir", "B12": "swir"}
    for band_name, output_name in bands_to_keep.items():
        if band_name in processed_bands and processed_bands[band_name] is not None:
            # Normalize and quantize to uint16
            processed_data[output_name] = normalize_and_quantize(
                processed_bands[band_name],
                band_type_mapping.get(band_name, "unknown"),
                "uint16",
            )

    # Save as compressed NPZ file
    if processed_data:
        npz_path = cell_dir / f"processed_data.npz"
        np.savez_compressed(npz_path, **processed_data)
        logger.info(f"Saved processed data to {npz_path}")
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


def normalize_nightlights(
    image: np.ndarray,
    cell_id: int,
    log_min: float = None,
    log_max: float = None,
    data_type: str = "uint8",
) -> np.ndarray:
    """Debug version that traces each step of the normalization process."""

    if image is None:
        return None

    # Handle NaN and Inf values
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure all values are non-negative
    image = np.maximum(image, 0)

    # Apply log transform
    log_values = np.log1p(image)

    # Set normalization parameters
    min_val = 0 if log_min is None else log_min
    max_val = 5 if log_max is None else log_max

    # Normalize to 0-1 range
    normalized = (log_values - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)

    # Scale and convert to target data type
    if data_type == "uint8":
        result = (normalized * 255).astype(np.uint8)
    elif data_type == "uint16":
        result = (normalized * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    final_min = np.min(result)
    final_max = np.max(result)
    final_unique = len(np.unique(result))
    if final_unique < 100:
        logger.info(
            f"Cell {cell_id} - normalize_nightlights - Final output: min={final_min}, max={final_max}, unique={final_unique}"
        )

    return result


def calculate_nightlight_gradient(nightlights: np.ndarray) -> np.ndarray:
    """
    Calculate the spatial gradient of nightlight intensity using Sobel operator.
    """
    import cv2

    # Debug check
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


def normalize_gradient(
    gradient: np.ndarray,
    grad_min: float = None,
    grad_max: float = None,
    data_type: str = "uint8",
) -> np.ndarray:
    """
    Normalize gradient values using global min/max values.
    """
    if gradient is None:
        return None

    # Debug check
    debug_unique_before = len(np.unique(gradient))

    # Handle NaN and Inf values
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    # Use provided global min/max if available, otherwise use sensible defaults
    min_val = (
        0.0 if grad_min is None else grad_min
    )  # Always use 0 as minimum for gradients
    max_val = 200 if grad_max is None else grad_max

    # Normalize to 0-1 range
    normalized = (
        (gradient - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(gradient)
    )
    normalized = np.clip(normalized, 0, 1)

    # Debug check after normalization
    debug_unique_after_norm = len(np.unique(normalized))

    # Scale and convert to target data type
    if data_type == "uint8":
        result = (normalized * 255).astype(np.uint8)
    elif data_type == "uint16":
        result = (normalized * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Debug check after conversion
    debug_unique_final = len(np.unique(result))

    # Log if significant reduction in unique values
    if debug_unique_final < 20 and debug_unique_before > 100:
        logger.warning(
            f"normalize_gradient: Significant reduction in unique values: "
            f"before={debug_unique_before}, after_norm={debug_unique_after_norm}, "
            f"final={debug_unique_final}"
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
            # Store a reference and remove from band_arrays to free memory
            avg_rad_array = band_arrays["avg_rad"].copy()
            band_arrays["avg_rad"] = None  # Clear original reference

            # Resample to 256x256
            resampled = resample_to_256x256(avg_rad_array)
            del avg_rad_array  # Free memory

            processed_data["nightlights"] = normalize_nightlights(
                resampled, cell_id, log_min=0, log_max=5, data_type="uint16"
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
                processed_data["gradient"] = normalize_gradient(
                    grad_resampled,
                    grad_min=0,
                    grad_max=200,  # Fixed parameter
                    data_type="uint16",
                )
                # Free memory
                del grad_resampled
                gc.collect()

            else:
                # Calculate gradient from the resampled nightlights
                nightlights_float = resampled.astype(np.float32)
                gradient = calculate_nightlight_gradient(nightlights_float)

                # Normalize with global values
                processed_data["gradient"] = normalize_gradient(
                    gradient, grad_min=0, grad_max=200, data_type="uint16"
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
            logger.info(f"Saved processed VIIRS data to {npz_path}")
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
        # Make sure to clean up memory even if an error occurs
        for key in list(band_arrays.keys()):
            band_arrays[key] = None
        if "processed_data" in locals():
            for key in list(processed_data.keys()):
                processed_data[key] = None
            processed_data.clear()
        gc.collect()


def debug_image_properties(
    cell_id, stage, image, description, output_dir=None, country_name=None, year=None
):
    """
    Debug helper to analyze image properties at different stages of processing.

    Args:
        cell_id: ID of the cell being processed
        stage: Processing stage name (for logging)
        image: The image array to analyze
        description: Description of what this image represents
        output_dir, country_name, year: Optional parameters to save histogram images
    """
    if image is None:
        logger.warning(f"Cell {cell_id}: {stage} - {description} is None")
        return

    # Basic statistics
    img_min = np.min(image)
    img_max = np.max(image)
    img_mean = np.mean(image)
    img_std = np.std(image)

    # Count unique values
    unique_values = np.unique(image)
    num_unique = len(unique_values)

    # Check for suspicious quantization
    is_quantized = (
        num_unique < 20
    )  # Arbitrary threshold for what we consider "quantized"

    # Log the information
    logger.info(
        f"Cell {cell_id}: {stage} - {description} "
        f"shape={image.shape}, dtype={image.dtype}, "
        f"range=[{img_min:.3f}, {img_max:.3f}], "
        f"mean={img_mean:.3f}, std={img_std:.3f}, "
        f"unique_values={num_unique}"
    )

    if is_quantized:
        # Log the actual unique values if there aren't too many
        if num_unique < 30:
            logger.warning(
                f"Cell {cell_id}: {stage} - {description} appears QUANTIZED! "
                f"Only {num_unique} unique values: {unique_values}"
            )
        else:
            logger.warning(
                f"Cell {cell_id}: {stage} - {description} appears QUANTIZED! "
                f"Only {num_unique} unique values"
            )

    # Save histogram if we have directory information
    if (
        is_quantized
        and output_dir is not None
        and country_name is not None
        and year is not None
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure

            # Create debug directory
            debug_dir = (
                output_dir / country_name / str(year) / "debug" / f"cell_{cell_id}"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Create histogram
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.hist(image.flatten(), bins=min(100, num_unique * 2))
            ax.set_title(f"Cell {cell_id}: {stage} - {description} histogram")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            # Save figure
            fig.savefig(
                debug_dir / f"{stage}_{description.replace(' ', '_')}_histogram.png"
            )

            # Create image visualization
            fig2 = Figure(figsize=(8, 8))
            ax2 = fig2.add_subplot(111)
            im = ax2.imshow(image, cmap="gray")
            ax2.set_title(f"Cell {cell_id}: {stage} - {description}")
            fig2.colorbar(im)

            # Save figure
            fig2.savefig(
                debug_dir / f"{stage}_{description.replace(' ', '_')}_image.png"
            )

            logger.info(f"Saved debug visualizations to {debug_dir}")
        except Exception as e:
            logger.error(f"Failed to save debug visualizations: {e}")
