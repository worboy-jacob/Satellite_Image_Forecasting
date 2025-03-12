"""
Processing module for satellite image data.

This package contains modules for processing, transforming, and
optimizing satellite imagery for machine learning applications.
"""

# Import key functions for easier access
from src.processing.resampling import (
    resample_to_256x256,
    normalize_and_quantize,
    calculate_ndvi,
    calculate_built_up_index,
    create_rgb_composite,
    process_and_save_bands,
    cleanup_original_files,
)

__all__ = [
    "resample_to_256x256",
    "normalize_and_quantize",
    "calculate_ndvi",
    "calculate_built_up_index",
    "create_rgb_composite",
    "process_and_save_bands",
    "cleanup_original_files",
]
