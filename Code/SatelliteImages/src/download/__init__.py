"""
Download module for satellite imagery.

This package contains modules for downloading satellite imagery
from various sources, including Sentinel-2 and VIIRS.
"""

# Import key functions for easier access
from src.download.sentinel import (
    download_sentinel_for_country_year,
    process_sentinel_cell_optimized,
    download_sentinel_data_optimized,
)

__all__ = [
    "download_sentinel_for_country_year",
    "process_sentinel_cell_optimized",
    "download_sentinel_data_optimized",
]
