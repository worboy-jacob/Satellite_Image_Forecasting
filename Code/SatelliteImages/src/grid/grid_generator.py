"""
Grid generation utilities for creating and managing geospatial grids.

Provides functionality to create uniform grid cells over country boundaries
for consistent spatial analysis and satellite image processing.
"""

import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from shapely.geometry import box
from typing import Optional, Union, List

from src.utils.paths import find_shapefile, get_results_dir

logger = logging.getLogger("image_processing")


def create_grid(
    country_name: str,
    cell_size_km: float,
    target_crs: str,
    min_area_percent: float = 40.0,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Create a grid of uniform cells over a country boundary.

    Generates a grid of rectangular cells aligned to the country's bounding box,
    then filters cells based on their overlap with the actual country boundary.
    Each cell is assigned a unique sequential ID for reference.

    Args:
        country_name: Name of the country to create grid for
        cell_size_km: Size of each grid cell in kilometers
        target_crs: Coordinate reference system (e.g., 'EPSG:32628')
        min_area_percent: Minimum percentage of cell area that must overlap with country
        output_path: Path to save the grid shapefile (optional)

    Returns:
        GeoDataFrame with grid cells that meet the overlap threshold

    Raises:
        FileNotFoundError: If the country shapefile cannot be found
        ValueError: If the grid generation fails
    """
    try:
        # Find and load the country shapefile
        country_shapefile = find_shapefile(country_name)
        logger.info(f"Loading country shapefile: {country_shapefile}")

        country_gdf = gpd.read_file(country_shapefile)

        # Ensure consistent coordinate reference system
        if country_gdf.crs is None:
            logger.warning(
                f"Country shapefile has no CRS defined. Assuming {target_crs}"
            )
            country_gdf.set_crs(target_crs, inplace=True)
        elif country_gdf.crs != target_crs:
            logger.info(
                f"Reprojecting country shapefile from {country_gdf.crs} to {target_crs}"
            )
            country_gdf = country_gdf.to_crs(target_crs)

        # Get the bounds of the country
        bounds = country_gdf.total_bounds  # (minx, miny, maxx, maxy)

        # Convert cell size from kilometers to CRS units (typically meters)
        cell_size = cell_size_km * 1000  # Convert km to meters

        # Create grid cells
        x_min, y_min, x_max, y_max = bounds

        # Determine grid dimensions based on country bounds and cell size
        n_cells_x = int(np.ceil((x_max - x_min) / cell_size))
        n_cells_y = int(np.ceil((y_max - y_min) / cell_size))

        logger.info(
            f"Creating grid with {n_cells_x}x{n_cells_y} cells of size {cell_size_km}km"
        )

        # Create grid cells
        grid_cells = []
        cell_ids = []
        cell_id = 0

        for i in range(n_cells_x):
            for j in range(n_cells_y):
                # Calculate cell bounds
                cell_x_min = x_min + i * cell_size
                cell_y_min = y_min + j * cell_size
                cell_x_max = min(
                    cell_x_min + cell_size, x_max
                )  # Ensure we don't exceed bounds
                cell_y_max = min(cell_y_min + cell_size, y_max)

                # Create rectangular cell geometry
                cell = box(cell_x_min, cell_y_min, cell_x_max, cell_y_max)
                grid_cells.append(cell)
                cell_ids.append(cell_id)
                cell_id += 1

        # Create GeoDataFrame from the grid cells
        grid_gdf = gpd.GeoDataFrame(
            {"cell_id": cell_ids, "geometry": grid_cells}, crs=target_crs
        )

        # Filter cells based on minimum area percentage within the country
        logger.info(
            f"Filtering grid cells based on {min_area_percent}% minimum area overlap with {country_name}"
        )
        country_union = country_gdf.unary_union

        # Calculate how much of each cell overlaps with the country boundary
        grid_gdf["intersection"] = grid_gdf.geometry.apply(
            lambda cell: cell.intersection(country_union)
        )
        grid_gdf["intersection_area"] = grid_gdf["intersection"].area
        grid_gdf["cell_area"] = grid_gdf.geometry.area
        grid_gdf["overlap_percent"] = (
            grid_gdf["intersection_area"] / grid_gdf["cell_area"]
        ) * 100

        # Filter cells with sufficient overlap
        grid_gdf = grid_gdf[grid_gdf["overlap_percent"] >= min_area_percent].copy()

        # Clean up temporary columns
        grid_gdf.drop(
            columns=[
                "intersection",
                "intersection_area",
                "cell_area",
                "overlap_percent",
            ],
            inplace=True,
        )

        # Reset index and ensure cell_id is sequential
        grid_gdf.reset_index(drop=True, inplace=True)
        grid_gdf["cell_id"] = range(len(grid_gdf))

        logger.info(
            f"Created grid with {len(grid_gdf)} cells for {country_name} (min {min_area_percent}% area overlap)"
        )
        # Save grid if output path is provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            grid_gdf.to_file(output_path)
            logger.info(f"Saved grid to {output_path}")

        return grid_gdf

    except Exception as e:
        logger.error(f"Error creating grid for {country_name}: {str(e)}")
        raise ValueError(f"Failed to create grid for {country_name}: {str(e)}")


def get_or_create_grid(
    country_name: str,
    cell_size_km: float,
    target_crs: str,
    min_area_percent: float = 40.0,
    force_recreate: bool = False,
) -> gpd.GeoDataFrame:
    """
    Get an existing grid or create a new one if it doesn't exist.

    Looks for a previously created grid file and loads it if available,
    otherwise generates a new grid and saves it for future use.

    Args:
        country_name: Name of the country
        cell_size_km: Size of each cell in kilometers
        target_crs: Coordinate reference system to use
        min_area_percent: Minimum percentage of cell area that must overlap with country
        force_recreate: If True, recreate the grid even if it exists

    Returns:
        GeoDataFrame with grid cells that meet the overlap threshold
    """
    # Define the path where the grid would be saved
    grid_dir = get_results_dir() / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / f"{country_name}_grid_{cell_size_km}km.gpkg"

    # Check if grid exists and load it if it does
    if grid_path.exists() and not force_recreate:
        logger.info(f"Loading existing grid from {grid_path}")
        try:
            grid_gdf = gpd.read_file(grid_path)
            return grid_gdf
        except Exception as e:
            logger.warning(
                f"Failed to load existing grid: {str(e)}. Creating new grid."
            )

    # Create new grid
    logger.info(f"Creating new grid for {country_name} with cell size {cell_size_km}km")
    return create_grid(
        country_name, cell_size_km, target_crs, min_area_percent, output_path=grid_path
    )
