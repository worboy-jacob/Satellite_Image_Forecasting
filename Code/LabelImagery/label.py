import os
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import matplotlib.pyplot as plt
from esda.moran import Moran
from libpysal.weights import Queen
import scipy.stats as stats
from tqdm import tqdm
from logging_config import setup_logging
import sys


logger = setup_logging("INFO")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


def load_grid_cells(grid_gpkg_path):
    """Load grid cells from existing geopackage."""
    try:
        grid_gdf = gpd.read_file(grid_gpkg_path)
        logger.info(f"Loaded {len(grid_gdf)} grid cells with CRS: {grid_gdf.crs}")
        return grid_gdf
    except Exception as e:
        logger.error(f"Error loading grid cells: {e}")
        raise


def load_wealth_data(wealth_gpkg_path):
    """Load wealth index data from GPKG file."""
    try:
        wealth_gdf = gpd.read_file(wealth_gpkg_path)
        logger.info(
            f"Loaded {len(wealth_gdf)} wealth data points with CRS: {wealth_gdf.crs}"
        )
        return wealth_gdf
    except Exception as e:
        logger.error(f"Error loading wealth data: {e}")
        raise


def calculate_coverage(grid_gdf, wealth_gdf):
    """Calculate the coverage percentage for each grid cell."""
    # Ensure both GDFs have the same CRS
    if grid_gdf.crs != wealth_gdf.crs:
        logger.info(f"Converting CRS: {wealth_gdf.crs} to {grid_gdf.crs}")
        wealth_gdf = wealth_gdf.to_crs(grid_gdf.crs)

    # Calculate area of each grid cell
    grid_gdf["total_area"] = grid_gdf.geometry.area

    # Initialize coverage columns
    grid_gdf["covered_area"] = 0.0
    grid_gdf["coverage_percent"] = 0.0
    grid_gdf["weighted_wealth"] = np.nan

    # For each grid cell, calculate intersection with wealth data
    for idx, grid_cell in tqdm(
        grid_gdf.iterrows(), total=len(grid_gdf), desc="Calculating coverage"
    ):
        # Find intersecting wealth cells
        try:
            intersections = wealth_gdf[wealth_gdf.intersects(grid_cell.geometry)]

            if len(intersections) == 0:
                continue

            # Calculate intersection areas and weights
            intersection_areas = []
            weights = []
            wealth_values = []

            for _, wealth_cell in intersections.iterrows():
                intersection = grid_cell.geometry.intersection(wealth_cell.geometry)
                area = intersection.area
                intersection_areas.append(area)
                weights.append(area)
                wealth_values.append(wealth_cell["wealth_index"])

            # Store results
            grid_gdf.at[idx, "covered_area"] = sum(intersection_areas)
            grid_gdf.at[idx, "coverage_percent"] = (
                sum(intersection_areas) / grid_cell.total_area
            ) * 100

            # Calculate weighted wealth index
            if sum(weights) > 0:
                weighted_wealth = sum(
                    weights[i] * wealth_values[i] for i in range(len(weights))
                ) / sum(weights)
                grid_gdf.at[idx, "weighted_wealth"] = weighted_wealth

        except Exception as e:
            logger.error(f"Error processing grid cell {idx}: {e}")
            continue

    return grid_gdf


def apply_threshold(grid_gdf, threshold_percent):
    """Apply coverage threshold to grid cells."""
    logger.info(f"Applying coverage threshold of {threshold_percent}%")
    # Create a copy with only cells that meet the threshold
    thresholded_gdf = grid_gdf.copy()
    # Set weighted_wealth to NaN for cells below threshold
    mask = thresholded_gdf["coverage_percent"] < threshold_percent
    thresholded_gdf.loc[mask, "weighted_wealth"] = np.nan

    # Count how many cells meet the threshold
    cells_above_threshold = (~mask & ~thresholded_gdf["weighted_wealth"].isna()).sum()
    total_cells = len(grid_gdf)
    logger.info(
        f"{cells_above_threshold} of {total_cells} cells ({cells_above_threshold/total_cells*100:.2f}%) meet the {threshold_percent}% coverage threshold"
    )

    return thresholded_gdf
