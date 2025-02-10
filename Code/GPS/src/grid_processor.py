# src/grid_processor.py
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import prepare, intersects
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class GridProcessor:
    def __init__(self, config):
        self.config = config
        self.cell_size = config["processing"]["cell_size"]

    def calculate_weighted_average(self, intersections):
        """Calculate double-weighted average using vectorized operations."""
        areas = intersections["intersection_area"].values
        hv005 = intersections["hv005"].values
        wealth_indices = intersections["wealth_index"].values

        area_weights = areas / np.sum(areas)
        hv005_weights = hv005 / np.sum(hv005)
        combined_weights = area_weights * hv005_weights
        combined_weights /= np.sum(combined_weights)

        return np.sum(wealth_indices * combined_weights)

    def create_grid(self, shapefile, wealth_df):
        """Create grid and process intersections."""
        logger.info("Creating processing grid...")

        # Implementation of create_grid method
        # (Previous implementation with added logging)
