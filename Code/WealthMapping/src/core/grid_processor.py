# src/core/grid_processor.py
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import prepare, intersects
from typing import Dict, Any, List, Tuple
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import multiprocessing
from time import time
import gc  # For garbage collection

###TODO: Optimize parallel
logger = logging.getLogger("wealth_mapping.grid")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


def setup_logging(level_str: str) -> int:
    """Setup logging configuration."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(level_str.upper(), logging.INFO)
    multiprocessing.log_to_stderr(level=level)
    return level


# src/core/grid_processor.py
class GridProcessor:
    """Handles creation and processing of spatial grids."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cell_size = config.get("cell_size")
        logger.debug(f"GridProcessor initialized with cell size: {self.cell_size}")
        self.verbose = config.get("verbose", 0)
        self.n_jobs = config.get("n_jobs", 0)
        setup_logging(config.get("log_level", "INFO"))

    def _calculate_weighted_average(self, intersections_df: pd.DataFrame) -> float:
        """
        Calculate double-weighted average using area and survey weights.

        Args:
            intersections_df: DataFrame with intersection_area, hv005, and wealth_index

        Returns:
            Weighted average of wealth indices
        """
        areas = intersections_df["intersection_area"].values
        area_weights = areas / np.sum(areas)

        hv005 = intersections_df["hv005"].values
        survey_weights = hv005 / np.sum(hv005)

        combined_weights = area_weights * survey_weights
        combined_weights /= np.sum(combined_weights)

        return np.sum(intersections_df["wealth_index"].values * combined_weights)

    def _create_grid_cell(
        self, x: float, y: float, cell_size: float, unified_shape, pbar: tqdm
    ) -> Dict[str, Any]:
        """Create a single grid cell if it intersects with the unified shape."""
        cell = box(x, y, x + cell_size, y + cell_size)
        pbar.update(1)
        if intersects(cell, unified_shape):
            return {"geometry": cell}
        return None

    def _process_cell(self, cell_geom, wealth_data, wealth_union, wealth_sindex, pbar):
        """Process a single grid cell for wealth calculation."""
        if not intersects(cell_geom, wealth_union):
            pbar.update(1)
            return None

        possible_matches_idx = list(wealth_sindex.intersection(cell_geom.bounds))
        if not possible_matches_idx:
            pbar.update(1)
            return None

        matches = wealth_data.iloc[possible_matches_idx]
        intersecting = matches[matches.geometry.intersects(cell_geom)]

        if len(intersecting) == 0:
            pbar.update(1)
            return None

        # Calculate intersection areas
        intersection_areas = intersecting.geometry.apply(
            lambda x: x.intersection(cell_geom).area
        )

        intersections_df = pd.DataFrame(
            {
                "hv005": intersecting["hv005"].values,
                "wealth_index": intersecting["wealth_index"].values,
                "intersection_area": intersection_areas.values,
            }
        )

        value = (
            intersections_df["wealth_index"].iloc[0]
            if len(intersections_df) == 1
            else self._calculate_weighted_average(intersections_df)
        )

        # Return the full cell geometry instead of clipping it
        return {
            "geometry": cell_geom,
            "wealth_index": value,
            "local_wealth_union": unary_union(intersecting.geometry),
        }

    ###TODO: make chunks for better parallel processing
    def process(
        self,
        boundary: gpd.GeoDataFrame,
        wealth_data: gpd.GeoDataFrame,
        country_crs: str,
        default_crs: str,
    ) -> gpd.GeoDataFrame:
        """
        Deal with overlapping polygons

        This method creates a grid of cells and processes them against wealth data polygons.
        When cells are clipped to wealth data boundaries, the resulting geometries will naturally
        share boundaries with adjacent cells. These shared boundaries appear as:
        - LineStrings: Where two adjacent cells share an edge
        - Points: Where corners of cells meet
        - MultiLineStrings: For complex shared boundaries

        These shared boundaries are not overlaps - they're the expected result of adjacent polygons
        having common edges. The overlay operation reports these as separate geometries, but they
        do not represent duplicate or overlapping areas.

        Args:
            boundary: GeoDataFrame containing the study area boundary
            wealth_data: Processed wealth data with geometries
            country_crs: Coordinate reference system for the country
            default_crs: Default coordinate reference system to return results in

        Returns:
            GeoDataFrame containing processed grid cells with wealth indices, where each cell
            is clipped to the extent of its intersection with wealth data. Adjacent cells may
            share boundaries but do not overlap in area.
        """

        try:
            gc.collect()
            invalid_geoms = wealth_data[~wealth_data.geometry.is_valid]
            logger.info(f"Invalid wealth geometries: {len(invalid_geoms)}")
            logger.info("Starting grid processing pipeline")
            cell_size = self.cell_size
            # Create base grid
            logger.info("Creating base grid")
            boundary.to_crs(country_crs, inplace=True)
            wealth_data.to_crs(country_crs, inplace=True)
            bounds = boundary.total_bounds
            unified_shape = unary_union(boundary.geometry)
            prepare(unified_shape)
            # Calculate grid dimensions
            xmin, ymin, xmax, ymax = [
                np.floor(bounds[0] / cell_size) * cell_size,
                np.floor(bounds[1] / cell_size) * cell_size,
                np.ceil(bounds[2] / cell_size) * cell_size,
                np.ceil(bounds[3] / cell_size) * cell_size,
            ]

            # Create grid cells
            x_coords = np.arange(xmin, xmax + cell_size, cell_size)
            y_coords = np.arange(ymin, ymax + cell_size, cell_size)
            xx, yy = np.meshgrid(x_coords, y_coords)

            grid_cells = []
            start_time = time()
            with tqdm(total=len(xx.flat), desc="Creating grid cells") as pbar:
                results = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads"
                )(
                    delayed(self._create_grid_cell)(
                        x, y, cell_size, unified_shape, pbar
                    )
                    for x, y in zip(xx.flat, yy.flat)
                )
            gc.collect()
            logger.info(f"Created {len(results)} cells in {time()-start_time}s")

            # Filter out None results and create grid cells list
            grid_cells = [result for result in results if result is not None]

            grid_initial = gpd.GeoDataFrame(grid_cells, crs=boundary.crs)

            # Process wealth data
            logger.info("Processing wealth data")
            gc.collect()

            wealth_sindex = wealth_data.sindex
            wealth_union = unary_union(wealth_data.geometry)
            prepare(wealth_union)
            ###TODO: keep the other ID columns if need be
            # Process intersections and calculate wealth values
            logger.info("Removing unneeded grid cells and calculating wealth index.")
            start_time = time()
            results = []
            with tqdm(
                total=len(grid_initial), desc="Calculating wealth indices"
            ) as pbar:
                parallel_results = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads"
                )(
                    delayed(self._process_cell)(
                        cell_geom, wealth_data, wealth_union, wealth_sindex, pbar
                    )
                    for _, cell_geom in grid_initial.geometry.items()
                )
                # Filter out None results
                results = [result for result in parallel_results if result is not None]
            gc.collect()
            logger.info(
                f"Changed number of cells from {len(grid_initial)} to {len(results)} in {time()-start_time}s"
            )
            grid = gpd.GeoDataFrame(
                data={
                    "wealth_index": [r["wealth_index"] for r in results],
                    "local_wealth_union": [r["local_wealth_union"] for r in results],
                },
                geometry=[r["geometry"] for r in results],
                crs=grid_initial.crs,
            )
            overlaps = grid.overlay(grid, how="intersection")
            logger.info(
                f"Number of overlapping areas before clipping: {len(overlaps) - len(grid)}"
            )
            gc.collect()
            grid["geometry"] = grid.apply(
                lambda row: row.geometry.intersection(row.local_wealth_union), axis=1
            )
            grid = grid.drop(columns=["local_wealth_union"])
            # The following overlap checks analyze all intersection geometries:
            # - LineStrings represent shared edges between adjacent cells
            # - Points represent shared corners
            # - MultiLineStrings represent complex shared boundaries
            # - Polygons/MultiPolygons include self-intersections (each cell with itself)
            # True overlapping areas between different cells would appear as additional
            # Polygons/MultiPolygons beyond the self-intersections, resulting in
            # len(polygon_overlaps) - len(grid) > 0
            if self.config.get("log_level", "INFO") == "DEBUG":
                logger.debug(f"Number of cells before overlap check: {len(grid)}")
                logger.debug(
                    f"Geometry types before overlap check: {grid.geometry.geom_type.value_counts()}"
                )
                overlaps = grid.overlay(grid, how="intersection", keep_geom_type=False)
                logger.debug(
                    f"Number of overlapping areas (all geometry types): {len(overlaps) - len(grid)}"
                )
                logger.debug(
                    f"Overlap geometry types: {overlaps.geometry.geom_type.value_counts()}"
                )
                polygon_overlaps = grid.overlay(
                    grid, how="intersection", keep_geom_type=True
                )
                logger.debug(
                    f"Number of overlapping polygons: {len(polygon_overlaps) - len(grid)}"
                )
            grid.to_crs(default_crs, inplace=True)
            logger.info("Grid processing completed successfully")
            gc.collect()
            return grid

        except Exception as e:
            logger.error(f"Error in grid processing: {str(e)}")
            raise
