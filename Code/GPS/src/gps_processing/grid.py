# src/gps_processing/grid.py
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class GridProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cell_size = config["processing"]["cell_size"]

    def create_grid(
        self, shapefile: gpd.GeoDataFrame, wealth_data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Create and process grid for wealth mapping.
        """
        logger.info("Creating processing grid...")

        try:
            # Ensure same CRS
            shapefile = shapefile.to_crs(wealth_data.crs)
            # Create union of wealth data error bounds once
            wealth_union = gpd.GeoSeries(
                [unary_union(wealth_data.error_bounds)], crs=wealth_data.crs
            )
            # Get bounds and create grid cells
            bounds = shapefile.total_bounds
            rounded_bounds = self._round_bounds(bounds)

            logger.info("Generating grid cells...")
            grid_cells = self._generate_grid_cells(rounded_bounds)

            # Create GeoDataFrame from grid
            grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=shapefile.crs)

            # Get country boundary
            country_boundary = unary_union(shapefile.geometry)
            # Filter grid cells
            logger.info("Filtering grid cells...")
            # Combine boundaries into single geometry for one-time intersection check
            combined_boundary = country_boundary.intersection(wealth_union)
            # Use spatial index for efficient intersection check
            # Quick debug visualization
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot country boundary
            if isinstance(country_boundary, (gpd.GeoDataFrame, gpd.GeoSeries)):
                country_boundary.plot(ax=ax1, color="blue", alpha=0.5)
            else:
                gpd.GeoSeries([country_boundary]).plot(ax=ax1, color="blue", alpha=0.5)
            ax1.set_title("Country Boundary")

            # Plot wealth union
            gpd.GeoSeries([wealth_union]).plot(ax=ax2, color="red", alpha=0.5)
            ax2.set_title("Wealth Union")

            # Plot intersection
            combined_boundary = country_boundary.intersection(wealth_union)
            gpd.GeoSeries([combined_boundary]).plot(ax=ax3, color="green", alpha=0.5)
            ax3.set_title("Intersection")

            plt.tight_layout()
            plt.show()
            mask = grid_gdf.geometry.intersects(combined_boundary)
            grid_gdf = grid_gdf[mask]

            # Process wealth data for filtered cells
            grid_gdf["wealth_value"] = self._process_grid_cells(grid_gdf, wealth_data)

            # Final clip to wealth union
            logger.info("Clipping grid cells...")
            with tqdm(total=len(grid_gdf), desc="Clipping cells") as pbar:
                grid_gdf.geometry = [
                    geom.intersection(wealth_union) for geom in tqdm(grid_gdf.geometry)
                ]

            logger.info("Grid processing completed successfully")
            return grid_gdf

        except Exception as e:
            logger.error(f"Error in grid processing: {e}")
            raise

    def _round_bounds(self, bounds: tuple) -> tuple:
        """Round bounds to ensure they're divisible by cell size."""
        minx, miny, maxx, maxy = bounds
        return (
            np.floor(minx / self.cell_size) * self.cell_size,
            np.floor(miny / self.cell_size) * self.cell_size,
            np.ceil(maxx / self.cell_size) * self.cell_size,
            np.ceil(maxy / self.cell_size) * self.cell_size,
        )

    def _generate_grid_cells(self, bounds: tuple) -> List:
        """Generate grid cells based on bounds."""
        minx, miny, maxx, maxy = bounds
        x_coords = np.arange(minx, maxx, self.cell_size)
        y_coords = np.arange(miny, maxy, self.cell_size)

        total_cells = len(x_coords) * len(y_coords)
        grid_cells = []

        with tqdm(total=total_cells, desc="Creating grid") as pbar:
            for x in x_coords:
                for y in y_coords:
                    cell = box(x, y, x + self.cell_size, y + self.cell_size)
                    grid_cells.append(cell)
                    pbar.update(1)

        return grid_cells

    def _process_grid_cells(
        self, grid_gdf: gpd.GeoDataFrame, wealth_data: gpd.GeoDataFrame
    ) -> pd.Series:
        """Calculate wealth values for each grid cell using double weighting."""
        wealth_values = []

        for cell in tqdm(grid_gdf.geometry, desc="Processing cells"):
            # Find intersecting wealth points
            intersecting = wealth_data[wealth_data.error_bounds.intersects(cell)]

            if not intersecting.empty:
                # Calculate intersection areas
                areas = [
                    bound.intersection(cell).area for bound in intersecting.error_bounds
                ]

                # Calculate final weights combining area and wealth weights
                area_weights = np.array(areas) / np.sum(areas)
                wealth_weights = np.array(intersecting.weight)
                final_weights = (area_weights * wealth_weights) / np.sum(
                    area_weights * wealth_weights
                )

                # Calculate weighted average
                cell_value = np.average(
                    intersecting.wealth_index, weights=final_weights
                )
            else:
                cell_value = np.nan

            wealth_values.append(cell_value)

        return pd.Series(wealth_values)
