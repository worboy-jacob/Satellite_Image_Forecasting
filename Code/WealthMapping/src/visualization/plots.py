"""
Visualization module for wealth distribution maps.

Provides functionality for creating and saving geospatial visualizations
of wealth distribution data using matplotlib and geopandas.
"""

import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Tuple, Optional, Dict, Any
import logging
import sys
import multiprocessing

logger = logging.getLogger("wealth_mapping.plot")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


def setup_logging(level_str: str) -> int:
    """
    Configure logging with the specified level.

    Args:
        level_str: String representation of logging level (DEBUG, INFO, etc.)

    Returns:
        Numeric logging level constant from logging module
    """
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


class WealthMapVisualizer:
    """
    Creates and saves geospatial visualizations of wealth distribution.

    Handles the creation of maps showing wealth distribution across geographic
    areas, with customizable figure size, resolution, and styling.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 15),
        dpi: int = 300,
    ):
        """
        Initialize the visualizer with configuration settings.

        Args:
            config: Configuration dictionary containing visualization parameters
            figsize: Size of the figure in inches as (width, height)
            dpi: Resolution of the output figure in dots per inch
        """
        self.config = config
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        setup_logging(config.get("log_level", "INFO"))

    def create_map(
        self,
        wealth_grid: gpd.GeoDataFrame,
        boundary: gpd.GeoDataFrame,
        title: Optional[str] = "Wealth Distribution",
    ) -> None:
        """
        Create a wealth distribution map visualization.

        Generates a map showing the spatial distribution of wealth indices,
        with administrative boundaries overlaid for geographic context.

        Args:
            wealth_grid: GeoDataFrame containing processed wealth data with geometry
            boundary: GeoDataFrame containing administrative boundary geometry
            title: Title for the map

        Raises:
            Exception: If map creation fails
        """
        try:
            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

            # Plot administrative boundaries in black
            boundary.boundary.plot(ax=self.ax, color="black", linewidth=1)

            # Plot wealth grid with colormap and legend
            wealth_grid.plot(
                column="wealth_index",
                ax=self.ax,
                legend=True,
                legend_kwds={"label": "Wealth Index"},
                cmap="RdYlBu_r",
            )

            self.ax.set_title(title)
            self.ax.axis("off")

            logger.info("Successfully created wealth distribution map")

        except Exception as e:
            logger.error(f"Error creating map: {str(e)}")
            raise

    def save_map(self, output_path: str) -> None:
        """
        Save the created map to a file.

        Args:
            output_path: File path where the map should be saved

        Raises:
            ValueError: If save is attempted before creating a map
            Exception: If saving fails
        """

        # Ensure the map exists before trying to save it
        if self.fig is None:
            raise ValueError("Map must be created before saving")

        try:
            # Save with high quality and proper cropping
            self.fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(self.fig)
            logger.info(f"Map saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving map: {str(e)}")
            raise
