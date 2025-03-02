# wealth_mapping/visualization/plots.py
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


class WealthMapVisualizer:
    """Handles creation and saving of wealth distribution maps."""

    def __init__(
        self,
        config: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 15),
        dpi: int = 300,
    ):
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
        Create wealth distribution map.

        Args:
            wealth_grid: GeoDataFrame containing processed wealth data
            boundary: GeoDataFrame containing boundary geometry
            title: Optional title for the map
        """
        try:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

            # Plot boundary
            boundary.boundary.plot(ax=self.ax, color="black", linewidth=1)

            # Plot wealth distribution
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
        Save the created map to file.

        Args:
            output_path: Path where to save the map
        """
        if self.fig is None:
            raise ValueError("Map must be created before saving")

        try:
            self.fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(self.fig)
            logger.info(f"Map saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving map: {str(e)}")
            raise
