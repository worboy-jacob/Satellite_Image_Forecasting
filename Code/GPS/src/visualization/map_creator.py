# src/visualization/map_creator.py
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WealthMapVisualizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.viz_config = config["visualization"]
        self.figure_size = self.viz_config["figsize"]
        self.dpi = self.viz_config["dpi"]
        self.basemap = self.viz_config["basemap"]
        self.colormap = self.viz_config["colormap"]

    def create_wealth_map(
        self,
        grid_data: gpd.GeoDataFrame,
        boundary: gpd.GeoDataFrame,
        output_path: str,
        title: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Create wealth distribution map with consistent color scaling.
        """
        logger.info(f"Creating wealth map: {title}")

        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            # Set value limits if not provided
            if vmin is None:
                vmin = grid_data.wealth_value.min()
            if vmax is None:
                vmax = grid_data.wealth_value.max()

            # Plot wealth grid
            grid_data.plot(
                column="wealth_value",
                cmap=self.colormap["type"],
                alpha=0.7,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                legend=True,
                legend_kwds={"label": "Wealth Index", "orientation": "vertical"},
            )

            # Plot country boundary
            boundary.boundary.plot(ax=ax, color="black", linewidth=1, alpha=0.5)

            # Add basemap
            ctx.add_basemap(ax, source=self.basemap, alpha=0.5)

            # Customize plot
            ax.set_title(title, pad=20)
            ax.set_axis_off()

            # Save plot
            plt.savefig(output_path, bbox_inches="tight", dpi=self.dpi)
            plt.close()

            logger.info(f"Map saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Error creating wealth map: {e}")
            raise
