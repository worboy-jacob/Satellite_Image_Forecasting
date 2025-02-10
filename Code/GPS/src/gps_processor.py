# src/gps_processor.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class GPSDataProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_gps_data(self, file_path):
        """Load GPS data from shapefile."""
        try:
            self.logger.info(f"Loading GPS data from {file_path}")
            return gpd.read_file(file_path)
        except Exception as e:
            self.logger.error(f"Error loading GPS data: {e}")
            raise

    def load_wealth_index(self, file_path):
        """Load wealth index data."""
        try:
            self.logger.info(f"Loading wealth index data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            self.logger.error(f"Error loading wealth index data: {e}")
            raise

    def calculate_cluster_wealth(self, wealth_df):
        """Calculate weighted average wealth index for each cluster."""
        return (
            wealth_df.groupby("hv001")
            .apply(
                lambda x: pd.Series(
                    {"wealth_index": np.average(x["wealth_index"], weights=x["hv005"])}
                )
            )
            .reset_index()
        )

    def create_gps_points(self, df):
        """Create geometry points from GPS coordinates."""
        return gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["LONGNUM"], df["LATNUM"])],
            crs=self.config["processing"]["crs"]["input"],
        )

    def create_error_bounds(self, row):
        """Create buffer based on urban/rural classification."""
        if row["URBAN_RURA"] == "U":
            return row["geometry"].buffer(self.config["processing"]["urban_buffer"])
        elif row["URBAN_RURA"] == "R":
            return row["geometry"].buffer(self.config["processing"]["rural_buffer"])
        else:
            self.logger.error(f"Invalid urban/rural classification for row {row}")
            raise ValueError("Invalid urban/rural classification")
