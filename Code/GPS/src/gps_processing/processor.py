# src/gps_processing/processor.py
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import logging
from typing import Dict, Any
from utils.constants import WEALTH_INDEX_COLS

logger = logging.getLogger(__name__)


class GPSDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.urban_buffer = config["processing"]["urban_buffer"]
        self.rural_buffer = config["processing"]["rural_buffer"]
        self.input_crs = config["processing"]["crs"]["input"]

    def process_gps_data(
        self, gps_data: gpd.GeoDataFrame, wealth_data: pd.DataFrame, target_crs: int
    ) -> gpd.GeoDataFrame:
        """
        Process GPS data with wealth information.
        """
        logger.info("Processing GPS data")

        try:
            # Ensure correct CRS
            gps_data = gps_data.to_crs(self.input_crs)

            # Create points and set CRS
            gps_points = self.create_gps_points(gps_data)

            # Calculate cluster wealth
            cluster_wealth = self.calculate_cluster_wealth(wealth_data)

            # Merge wealth data with GPS points
            merged_data = gps_points.merge(
                cluster_wealth,
                left_on="DHSCLUST",
                right_on=WEALTH_INDEX_COLS["CLUSTER_ID"],
            )

            # Create error bounds
            merged_data["error_bounds"] = merged_data.apply(
                self.create_error_bounds, axis=1
            )

            # Convert to target CRS
            processed_data = merged_data.to_crs(target_crs)

            logger.info("GPS data processing completed successfully")
            return processed_data

        except Exception as e:
            logger.error(f"Error in GPS data processing: {e}")
            raise

    def create_gps_points(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create geometry points from GPS coordinates.
        """
        return gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["LONGNUM"], df["LATNUM"])],
            crs=self.input_crs,
        )

    def calculate_cluster_wealth(self, wealth_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted average wealth index for each cluster.
        Uses vectorized operations for efficiency.
        """
        return (
            wealth_df.groupby(WEALTH_INDEX_COLS["CLUSTER_ID"])
            .apply(
                lambda x: pd.Series(
                    {
                        "wealth_index": np.average(
                            x[WEALTH_INDEX_COLS["WEALTH_INDEX"]],
                            weights=x[WEALTH_INDEX_COLS["WEIGHT"]],
                        ),
                        "weight": x[WEALTH_INDEX_COLS["WEIGHT"]].sum(),
                    }
                )
            )
            .reset_index()
        )

    def create_error_bounds(self, row: pd.Series) -> Any:
        """
        Create buffer based on urban/rural classification.
        """
        buffer_size = (
            self.urban_buffer
            if row["URBAN_RURA"] == "U"
            else self.rural_buffer if row["URBAN_RURA"] == "R" else None
        )

        if buffer_size is None:
            logger.error(f"Invalid urban/rural classification for row {row.name}")
            raise ValueError("Invalid urban/rural classification")

        return row["geometry"].buffer(buffer_size)
