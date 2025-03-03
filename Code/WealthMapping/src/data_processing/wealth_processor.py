# src/data_processing/wealth_processor.py
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from shapely.geometry import Point, Polygon, MultiPolygon
import logging
from shapely.geometry import Point

logger = logging.getLogger("wealth_mapping.wealth")


# src/data_processing/wealth_processor.py
class WealthProcessor:
    """Handles processing of wealth index data and GPS coordinates."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffer_sizes = {
            "U": self.config["buffer"]["urban"],
            "R": self.config["buffer"]["rural"],
        }
        logger.debug("WealthProcessor initialized")

    def _calculate_cluster_wealth(self, wealth_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted average wealth index for each cluster."""
        return (
            wealth_data.groupby("hv001")
            .agg(
                {
                    "wealth_index": lambda x: np.average(
                        x, weights=wealth_data.loc[x.index, "hv005"]
                    ),
                    "hv000": "first",  # Take first value as it's constant within cluster
                    "hv007": "first",  # Take first value as it's constant within cluster
                    "hv005": "sum",  # Sum the weights for each cluster
                }
            )
            .reset_index()
        )

    def _create_buffer(self, row: pd.Series) -> Union[Polygon, MultiPolygon]:
        """
        Create buffer around point based on urban/rural classification.

        Args:
            row: DataFrame row containing URBAN_RURA field and geometry

        Returns:
            Buffered geometry as Polygon or MultiPolygon
        """
        try:
            buffer_size = self.buffer_sizes.get(row["URBAN_RURA"])
            if buffer_size is None:
                logger.error(f"Invalid urban/rural classification: {row['URBAN_RURA']}")
                raise ValueError(
                    f"Invalid urban/rural classification: {row['URBAN_RURA']}"
                )
            return row.geometry.buffer(buffer_size)
        except Exception as e:
            logger.error(f"Error creating buffer: {str(e)}")
            raise

    def process(
        self,
        gps_data: gpd.GeoDataFrame,
        wealth_data: pd.DataFrame,
        default_crs: str = None,
        country_crs: str = None,
    ) -> gpd.GeoDataFrame:
        """
        Process wealth and GPS data in one go, including CRS conversion.

        Args:
            gps_data: GPS coordinates data
            wealth_data: Household wealth index data
            target_crs: Target coordinate reference system

        Returns:
            Processed GeoDataFrame with wealth indices and geometries
        """
        try:
            # Calculate cluster-level wealth indices
            cluster_wealth = self._calculate_cluster_wealth(wealth_data)
            # Merge GPS and wealth data
            merged_data = pd.merge(
                gps_data,
                cluster_wealth,
                left_on="DHSCLUST",
                right_on="hv001",
                how="right",
            )

            # Create point geometries
            geometry = [
                Point(xy) for xy in zip(merged_data["LONGNUM"], merged_data["LATNUM"])
            ]
            gdf = gpd.GeoDataFrame(merged_data, geometry=geometry, crs=default_crs)
            gdf.to_crs(country_crs, inplace=True)
            # Create buffers
            gdf["buffer"] = gdf.apply(self._create_buffer, axis=1)
            # Prepare final dataset
            processed_data = gdf[
                ["wealth_index", "hv005", "buffer", "hv000", "hv007"]
            ].copy()
            processed_data.set_geometry("buffer", inplace=True)
            processed_data.set_crs(gdf.crs, inplace=True)
            # Convert CRS if specified
            if default_crs is not None:
                processed_data = processed_data.to_crs(default_crs)
                logger.info(f"Converted data to CRS: {default_crs}")

            logger.info(f"Successfully processed {len(processed_data)} clusters")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing wealth data: {str(e)}")
            raise
