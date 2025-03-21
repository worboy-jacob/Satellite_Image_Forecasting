"""
Wealth processing module for DHS survey data.

Processes household wealth indices and GPS cluster data to create
spatial representations of wealth distribution. Handles the creation
of appropriate buffers around survey cluster points based on urban/rural
classification.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from shapely.geometry import Point, Polygon, MultiPolygon
import logging

logger = logging.getLogger("wealth_mapping.wealth")


# src/data_processing/wealth_processor.py
class WealthProcessor:
    """
    Processes DHS wealth index data with GPS coordinates.

    Combines household wealth indices with GPS cluster locations,
    calculates weighted averages by cluster, and creates appropriate
    spatial buffers based on urban/rural classification.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the wealth processor with configuration.

        Args:
            config: Configuration dictionary containing buffer sizes for
                urban and rural areas under the 'buffer' key
        """
        self.config = config
        self.buffer_sizes = {
            "U": self.config["buffer"]["urban"],
            "R": self.config["buffer"]["rural"],
        }
        logger.debug("WealthProcessor initialized")

    def _calculate_cluster_wealth(self, wealth_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted average wealth index for each DHS cluster.

        Aggregates household-level wealth indices to cluster level using
        the survey sampling weights (hv005) to compute properly weighted averages.

        Args:
            wealth_data: DataFrame containing household wealth indices and weights

        Returns:
            DataFrame with cluster-level weighted wealth indices
        """
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
        Create appropriate buffer around cluster point.

        Creates a circular buffer with radius based on the urban/rural
        classification of the cluster, using larger buffers for rural areas
        and smaller ones for urban areas.

        Args:
            row: DataFrame row containing URBAN_RURA field and geometry

        Returns:
            Buffered geometry as Polygon or MultiPolygon

        Raises:
            ValueError: If urban/rural classification is invalid
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
        Process wealth and GPS data to create spatial wealth representation.

        Combines household wealth data with GPS cluster locations, creates
        appropriate buffers, and handles coordinate reference system transformations.

        Args:
            gps_data: GPS coordinates data for DHS clusters
            wealth_data: Household wealth index data
            default_crs: Default coordinate reference system (typically WGS84)
            country_crs: Country-specific coordinate reference system

        Returns:
            GeoDataFrame with cluster-level wealth indices and buffer geometries

        Raises:
            Exception: If processing fails for any reason
        """
        try:
            # Calculate cluster-level wealth indices with survey weights
            cluster_wealth = self._calculate_cluster_wealth(wealth_data)

            # Merge GPS and wealth data on cluster ID
            merged_data = pd.merge(
                gps_data,
                cluster_wealth,
                left_on="DHSCLUST",
                right_on="hv001",
                how="right",
            )

            # Create point geometries from latitude/longitude
            geometry = [
                Point(xy) for xy in zip(merged_data["LONGNUM"], merged_data["LATNUM"])
            ]
            gdf = gpd.GeoDataFrame(merged_data, geometry=geometry, crs=default_crs)
            gdf.to_crs(country_crs, inplace=True)

            # Create buffers based on urban/rural classification
            gdf["buffer"] = gdf.apply(self._create_buffer, axis=1)

            # Extract only the columns we need for the final dataset
            processed_data = gdf[
                ["wealth_index", "hv005", "buffer", "hv000", "hv007"]
            ].copy()

            # Use buffer geometries as the primary geometry column
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
