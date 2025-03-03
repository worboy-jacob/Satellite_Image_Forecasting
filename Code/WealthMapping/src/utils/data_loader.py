# src/utils/data_loader.py
from pathlib import Path
import geopandas as gpd
import pandas as pd
from typing import Tuple, Dict, Any, List
import logging
import os
from .paths import get_gps_dir, get_data_dir, get_shapefiles_dir, get_wealthindex_dir

logger = logging.getLogger("wealth_mapping.loader")


def load_shapefile(country_name: str) -> gpd.GeoDataFrame:
    """
    Load a shapefile for a specific country.

    Args:
        country_name: Name of the country

    Returns:
        GeoDataFrame containing the shapefile data
    """
    try:
        shapefile_dir = get_shapefiles_dir() / country_name
        shapefile_paths = list(shapefile_dir.glob("*.shp"))

        if not shapefile_paths:
            raise FileNotFoundError(f"No shapefile found for {country_name}")

        # Take the first shapefile found
        shapefile_path = shapefile_paths[0]
        data = gpd.read_file(shapefile_path)
        logger.debug(f"Successfully loaded shapefile from {shapefile_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading shapefile for {country_name}: {str(e)}")
        raise


def load_gps_data(country_name: str, year: int) -> gpd.GeoDataFrame:
    """
    Load GPS data from shapefile for a specific country and year.

    Args:
        country_name: Name of the country
        year: Year of the data

    Returns:
        GeoDataFrame containing GPS data
    """
    try:
        gps_dir = get_gps_dir() / country_name / str(year)
        shapefile_paths = list(gps_dir.glob("*.shp"))

        if not shapefile_paths:
            raise FileNotFoundError(
                f"No GPS data found for {country_name} in year {year}"
            )

        # Take the first shapefile found
        shapefile_path = shapefile_paths[0]
        data = gpd.read_file(shapefile_path)
        logger.debug(f"Successfully loaded GPS data from {shapefile_path}")
        return data
    except Exception as e:
        logger.error(
            f"Error loading GPS data for {country_name} in year {year}: {str(e)}"
        )
        raise


def load_wealth_index(file_path: Path) -> pd.DataFrame:
    """
    Load wealth index data from parquet file.

    Args:
        file_path: Path to the wealth index parquet file

    Returns:
        DataFrame containing wealth index data
    """
    try:
        data = pd.read_parquet(file_path)
        logger.debug(f"Successfully loaded wealth index from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading wealth index {file_path}: {str(e)}")
        raise


def filter_wealth_data(wealth_data: pd.DataFrame, iso2: str, year: int) -> pd.DataFrame:
    """
    Filter wealth data for a specific country and year.

    Args:
        wealth_data: Full wealth index dataframe
        iso2: ISO2 country code
        year: Year to filter

    Returns:
        Filtered dataframe
    """
    # Extract country code from hv000 (first two characters)
    country_filter = wealth_data["hv000"].str[:2] == iso2
    # Filter by year
    year_filter = wealth_data["hv007"].astype(int) == year

    filtered_data = wealth_data[country_filter & year_filter]

    if filtered_data.empty:
        logger.warning(f"No wealth data found for country {iso2} in year {year}")

    return filtered_data


def load_all_data(config: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Load all required data files based on configuration.

    Args:
        config: Configuration dictionary containing file paths and country-year specifications

    Returns:
        Dictionary with structure: {country_iso2: {year: {"boundary": gdf, "gps": gdf, "wealth": df, "crs": str}}}
    """
    logger.info("Starting to load all data files")

    # Load wealth index data
    wealth_path = get_wealthindex_dir() / config.get("wealth_index")
    wealth_data = load_wealth_index(wealth_path)

    # Process data for each country and year specified in config
    result = {}

    for country_config in config.get("countries", []):
        country_name = country_config.get("name")
        country_iso2 = country_config.get("iso2")
        country_crs = country_config.get("crs")
        years = country_config.get("years", [])

        logger.info(
            f"Processing data for {country_name} ({country_iso2}) for years: {years}"
        )

        # Initialize country dict if not exists
        if country_iso2 not in result:
            result[country_iso2] = {}

        try:
            # Load country boundary once per country
            boundary_data = load_shapefile(country_name)
            boundary_data.to_crs(config.get("default_crs"), inplace=True)

            # Process each year for this country
            for year in years:
                try:
                    # Load GPS data for this country and year
                    gps_data = load_gps_data(country_name, year)

                    # Filter wealth data for this country and year
                    year_wealth = filter_wealth_data(wealth_data, country_iso2, year)

                    # Store data for this country-year pair
                    result[country_iso2][year] = {
                        "boundary": boundary_data,
                        "gps": gps_data,
                        "wealth": year_wealth,
                        "crs": country_crs,
                    }

                    logger.info(
                        f"Successfully loaded data for {country_name} ({country_iso2}) - {year}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing data for {country_name} - {year}: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error loading boundary data for {country_name}: {str(e)}")

    if not result or all(len(years) == 0 for years in result.values()):
        raise ValueError("No valid country-year data could be loaded")

    return result
