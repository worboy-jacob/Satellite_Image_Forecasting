# src/utils/constants.py
from typing import Dict, Final
import pycountry
import geopandas as gpd
import logging
from pathlib import Path
import requests
import io

logger = logging.getLogger(__name__)


def get_natural_earth_data() -> gpd.GeoDataFrame:
    """
    Get Natural Earth data from local file or download if not present.
    Returns:
        GeoDataFrame containing country boundaries and metadata.
    """
    data_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "natural_earth"
        / "countries.geojson"
    )

    if data_path.exists():
        logger.info("Loading Natural Earth data from local file...")
        return gpd.read_file(data_path)

    logger.info("Downloading Natural Earth data...")
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        world = gpd.read_file(io.StringIO(response.text))
        if world.empty:
            raise ValueError("Downloaded GeoJSON contains no data")

        data_path.parent.mkdir(parents=True, exist_ok=True)
        world.to_file(data_path, driver="GeoJSON")

        return world

    except Exception as e:
        logger.error(f"Error downloading Natural Earth data: {e}")
        raise


def determine_utm_zone(country_data: gpd.GeoDataFrame) -> int:
    """
    Determine the appropriate UTM zone for a country using its centroid.
    Args:
        country_data: GeoDataFrame containing single country geometry
    Returns:
        EPSG code for appropriate UTM zone
    """
    centroid = country_data.geometry.iloc[0].centroid
    lon, lat = centroid.x, centroid.y

    zone_number = int((lon + 180) / 6) + 1
    base_epsg = 32600 if lat >= 0 else 32700

    return base_epsg + zone_number


def get_country_crs(country_code: str) -> int:
    """
    Get the appropriate CRS for a specific country.

    Args:
        country_code: Three-letter ISO country code
    Returns:
        EPSG code for appropriate UTM zone
    """
    world = get_natural_earth_data()
    country_data = world[world["ADM0_A3"] == country_code]

    if country_data.empty:
        raise ValueError(f"Country {country_code} not found in Natural Earth dataset")

    return determine_utm_zone(country_data)


class LazyCountryCRS:
    """Lazy loading dictionary-like object for country CRS values."""

    def __getitem__(self, country_code: str) -> int:
        """Get CRS for a country code."""
        return get_country_crs(country_code)

    def __contains__(self, country_code: str) -> bool:
        """Check if a country code is valid."""
        try:
            self[country_code]
            return True
        except ValueError:
            return False


def get_country_name(country_code: str) -> str:
    """
    Get full country name from ISO alpha-3 code.

    Args:
        country_code: Three-letter ISO country code
    Returns:
        Full country name
    Raises:
        ValueError: If country code is invalid
    """
    try:
        country = pycountry.countries.get(alpha_3=country_code)
        if not country:
            raise ValueError(f"Invalid country code: {country_code}")
        return country.name
    except Exception as e:
        logger.error(f"Error getting country name for {country_code}: {e}")
        raise


# ISO code mappings
ISO_MAPPINGS: Final[Dict[str, str]] = {
    country.alpha_2: country.alpha_3 for country in pycountry.countries
}

ISO_MAPPINGS_REVERSE: Final[Dict[str, str]] = {v: k for k, v in ISO_MAPPINGS.items()}

# File patterns
SHAPEFILE_PATTERN: Final[str] = "*.shp"
GPS_FILE_PATTERN: Final[str] = "*.shp"

# Column names for wealth index data
WEALTH_INDEX_COLS: Final[Dict[str, str]] = {
    "CLUSTER_ID": "hv001",
    "COUNTRY_CODE": "hv000",
    "YEAR": "hv007",
    "WEIGHT": "hv005",
    "WEALTH_INDEX": "wealth_index",
}

# Default CRS for initial data loading
DEFAULT_CRS: Final[int] = 4326  # WGS84

# Create lazy-loading COUNTRY_CRS object
COUNTRY_CRS = LazyCountryCRS()
