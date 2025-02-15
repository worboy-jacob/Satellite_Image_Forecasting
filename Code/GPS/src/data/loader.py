# src/data/loader.py
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
import pycountry
from typing import Tuple, List, Dict
from utils.constants import ISO_MAPPINGS, WEALTH_INDEX_COLS, ISO_MAPPINGS_REVERSE

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.wealth_data = None
        self.country_mappings = {}

    def load_wealth_index(self, file_path: Path) -> pd.DataFrame:
        """
        Load and cache wealth index data efficiently.
        Uses optimized reading based on file format.
        """
        logger.info(f"Loading wealth index from {file_path}")

        if self.wealth_data is not None:
            return self.wealth_data

        try:
            if file_path.suffix == ".parquet":
                self.wealth_data = pd.read_parquet(file_path)
            else:
                # Optimize CSV reading by specifying dtypes and only needed columns
                self.wealth_data = pd.read_csv(
                    file_path,
                    usecols=list(WEALTH_INDEX_COLS.values()),
                    dtype={
                        WEALTH_INDEX_COLS["CLUSTER_ID"]: "int32",
                        WEALTH_INDEX_COLS["YEAR"]: "int16",
                        WEALTH_INDEX_COLS["WEIGHT"]: "float32",
                        WEALTH_INDEX_COLS["WEALTH_INDEX"]: "float32",
                    },
                )
            unique_country_codes = (
                self.wealth_data[WEALTH_INDEX_COLS["COUNTRY_CODE"]].str[:2].unique()
            )
            self.country_mappings = {
                ISO_MAPPINGS[code]: pycountry.countries.get(alpha_2=code).name
                for code in unique_country_codes
                if code in ISO_MAPPINGS
            }
            logger.info("Wealth index data loaded successfully")
            return self.wealth_data

        except Exception as e:
            logger.error(f"Error loading wealth index data: {e}")
            raise

    def get_available_data_points(self) -> List[Tuple[str, int]]:
        """
        Extract available country-year combinations from wealth index data.
        Returns list of (country_code, year) tuples.
        """
        if self.wealth_data is None:
            raise ValueError("Wealth index data must be loaded first")

        # Efficient extraction of unique combinations
        combinations = (
            self.wealth_data[WEALTH_INDEX_COLS["COUNTRY_CODE"]].str[:2].unique()
        )

        data_points = []
        for code in combinations:
            if code in ISO_MAPPINGS:
                years = self.wealth_data[
                    self.wealth_data[WEALTH_INDEX_COLS["COUNTRY_CODE"]].str[:2] == code
                ][WEALTH_INDEX_COLS["YEAR"]].unique()
                data_points.extend([(ISO_MAPPINGS[code], int(year)) for year in years])

        return sorted(data_points)

    def load_gps_data(self, file_path: Path) -> gpd.GeoDataFrame:
        """
        Load GPS data efficiently from shapefile.
        """
        logger.info(f"Loading GPS data from {file_path}")
        try:
            return gpd.read_file(file_path)
        except Exception as e:
            logger.error(f"Error loading GPS data: {e}")
            raise

    def filter_wealth_data(self, country_code: str, year: int) -> pd.DataFrame:
        """
        Filter wealth index data for specific country and year.
        """
        if self.wealth_data is None:
            raise ValueError("Wealth index data must be loaded first")

        country_alpha2 = ISO_MAPPINGS_REVERSE[country_code]
        mask = (
            self.wealth_data[WEALTH_INDEX_COLS["COUNTRY_CODE"]].str[:2]
            == country_alpha2
        ) & (self.wealth_data[WEALTH_INDEX_COLS["YEAR"]] == year)

        filtered_data = self.wealth_data[mask].copy()

        if filtered_data.empty:
            raise ValueError(
                f"No data found for country {country_code} and year {year}"
            )

        return filtered_data
