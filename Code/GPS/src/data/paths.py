# src/data/paths.py
from pathlib import Path
from typing import Dict, Any
import logging
from utils.constants import SHAPEFILE_PATTERN, GPS_FILE_PATTERN, get_country_name
import pycountry

logger = logging.getLogger(__name__)


class PathManager:

    def __init__(
        self,
        config: Dict[str, Any],
        project_root: Path,
        country_mappings: Dict[str, str] = None,
    ):
        self.config = config
        self.project_root = project_root
        self.country_mappings = country_mappings or {}
        self.data_dir = self.project_root / "data"

        # Specific directories
        self.gps_dir = self.data_dir / "GPS"
        self.shapefile_dir = self.data_dir / "ShapeFiles"
        self.results_dir = self.data_dir / "Results"
        self.wealth_index_dir = self.results_dir / "WealthIndex"

        # Get wealth index formats from config
        self.wealth_index_formats = self.config["paths"]["wealth_index_formats"]

        # Create directories if they don't exist
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.gps_dir,
            self.shapefile_dir,
            self.results_dir,
            self.wealth_index_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def find_wealth_index(self) -> Path:
        """
        Find the wealth index file in the WealthIndex directory using configured formats.

        Returns:
            Path to the wealth index file

        Raises:
            FileNotFoundError if no wealth index file is found in any configured format
        """
        logger.info(f"Searching for wealth index file in {self.wealth_index_dir}")

        # Check each configured format
        for format_pattern in self.wealth_index_formats:
            file_path = self.wealth_index_dir / format_pattern
            if file_path.exists():
                logger.info(f"Found wealth index file: {file_path}")
                return file_path

        # If no file found, raise error with detailed message
        raise FileNotFoundError(
            f"No wealth index file found in {self.wealth_index_dir}. "
            f"Expected formats: {', '.join(self.wealth_index_formats)}"
        )

    # In paths.py
    def find_gps_data(self, country_code: str, year: int) -> Path:
        """
        Find GPS data file for given country and year.
        Uses full country name for folder lookup.

        Args:
            country_code: Three-letter ISO country code (e.g., 'KEN')
            year: Year of data (e.g., 2014)

        Returns:
            Path to GPS data file

        Raises:
            FileNotFoundError: If GPS directory or files not found
            ValueError: If country code not valid
        """
        try:
            # Get full country name for folder lookup
            country_name = get_country_name(country_code)
            gps_dir = self.gps_dir / country_name / str(year)

            if not gps_dir.exists():
                raise FileNotFoundError(
                    f"GPS directory not found for {country_name} - {year}: {gps_dir}"
                )

            gps_files = list(gps_dir.glob(GPS_FILE_PATTERN))
            if not gps_files:
                raise FileNotFoundError(f"No GPS files found in {gps_dir}")

            return gps_files[0]

        except Exception as e:
            logger.error(f"Error finding GPS data for {country_code}: {str(e)}")
            raise

    def find_shapefile(self, country_code: str) -> Path:
        """
        Find shapefile for given country using full country name.

        Args:
            country_code: Three-letter ISO country code (e.g., 'KEN')

        Returns:
            Path to shapefile

        Raises:
            FileNotFoundError: If shapefile directory or files not found
            ValueError: If country code not valid
        """
        try:
            # Get full country name for folder lookup
            country_name = get_country_name(country_code)
            country_dir = self.shapefile_dir / country_name

            if not country_dir.exists():
                raise FileNotFoundError(
                    f"Shapefile directory not found for {country_name}: {country_dir}"
                )

            shapefiles = list(country_dir.glob(SHAPEFILE_PATTERN))
            if not shapefiles:
                raise FileNotFoundError(f"No shapefiles found in {country_dir}")

            return shapefiles[0]

        except Exception as e:
            logger.error(f"Error finding shapefile for {country_code}: {str(e)}")
            raise

    def create_output_filename(self, country_code: str, year: int) -> Path:
        """Create output filename for results."""
        output_dir = self.results_dir / "GPSResults"
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / f"wealth_map_{country_code}_{year}.png"
