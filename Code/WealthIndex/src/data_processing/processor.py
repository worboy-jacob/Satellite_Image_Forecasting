from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger("wealth_index.processor")


class DataProcessor:
    """Handles data loading and preprocessing."""

    def __init__(self, config: dict):
        self.config = config

    def load_all_data(self) -> dict:
        """Load all data files based on configuration."""
        dfs = {}
        project_root = Path(__file__).parent.parent.parent

        for country, years in self.config["country_year"].items():
            country = f"{country}_Data"
            if isinstance(years, str):
                years = [years]
            for year in years:
                year_str = str(year)
                key = f"{country}_{year_str}"
                data_path = project_root / "data" / "DHS" / country / year_str

                if not data_path.exists():
                    logger.warning(f"Directory not found: {data_path}")
                    continue

                # Find first .DTA file
                data_file = next(data_path.glob("*.DTA"), None)
                if data_file:
                    dfs[key] = self.load_data(data_file)

        return dfs

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess single data file."""
        df = pd.read_stata(file_path)
        return df[self.config["columns_to_include"]]
