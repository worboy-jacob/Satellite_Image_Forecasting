"""
Data preprocessing module for DHS survey data.

Provides functionality for loading, cleaning, and standardizing household survey
data from Demographic and Health Surveys (DHS) across multiple countries and years.
Handles data type determination, missing value replacement, and special value mappings
to ensure consistency across datasets.
"""

from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Set
import numpy as np

logger = logging.getLogger("wealth_index.processor")


class DataProcessor:
    """
    Handles data loading and preprocessing for household survey data.

    Provides methods for loading DHS survey data files, applying consistent
    data transformations, handling missing values, and ensuring comparability
    across different countries and survey years.
    """

    def __init__(self, config: dict):
        """
        Initialize DataProcessor with configuration settings.

        Args:
            config: Configuration dictionary containing:
                - country_year: Dict mapping countries to their survey years
                - columns_to_include: List of column names to retain
                - missing_threshold: Maximum allowable proportion of missing values (0-1)
                - replace_val: Dict mapping values to be replaced
        """
        self.config = config
        self.missing_threshold = config.get("missing_threshold", 0.5)
        self._init_replacement_dict()
        self._init_special_mappings()

    def _init_special_mappings(self) -> None:
        """
        Initialize special mappings for specific categorical columns.

        Creates standardized value mappings for columns with coded values
        that need to be translated to meaningful categories (e.g., roof and
        wall materials).
        """
        self.special_mappings = {
            "hv215": {
                "10": "natural",
                "11": "no roof",
                "12": "thatch/palm leaf",
                "13": "clods of earth",
                "20": "rudimentary",
                "21": "rustic mat",
                "22": "palm/bamboo",
                "23": "wood planks",
                "24": "cardboard",
                "25": "plastic",
                "30": "finished",
                "31": "metal",
                "32": "wood",
                "33": "zinc/cement fiber",
                "34": "ceramic tiles",
                "35": "cement",
                "36": "roofing shingles",
                "96": "other",
            },
            "hv214": {
                "10": "natural",
                "11": "no walls",
                "12": "bamboo/cane/palm/trunks",
                "13": "dirt",
                "20": "rudimentary",
                "21": "bamboo with mud",
                "22": "stone with mud",
                "23": "uncovered adobe",
                "24": "plywood",
                "25": "cardboard",
                "26": "reused wood",
                "30": "finished",
                "31": "cement",
                "32": "stone with lime/cement",
                "33": "bricks",
                "34": "cement blocks",
                "35": "covered adobe",
                "36": "wood planks/shingles",
                "37": "tile",
                "96": "other",
            },
        }

    def _apply_special_mapping(self, series: pd.Series, column_name: str) -> pd.Series:
        """
        Apply column-specific value mapping to standardize categorical values.

        Transforms coded values into standardized category names based on predefined
        mappings while preserving values not found in the mapping dictionary.

        Args:
            series: Input series to transform
            column_name: Name of column being processed

        Returns:
            Series with mapped values and string dtype
        """
        if column_name not in self.special_mappings:
            return series

        mapping = self.special_mappings[column_name]
        # Convert to string for consistent comparison
        temp_series = series.astype(str)
        result = pd.Series(index=series.index, dtype="string")

        # Create mask for values that exist in the mapping dictionary
        mappable_mask = temp_series.isin(mapping.keys())

        # Apply mapping where possible
        result[mappable_mask] = temp_series[mappable_mask].map(mapping)

        # Keep original values where no mapping exists
        result[~mappable_mask] = temp_series[~mappable_mask]

        # Preserve NaN values
        result[series.isna()] = np.nan

        return result

    def _init_replacement_dict(self) -> None:
        """
        Initialize the value replacement dictionary with case variations.

        Creates a comprehensive replacement dictionary that includes multiple
        case variations (upper, lower, original) for each replacement value
        to ensure consistent handling regardless of input format.
        """
        base_replacements = self.config.get(
            "replace_val",
            {
                "unknown": np.nan,
                "none": 0,
                "95 or more": 95,
                "don't know": np.nan,
                "nan": np.nan,
            },
        )

        # Create comprehensive replacement dictionary with case variations
        self.replace_values = {}
        for k, v in base_replacements.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = np.nan

            self.replace_values[k] = v
            self.replace_values[k.upper()] = v
            self.replace_values[k.lower()] = v

    def determine_column_type(self, series: pd.Series) -> str:
        """
        Determine whether a column should be treated as numeric or categorical.

        Makes a definitive determination based on content analysis, checking
        if values can be converted to numbers without data loss and whether
        the numeric representation adds information value.

        Args:
            series: Column data to analyze

        Returns:
            'numeric' or 'categorical' based on content analysis
        """
        # Check if already numeric - simplest case
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # For non-numeric types, try conversion and analyze results
        non_null_values = series.dropna()

        try:
            numeric_converted = pd.to_numeric(non_null_values, errors="coerce")
            # Check if all values convert successfully to numbers
            if numeric_converted.notna().sum() == len(non_null_values):
                # For integer-like data, treat as numeric
                if np.all(numeric_converted.astype(float).mod(1) == 0):
                    return "numeric"
                # For decimal data, check if numeric form adds information
                elif (
                    numeric_converted.nunique()
                    > non_null_values.astype(str).nunique() * 0.5
                ):
                    return "numeric"
            return "categorical"
        except:
            return "categorical"

    def load_all_data(self) -> dict:
        """
        Load and process all data files specified in configuration.

        Finds and processes all DHS data files for the specified countries
        and years, then harmonizes them by retaining only columns that are
        present and valid across all datasets.

        Returns:
            Dictionary mapping country_year keys to processed DataFrames
        """
        dfs = {}
        from src.utils.paths import get_project_root

        project_root = get_project_root()
        for country, years in self.config["country_year"].items():
            country = f"{country}_Data"
            years = [years] if isinstance(years, str) else years

            for year in years:
                year_str = str(year)
                key = f"{country}_{year_str}"
                data_path = project_root / "data" / "DHS" / country / year_str

                if not data_path.exists():
                    logger.warning(f"Directory not found: {data_path}")
                    continue

                data_file = next(data_path.glob("*.DTA"), None)
                if data_file:
                    year_int = int(year_str)
                    dfs[key] = self.load_data(data_file, year_int)

        # Process common columns
        common_columns = self._get_common_columns(dfs)
        dfs = self._filter_to_common_columns(dfs, common_columns)

        return dfs

    def load_data(self, file_path: Path, year: int) -> pd.DataFrame:
        """
        Load and preprocess a single DHS data file.

        Loads the specified file, applies missing value thresholds to remove
        columns with insufficient data, standardizes values using replacement
        mappings, and adds a year identifier column.

        Args:
            file_path: Path to the DHS data file (.DTA format)
            year: Survey year to add as 'hv007' column

        Returns:
            Preprocessed DataFrame with standardized values
        """
        df = pd.read_stata(file_path, columns=self.config["columns_to_include"])
        df["hv007"] = year

        # Calculate and log initial missing data percentages
        missing_pct = df.isnull().mean()

        # Identify columns to drop
        cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index

        if len(cols_to_drop) > 0:
            logger.info(
                f"Dropping {len(cols_to_drop)} columns with >{self.missing_threshold*100}% missing data from {file_path.name}: {list(cols_to_drop)}"
            )
            df = df.drop(columns=cols_to_drop)

        # Validate remaining columns
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            if non_null_count == 0:
                logger.warning(
                    f"Column {col} has no valid data after preprocessing in {file_path.name}"
                )
                df = df.drop(columns=[col])

        # Apply value replacements
        df = self._replace_values(df)
        return df

    def _replace_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace values in the DataFrame according to replacement rules.

        Applies both special column-specific mappings and general value
        replacements, handling numeric and categorical columns appropriately
        to maintain data type integrity.

        Args:
            df: Input DataFrame to process

        Returns:
            DataFrame with standardized values
        """
        df_processed = df.copy()

        for col in df_processed.columns:
            # Check for special mapping first
            if col in self.special_mappings:
                df_processed[col] = self._apply_special_mapping(df_processed[col], col)
                continue

            # Regular replacement logic for other columns
            col_type = self.determine_column_type(df_processed[col])

            # Handle numeric columns with numeric replacements
            if col_type == "numeric":
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
                for old_val, new_val in self.replace_values.items():
                    if isinstance(new_val, (int, float)):
                        mask = df_processed[col] == old_val
                        df_processed.loc[mask, col] = new_val
                df_processed[col] = pd.to_numeric(
                    df_processed[col], errors="coerce"
                ).astype("float64")
            else:
                # Handle string/categorical columns with string replacements
                temp_series = df_processed[col].astype(str)
                for old_val, new_val in self.replace_values.items():
                    if not isinstance(new_val, (int, float)) or pd.isna(new_val):
                        mask = temp_series == str(old_val)
                        temp_series.loc[mask] = (
                            np.nan if pd.isna(new_val) else str(new_val)
                        )
                df_processed[col] = temp_series.astype("string")
        return df_processed

    def _get_common_columns(self, dfs: Dict[str, pd.DataFrame]) -> Set[str]:
        """
        Find columns that are present and valid across all datasets.

        Identifies columns that exist in all datasets and contain at least
        some valid (non-missing) values in each dataset, to ensure that
        only usable common columns are retained.

        Args:
            dfs: Dictionary of DataFrames to analyze

        Returns:
            Set of column names common to all datasets
        """
        if not dfs:
            return set()

        # Log initial column counts
        logger.info("Initial column counts per dataset:")
        for key, df in dfs.items():
            logger.info(f"{key}: {len(df.columns)} columns")

        # Check for columns with all missing values across datasets
        problematic_columns = set()
        for key, df in dfs.items():
            for col in df.columns:
                if df[col].isna().all():
                    problematic_columns.add(col)

        # Remove problematic columns from consideration
        for key in dfs:
            dfs[key] = dfs[key].drop(columns=list(problematic_columns), errors="ignore")

        all_column_sets = [set(df.columns) for df in dfs.values()]
        common_columns = set.intersection(*all_column_sets)

        # Validate common columns
        for col in common_columns:
            for key, df in dfs.items():
                non_null_count = df[col].notna().sum()
                if non_null_count == 0:
                    logger.warning(
                        f"Common column {col} has no valid data in dataset {key}"
                    )
                    common_columns.remove(col)
                    break

        if dfs:
            first_df_columns = set(next(iter(dfs.values())).columns)
            dropped_columns = first_df_columns - common_columns

            if dropped_columns:
                logger.info(
                    f"Columns dropped due to inconsistency across datasets: {sorted(list(dropped_columns))}"
                )
            logger.info(
                f"Retained {len(common_columns)} common columns across all datasets"
            )

        return common_columns

    def _filter_to_common_columns(
        self, dfs: Dict[str, pd.DataFrame], common_columns: Set[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter all DataFrames to include only the common column set.

        Creates new DataFrames containing only the columns identified as
        common across all datasets, with a final validation to ensure
        data quality.

        Args:
            dfs: Dictionary of DataFrames to filter
            common_columns: Set of column names to retain

        Returns:
            Dictionary of filtered DataFrames
        """
        filtered_dfs = {}

        for key, df in dfs.items():
            filtered_df = df[list(common_columns)].copy()

            # Final validation of filtered dataframe
            for col in filtered_df.columns:
                non_null_count = filtered_df[col].notna().sum()
                if non_null_count == 0:
                    logger.error(
                        f"Column {col} in dataset {key} has no valid data after filtering"
                    )

            filtered_dfs[key] = filtered_df

        return filtered_dfs
