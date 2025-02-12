# wealth_index/data_processing/imputer.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger("wealth_index.imputer")


class KNNImputer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k = self.config.get("k_values", [5])[0]

    def determine_column_type(self, series: pd.Series) -> str:
        """
        Definitively determine if a column should be numeric or categorical.

        Returns:
            str: 'numeric' or 'categorical'
        """
        # First check if already numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # If not numeric, check if can be converted to numeric without data loss
        non_null_values = series.dropna()

        try:
            numeric_converted = pd.to_numeric(non_null_values, errors="coerce")
            # Check if conversion lost any data
            if numeric_converted.notna().sum() == len(non_null_values):
                # All values successfully converted to numeric
                # Additional check: if all values are integers
                if np.all(numeric_converted.astype(float).mod(1) == 0):
                    return "numeric"
                # If has decimals, check if they're meaningful
                elif (
                    numeric_converted.nunique()
                    > non_null_values.astype(str).nunique() * 0.5
                ):
                    return "numeric"
            return "categorical"
        except:
            return "categorical"

    def _identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical columns in DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (numeric column names, categorical column names)
        """
        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """
        Check if all non-NA values in a series can be converted to numbers.

        Args:
            series: Input pandas Series

        Returns:
            bool indicating if all values are numeric
        """
        try:
            clean_series = series.dropna()
            numeric_values = pd.to_numeric(clean_series, errors="coerce")
            return not numeric_values.isna().any()
        except:
            return False

    def _finalize_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize column types ensuring complete uniformity within each column.
        """
        df_final = df.copy()

        for col in df_final.columns:
            col_type = self.determine_column_type(df_final[col])

            if col_type == "numeric":
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce").astype(
                    "float64"
                )
            else:
                # Handle categorical by converting to string type
                mask = df_final[col].isna()
                temp_series = df_final[col].astype(str)
                temp_series[mask] = np.nan
                df_final[col] = temp_series.astype("string")

        return df_final

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values while preserving original data types.

        Args:
            df: Input DataFrame with missing values

        Returns:
            DataFrame with imputed values and proper types
        """
        logger.info("Starting KNN imputation")
        df_imputed = df.copy()
        # Convert all columns to string initially for consistent processing
        for col in df_imputed.columns:
            df_imputed.loc[df_imputed[col] == "nan", col] = np.nan
            if pd.api.types.is_categorical_dtype(df_imputed[col]):
                # Handle categorical columns properly
                mask = df_imputed[col].isna()
                temp_series = df_imputed[col].astype(str)
                temp_series[mask] = np.nan
                df_imputed[col] = temp_series
            else:
                temp_series = df_imputed[col].astype(str)
                df_imputed[col] = temp_series.where(temp_series != "nan", np.nan)

        # Get columns with missing values
        columns_with_na = [col for col in df.columns if df[col].isna().any()]

        if not columns_with_na:
            logger.info("No missing values found")
            return self._finalize_column_types(df_imputed)

        logger.info(f"Found {len(columns_with_na)} columns with missing values")

        for column in columns_with_na:
            try:
                df_imputed[column] = self._impute_column(df_imputed, column)
                logger.info(f"Successfully imputed column: {column}")
            except Exception as e:
                logger.error(f"Error imputing column {column}: {e}")
                continue

        df_imputed = self._finalize_column_types(df_imputed)
        return df_imputed

    def _prepare_features(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
        """
        Prepare features for KNN imputation.
        """
        features = df.drop(columns=[target_column])

        # Convert to string while properly handling categorical data
        features_processed = features.copy()
        for col in features.columns:
            if pd.api.types.is_categorical_dtype(features[col]):
                mask = features[col].isna()
                temp_series = features[col].astype(str)
                temp_series[mask] = "MISSING_VALUE"
                features_processed[col] = temp_series
            else:
                features_processed[col] = (
                    features[col].fillna("MISSING_VALUE").astype(str)
                )

        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float32
        )
        return encoder.fit_transform(features_processed)

    def _impute_column(self, df: pd.DataFrame, target_column: str) -> pd.Series:
        """
        Impute missing values for a single column.
        """
        original_dtype = df[target_column].dtype
        y = df[target_column].copy()

        # Determine and store column type before any processing
        col_type = self.determine_column_type(y)

        # Handle the column based on its determined type
        if col_type == "numeric":
            y = pd.to_numeric(y, errors="coerce")
            y_train = y[y.notna()]
        else:
            # For categorical data, ensure everything is string
            mask = y.isna()
            y = y.astype(str)
            y[mask] = np.nan
            y_train = y[y.notna()]

        X = self._prepare_features(df, target_column)
        non_null_mask = y.notna()
        X_train = X[non_null_mask]

        knn = KNeighborsClassifier(
            n_neighbors=min(self.k, len(y_train)), weights="uniform", algorithm="auto"
        )

        try:
            knn.fit(X_train, y_train)
            null_mask = y.isna()
            if null_mask.any():
                X_missing = X[null_mask]
                imputed_values = knn.predict(X_missing)

                # Ensure imputed values match the column type
                if col_type == "numeric":
                    imputed_values = pd.to_numeric(imputed_values, errors="coerce")
                else:
                    imputed_values = imputed_values.astype(str)

                y[null_mask] = imputed_values

            # Final type conversion
            if col_type == "numeric":
                return pd.to_numeric(y, errors="coerce").astype("float64")
            else:
                return y.astype("string")

        except Exception as e:
            logger.error(f"KNN imputation failed for {target_column}: {e}")
            raise
