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
            # Check if column was originally numeric
            if df[col].dtype in ["int64", "float64"]:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols

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

        # Store original dtypes for restoration
        original_dtypes = df.dtypes.to_dict()

        # Identify numeric and categorical columns
        numeric_cols, categorical_cols = self._identify_column_types(df)
        logger.debug(f"Numeric columns: {numeric_cols}")
        logger.debug(f"Categorical columns: {categorical_cols}")

        # Get columns with missing values
        columns_with_na = [col for col in df.columns if df[col].isna().any()]

        if not columns_with_na:
            logger.info("No missing values found")
            return df_imputed

        logger.info(f"Found {len(columns_with_na)} columns with missing values")

        for column in columns_with_na:
            try:
                # Convert to string for KNN processing
                if column in numeric_cols:
                    df_imputed[column] = df_imputed[column].astype(str)

                # Perform imputation
                df_imputed[column] = self._impute_column(df_imputed, column)

                # Restore original type
                if column in numeric_cols:
                    df_imputed[column] = pd.to_numeric(
                        df_imputed[column], errors="coerce"
                    )
                    df_imputed[column] = df_imputed[column].astype(
                        original_dtypes[column]
                    )

                logger.info(f"Successfully imputed column: {column}")

            except Exception as e:
                logger.error(f"Error imputing column {column}: {e}")
                continue

        # Final type verification
        for col, dtype in original_dtypes.items():
            if col in df_imputed.columns:
                try:
                    if dtype in ["int64", "float64"]:
                        df_imputed[col] = df_imputed[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert {col} back to {dtype}: {e}")

        return df_imputed

    def _prepare_features(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
        """
        Prepare features for KNN imputation.

        Args:
            df: Input DataFrame
            target_column: Column being imputed

        Returns:
            Numpy array of prepared features
        """
        # Remove target column from features
        features = df.drop(columns=[target_column])

        # Convert all columns to string temporarily for consistent encoding
        features = features.astype(str)

        # Encode all features
        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float32
        )

        # Fill NA with a placeholder before encoding
        features_filled = features.fillna("MISSING_VALUE")

        return encoder.fit_transform(features_filled)

    def _impute_column(self, df: pd.DataFrame, target_column: str) -> pd.Series:
        """
        Impute missing values for a single column.

        Args:
            df: Input DataFrame
            target_column: Column to impute

        Returns:
            Series with imputed values
        """
        # Prepare target variable
        y = df[target_column].copy()

        # Prepare feature matrix
        X = self._prepare_features(df, target_column)

        # Split into training and imputation sets
        non_null_mask = y.notna()
        X_train = X[non_null_mask]
        y_train = y[non_null_mask]

        # Train KNN model
        knn = KNeighborsClassifier(
            n_neighbors=self.k, weights="uniform", algorithm="auto"
        )

        try:
            knn.fit(X_train, y_train)

            # Impute missing values
            null_mask = y.isna()
            if null_mask.any():
                X_missing = X[null_mask]
                imputed_values = knn.predict(X_missing)
                y[null_mask] = imputed_values

        except Exception as e:
            logger.error(f"KNN imputation failed for {target_column}: {e}")
            raise

        return y
