import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import logging
from typing import Dict, Any, Tuple, List
from joblib import Parallel, delayed
import multiprocessing
from time import time
from tqdm import tqdm
import sys

###TODO: add the datatype conversion to be the same as in the MICE method
###TODO: decide if we want to increase the number of combinations generated

logger = logging.getLogger("wealth_index.imputer")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


def setup_logging(level_str: str) -> int:
    """Setup logging configuration."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(level_str.upper(), logging.INFO)
    multiprocessing.log_to_stderr(level=level)
    return level


class KNNImputer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = config.get("verbose", 0)
        self.n_jobs = config.get("n_jobs", multiprocessing.cpu_count() - 1)
        self.skip_optimizing = config.get("skip_optimizing", False)
        setup_logging(config.get("log_level", "INFO"))

    def determine_column_type(self, series: pd.Series) -> str:
        """
        Definitively determine if a column should be numeric or categorical.
        Returns 'numeric' only if all values can be converted to numbers.

        Args:
            series (pd.Series): Input pandas Series to analyze

        Returns:
            str: 'numeric' or 'categorical'
        """
        # If series is empty, return categorical
        if len(series) == 0:
            return "categorical"

        # If already numeric dtype, verify all values are actually numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check for any infinity values
            if np.any(np.isinf(series.replace([np.inf, -np.inf], np.nan))):
                return "categorical"
            return "numeric"

        # Drop null values for checking
        non_null_values = series.dropna()

        # If all values are null, return categorical
        if len(non_null_values) == 0:
            return "categorical"

        try:
            # Attempt to convert to numeric
            numeric_converted = pd.to_numeric(non_null_values, errors="coerce")

            # Check if any values were coerced to NaN
            if numeric_converted.isna().any():
                return "categorical"

            # Check for infinity values
            if np.any(np.isinf(numeric_converted)):
                return "categorical"

            # If we got here, all values are valid numbers
            return "numeric"

        except (ValueError, TypeError):
            return "categorical"

    def _finalize_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure consistent column types throughout the DataFrame.
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

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for imputation by handling IDs and converting types."""
        df_imputed = df.copy()
        id_cols = ["hv000", "hv001", "hv005", "hv007"]
        df_ids = df_imputed[id_cols]
        df_imputed = df_imputed.drop(columns=id_cols)

        for col in df_imputed.columns:
            df_imputed.loc[df_imputed[col] == "nan", col] = np.nan
            mask = df_imputed[col].isna()
            temp_series = df_imputed[col].astype(str)
            temp_series[mask] = np.nan
            df_imputed[col] = temp_series

        return df_imputed, df_ids

    def _prepare_features(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
        """Prepare features for KNN imputation."""
        features = df.drop(columns=[target_column])
        features_processed = features.copy()

        for col in features.columns:
            if self.determine_column_type(features[col]) == "numeric":
                median_value = pd.to_numeric(features[col], errors="coerce").median()
                features_processed[col] = pd.to_numeric(
                    features[col], errors="coerce"
                ).fillna(median_value)
            else:
                mode_value = features[col].mode().iloc[0]
                features_processed[col] = features[col].fillna(mode_value).astype(str)

        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float32
        )
        return encoder.fit_transform(features_processed)

    def _generate_k_values(self, n_samples: int, col_name: str) -> List[int]:
        """Generate strategic k values for KNN."""
        center_k = int(np.sqrt(n_samples))
        center_k = max(3, min(center_k, 25))
        if center_k % 2 == 0:
            center_k += 1

        min_k = 3
        max_k = min(n_samples // 3, 51)

        small_k = list(range(min_k, min(11, max_k), 2))
        medium_k = [
            k + (k % 2 == 0)
            for k in [max(11, center_k - 4), center_k, min(center_k + 4, max_k)]
        ]
        large_k = list(range(min(center_k + 6, max_k - 4), max_k + 1, 2))[:3]

        return sorted(set(small_k + medium_k + large_k))

    def _validate_parameters(
        self, X_train: np.ndarray, y_train: pd.Series, params: Dict[str, Any]
    ) -> float:
        """Validate parameters using appropriate validation strategy."""
        min_class_count = pd.Series(y_train).value_counts().min()

        if min_class_count < 2:
            from sklearn.model_selection import train_test_split

            ###TODO: decide on 0.2 here
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2)
            knn = KNeighborsClassifier(**params)
            knn.fit(X_t, y_t)
            return knn.score(X_v, y_v)
        else:
            knn = KNeighborsClassifier(**params)
            return np.mean(
                cross_val_score(knn, X_train, y_train, cv=2, scoring="accuracy")
            )

    def _get_optimal_params(
        self, X_train: np.ndarray, y_train: pd.Series, col_type: str, col_name: str
    ) -> Dict[str, Any]:
        """Find optimal parameters for KNN imputation."""
        n_samples = len(y_train)
        if self.skip_optimizing:
            return {
                "weights": "distance" if n_samples > 1000 else "uniform",
                "leaf_size": (
                    30 if n_samples >= 1000 else (20 if n_samples >= 100 else 10)
                ),
                "algorithm": "auto",
                "n_jobs": 1,
                "n_neighbors": 3,
                "metric": "hamming" if col_type == "categorical" else "euclidean",
            }

        k_values = self._generate_k_values(n_samples, col_name)
        metrics = ["hamming"] if col_type == "categorical" else ["euclidean", "cosine"]

        constant_params = {
            "weights": "distance" if n_samples > 1000 else "uniform",
            "leaf_size": 30 if n_samples >= 1000 else (20 if n_samples >= 100 else 10),
            "algorithm": "auto",
            "n_jobs": 1,
        }

        best_score = float("-inf")
        best_params = None
        total = len(k_values) * len(metrics)
        start_time = time()
        finished = 0
        for k in k_values:
            for metric in metrics:
                params = {
                    "n_neighbors": k,
                    "metric": metric,
                    "p": 2 if metric == "euclidean" else None,
                    **constant_params,
                }
                score = self._validate_parameters(X_train, y_train, params)
                if score > best_score:
                    best_score = score
                    best_params = params
                finished += 1
                logger.info(
                    f"Finished {finished} parameter sets out of {total} for {col_name} in {time()-start_time}s."
                )

        return best_params or {
            "n_neighbors": 3,
            "metric": "hamming" if col_type == "categorical" else "euclidean",
            **constant_params,
        }

    def _restore_column_types(
        self, df: pd.DataFrame, original_dtypes: dict, numerical_columns: list
    ) -> pd.DataFrame:
        """
        Restore original column types after imputation.
        """
        df_restored = df.copy()

        for column, dtype in original_dtypes.items():
            if column in numerical_columns:
                df_restored[column] = pd.to_numeric(
                    df_restored[column], errors="coerce"
                )
            else:
                df_restored[column] = df_restored[column].astype(dtype)
        return df_restored

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values while preserving original data types."""
        logger.info("Starting KNN imputation")

        # Store original types
        original_dtypes = df.dtypes.to_dict()
        numerical_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        # Prepare data
        df_imputed, df_ids = self._prepare_data(df)

        # Find columns to impute
        columns_with_na = [col for col in df.columns if df[col].isna().any()]
        if not columns_with_na:
            logger.info("No missing values found")
            return self._finalize_column_types(pd.concat([df_imputed, df_ids], axis=1))

        # Perform imputation
        logger.info(f"Columns to impute: {len(columns_with_na)}")
        with tqdm(total=len(columns_with_na), desc="Imputing columns") as pbar:
            results = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads"
            )(
                delayed(self._impute_column)(df_imputed, column, pbar)
                for column in columns_with_na
            )

        # Update imputed values
        for column, imputed_values in zip(columns_with_na, results):
            df_imputed[column] = imputed_values

        # Combine with IDs and restore types
        df_imputed = pd.concat([df_imputed, df_ids], axis=1)
        return self._restore_column_types(
            df_imputed, original_dtypes, numerical_columns
        )

    def _impute_column(
        self, df: pd.DataFrame, target_column: str, pbar: tqdm
    ) -> pd.Series:
        """Impute missing values for a single column."""
        start_time = time()
        logger.info(f"Imputing column {target_column}")

        y = df[target_column].copy()
        col_type = self.determine_column_type(y)

        if col_type == "numeric":
            y = pd.to_numeric(y, errors="coerce")
        else:
            mask = y.isna()
            y = y.astype(str)
            y[mask] = np.nan

        y_train = y[y.notna()]
        X = self._prepare_features(df, target_column)
        X_train = X[y.notna()]

        optimal_params = self._get_optimal_params(
            X_train, y_train, col_type, target_column
        )
        knn = KNeighborsClassifier(**optimal_params)
        knn.fit(X_train, y_train)
        null_mask = y.isna()
        if null_mask.any():
            imputed_values = knn.predict(X[null_mask])
            y[null_mask] = imputed_values

        logger.info(f"Finished imputing {target_column} after {time() - start_time}s")
        pbar.update(1)

        return (
            pd.to_numeric(y, errors="coerce").astype("float64")
            if col_type == "numeric"
            else y.astype("string")
        )
