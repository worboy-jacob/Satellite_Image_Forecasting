"""
Base implementation for data imputation methods.

Provides a flexible framework for imputing missing values in survey data
with configurable optimization strategies, parallel processing, and
performance tracking. Supports multiple imputation methods through a common
workflow architecture.
"""

import pandas as pd
import numpy as np
import logging
import sys
import multiprocessing
from typing import Dict, Any, List, Tuple, Union, Callable
from time import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from joblib import Parallel, delayed
import gc

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


class PerformanceTracker:
    """
    Tracks performance metrics for imputation processes.

    Records timing information, convergence statistics, and other metrics
    to evaluate imputation performance across different columns.
    """

    def __init__(self):
        """Initialize performance tracking with empty statistics containers."""
        self.timings = {}
        self.column_stats = {}
        self.total_start_time = time()

    def start_column(self, column: str):
        """
        Record the start time for processing a column.

        Args:
            column: Name of the column being processed
        """
        if column not in self.timings:
            self.timings[column] = {}
        self.timings[column]["start"] = time()
        if column not in self.column_stats:
            self.column_stats[column] = {
                "iterations": 0,
                "convergence_metrics": [],
                "parameter_combinations": 0,
            }

    def update_column(self, column: str, stats: Dict[str, Any]):
        """
        Update column statistics with additional metrics.

        Args:
            column: Name of the column being processed
            stats: Dictionary of statistics to update or add
        """
        if column not in self.column_stats:
            self.column_stats[column] = {}
        self.column_stats[column].update(stats)

    def finish_column(self, column: str):
        """
        Record the end time and calculate processing duration for a column.

        Args:
            column: Name of the column being processed
        """
        if column in self.timings:
            end_time = time()
            self.timings[column]["end"] = end_time
            self.timings[column]["duration"] = end_time - self.timings[column].get(
                "start", end_time
            )

    def get_summary(self) -> Dict:
        """
        Get summary statistics for all processed columns.

        Returns:
            Dictionary containing timing and performance metrics
        """
        total_time = time() - self.total_start_time

        # Safely calculate average duration
        durations = []
        for col, timing in self.timings.items():
            if "duration" in timing:
                durations.append(timing["duration"])

        avg_duration = np.mean(durations) if durations else 0.0

        return {
            "total_time": total_time,
            "average_duration": avg_duration,
            "column_timings": self.timings,
            "column_stats": self.column_stats,
        }


class BaseImputer:
    """
    Base class for all imputation methods with common functionality.

    Provides core methods for data preparation, parameter optimization,
    and imputation workflows that are shared across different imputation
    implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = config.get("verbose", 0)
        self.n_jobs = config.get("n_jobs", multiprocessing.cpu_count() - 1)
        self.skip_optimizing = config.get("skip_optimizing", False)
        self.tracker = PerformanceTracker()
        setup_logging(config.get("log_level", "INFO"))
        self.id_cols = ["hv000", "hv001", "hv005", "hv007"]

    def determine_column_type(self, series: pd.Series) -> str:
        """
        Determine if a column should be treated as numeric or categorical.

        Makes a definitive determination by attempting numeric conversion
        on non-null values and checking for conversion failures.

        Args:
            series: The pandas Series to analyze

        Returns:
            'numeric' if all non-null values can be converted to numbers,
            'categorical' otherwise
        """
        series_test = series.copy()

        non_null_values = series_test.dropna()

        try:
            numeric_converted = pd.to_numeric(non_null_values, errors="coerce")
            if numeric_converted.isna().any() or np.any(np.isinf(numeric_converted)):
                return "categorical"
            return "numeric"
        except (ValueError, TypeError):
            return "categorical"

    def _preprocess_datatypes(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Preprocess dataframe to optimize datatypes for imputation.

        Converts columns to appropriate types while preserving NaN values,
        tracking the original and converted types for later restoration.

        Args:
            df: Input DataFrame to preprocess

        Returns:
            Tuple containing:
            - DataFrame with optimized data types
            - Dictionary mapping column names to their type ('numeric' or 'category')
        """
        df_processed = df.copy()
        column_types = {}

        for column in df.columns:
            # Store original NaN positions
            original_na_mask = df[column].isna()

            try:
                # Try converting to numeric
                numeric_series = pd.to_numeric(df[column], errors="raise")
                # If successful, convert to float
                df_processed[column] = numeric_series.astype("float64")
                # Restore original NaN positions
                df_processed.loc[original_na_mask, column] = np.nan
                column_types[column] = "numeric"
            except (ValueError, TypeError):
                # If conversion fails, convert to string category
                non_na_mask = ~original_na_mask
                df_processed.loc[non_na_mask, column] = df.loc[
                    non_na_mask, column
                ].astype(str)
                # Restore original NaN positions
                df_processed.loc[original_na_mask, column] = np.nan
                column_types[column] = "category"

        return df_processed, column_types

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for imputation by separating IDs and converting types.

        Extracts ID columns, handles "nan" strings, and converts columns to
        appropriate types for imputation processing.

        Args:
            df: Input DataFrame with potential missing values

        Returns:
            Tuple containing:
            - DataFrame with data ready for imputation
            - DataFrame with extracted ID columns
        """
        df_imputed = df.copy()

        # Check if ID columns exist in the dataframe
        existing_id_cols = [col for col in self.id_cols if col in df_imputed.columns]
        df_ids = df_imputed[existing_id_cols] if existing_id_cols else pd.DataFrame()
        df_imputed = df_imputed.drop(columns=existing_id_cols, errors="ignore")

        # Identify numeric columns first
        numeric_columns = []
        for col in df_imputed.columns:
            if self.determine_column_type(df_imputed[col]) == "numeric":
                numeric_columns.append(col)

        # Handle "nan" strings and proper typing
        for col in df_imputed.columns:
            df_imputed.loc[df_imputed[col] == "nan", col] = np.nan

            if col in numeric_columns:
                # For numeric columns, convert to numeric and preserve NaN
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors="raise")
            else:
                # For categorical columns, convert to string but preserve NaN
                mask = df_imputed[col].isna()
                temp_series = df_imputed[col].astype(str)
                temp_series[mask] = np.nan
                df_imputed[col] = temp_series

        return df_imputed, df_ids

    def _create_validation_masks(
        self, df: pd.DataFrame, columns: List[str]
    ) -> Dict[str, pd.Series]:
        """
        Create validation masks for parameter optimization.

        Randomly selects a portion of non-missing values in each column to use
        for validating imputation quality during parameter optimization.

        Args:
            df: Input DataFrame
            columns: List of column names to create masks for

        Returns:
            Dictionary mapping column names to boolean Series where True indicates
            values to use for validation
        """
        masks = {}
        for column in columns:
            # Create mask for non-missing values
            non_missing = ~df[column].isna()
            if non_missing.sum() > 0:
                # Randomly select 10% of non-missing values for validation
                validation_size = max(int(non_missing.sum() * 0.1), 1)
                validation_indices = np.random.choice(
                    np.where(non_missing)[0], size=validation_size, replace=False
                )
                mask = pd.Series(False, index=df.index)
                mask.iloc[validation_indices] = True
                masks[column] = mask
            else:
                masks[column] = pd.Series(False, index=df.index)
        return masks

    def _prepare_prediction_features(
        self, df: pd.DataFrame, target_column: str
    ) -> np.ndarray:
        """
        Prepare feature matrix for prediction models.

        Converts and encodes predictor columns appropriately for use in
        imputation models, handling both numeric and categorical features.

        Args:
            df: DataFrame containing the data
            target_column: Name of the column to be predicted

        Returns:
            numpy.ndarray containing the prepared feature matrix
        """
        df_to_encode = df.copy()
        predictor_columns = [
            col for col in df_to_encode.columns if col != target_column
        ]
        numeric_predictors = [
            col
            for col in predictor_columns
            if self.determine_column_type(df_to_encode[col]) == "numeric"
        ]
        # Convert all numeric predictors to numeric data
        for col in numeric_predictors:
            df_to_encode[col] = pd.to_numeric(df_to_encode[col], errors="raise").astype(
                "float64"
            )
        categorical_predictors = [
            col for col in predictor_columns if col not in numeric_predictors
        ]
        for col in categorical_predictors:
            df_to_encode[col] = df_to_encode[col].astype(str)

        X_parts = []
        if numeric_predictors:
            # Handle NaN values for numeric predictors
            numeric_df = df_to_encode[numeric_predictors].copy()
            for col in numeric_predictors:
                df_to_encode[col] = pd.to_numeric(
                    df_to_encode[col], errors="raise"
                ).astype("float64")
                if numeric_df[col].isna().any():
                    numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
            X_numeric = numeric_df.astype(float).values
            X_parts.append(X_numeric)

        if categorical_predictors:
            # Handle NA values for categorical predictors before encoding
            categorical_df = df_to_encode[categorical_predictors].copy()
            for col in categorical_predictors:
                if categorical_df[col].isna().any():
                    # Fill NA with the most common value
                    most_common = (
                        categorical_df[col].mode().iloc[0]
                        if not categorical_df[col].dropna().empty
                        else "missing"
                    )
                    categorical_df[col] = categorical_df[col].fillna(most_common)

            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_categorical = encoder.fit_transform(categorical_df)
            X_parts.append(X_categorical)

        if not X_parts:
            # Handle the case when there are no usable predictors
            return np.zeros((len(df), 1))  # Return a dummy feature matrix

        return np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]

    def _restore_column_types(
        self, df: pd.DataFrame, original_dtypes: dict, numerical_columns: list
    ) -> pd.DataFrame:
        """
        Restore original column types after imputation.

        Converts imputed data back to original datatypes to maintain
        consistency with the input data.

        Args:
            df: DataFrame with imputed values
            original_dtypes: Dictionary mapping column names to original datatypes
            numerical_columns: List of columns that should be numeric

        Returns:
            DataFrame with restored column types
        """
        df_restored = df.copy()

        for column, dtype in original_dtypes.items():
            if column in df_restored.columns:  # Only process columns that exist
                if column in numerical_columns:
                    df_restored[column] = pd.to_numeric(
                        df_restored[column], errors="coerce"
                    )
                else:
                    df_restored[column] = df_restored[column].astype(dtype)
        return df_restored

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        target_column: str,
        params: Dict[str, Any],
        validation_mask: pd.Series,
        country_year: str,
        impute_func: Callable,
    ) -> float:
        """
        Validate imputation parameters using artificial missing values.

        Tests imputation quality by artificially removing known values,
        imputing them, and measuring the error between imputed and actual values.

        Args:
            df: DataFrame containing the data
            target_column: Column to validate
            params: Parameter configuration to test
            validation_mask: Boolean mask indicating values to use for validation
            country_year: Country-year identifier for logging
            impute_func: Function to perform imputation

        Returns:
            Normalized error score (0-1 scale) where lower is better
        """
        df_validation = df.copy()
        df_validation.loc[validation_mask, target_column] = np.nan
        true_values = df.loc[validation_mask, target_column]

        start_time = time()
        imputed_values = impute_func(df_validation, target_column, params, country_year)
        processing_time = time() - start_time

        if self.determine_column_type(df[target_column]) == "numeric":
            # For numeric columns, normalize RMSE by the range of the data
            data_range = df[target_column].max() - df[target_column].min()
            if data_range == 0:  # Handle constant columns
                data_range = 1.0

            rmse = np.sqrt(
                mean_squared_error(true_values, imputed_values[validation_mask])
            )
            # Normalize to 0-1 scale
            error = min(rmse / data_range, 1.0)
        else:
            # For categorical columns, already 0-1 scale (misclassification rate)
            error = 1 - accuracy_score(true_values, imputed_values[validation_mask])

        logger.info(
            f"Validation for {country_year} {target_column} - "
            f"Parameters: {params}, "
            f"Normalized Error: {error:.4f}, "
            f"Time: {processing_time:.2f}s"
        )

        return error

    def _safe_fill_values(
        self, series: pd.Series, values: np.ndarray, missing_mask: pd.Series
    ) -> pd.Series:
        """
        Safely fill values while respecting the original dtype.

        Handles type conversion and compatibility issues when filling
        missing values in different types of Series.

        Args:
            series: Series to fill values in
            values: Array of values to use for filling
            missing_mask: Boolean mask indicating positions to fill

        Returns:
            Series with filled values and preserved dtype where possible
        """
        result = series.copy()

        # Handle string/categorical types
        if pd.api.types.is_string_dtype(series.dtype):
            # Convert numeric values to strings for string columns
            values = np.array([str(v) for v in values])
        elif pd.api.types.is_categorical_dtype(series.dtype):
            # Handle categorical data
            values = pd.Categorical(values, categories=series.cat.categories)
        elif pd.api.types.is_integer_dtype(series.dtype):
            # Round values for integer types
            values = np.round(values).astype(series.dtype)

        result[missing_mask] = values
        return result

    def _safe_fill_numeric(self, series: pd.Series, fill_value: float) -> pd.Series:
        """
        Safely fill numeric series while preserving dtype compatibility.

        Handles special cases like integer types that can't represent NaN
        and string types that need conversion.

        Args:
            series: Series to fill values in
            fill_value: Value to use for filling

        Returns:
            Series with filled values and preserved dtype where possible
        """
        # If the series is string type, handle differently
        if pd.api.types.is_string_dtype(series.dtype):
            # For string types, convert fill value to string
            return series.fillna(str(fill_value))

        original_dtype = series.dtype

        # Handle integer types
        if pd.api.types.is_integer_dtype(original_dtype):
            # Round the fill value for integer columns
            fill_value = int(round(fill_value))
            # Convert to float64 first to handle NaN, then convert back
            filled_series = series.astype("float64").fillna(fill_value)
            # Convert back to integer if possible
            if not filled_series.isna().any():
                return filled_series.astype(original_dtype)
            # Use nullable integer type if we still have NaN
            return filled_series.astype(f"Int{original_dtype.itemsize * 8}")

        # For float types, fill directly
        if pd.api.types.is_float_dtype(original_dtype):
            return series.fillna(fill_value)

        # For any other type, convert to string
        return series.astype(str).fillna(str(fill_value))

    def _check_convergence(
        self,
        current_imp: pd.Series,
        previous_imp: pd.Series,
        missing_mask: pd.Series,
        iteration: int,
        min_iterations: int = 3,
        convergence_threshold: float = 1e-3,
    ) -> Tuple[bool, float]:
        """
        Check convergence of iterative imputation.

        For numeric values: calculates normalized difference metric
        For categorical values: calculates proportion of values that changed

        Args:
            current_imp: Current imputed values
            previous_imp: Previous iteration's imputed values
            missing_mask: Mask indicating missing values
            iteration: Current iteration number
            min_iterations: Minimum iterations before checking convergence
            convergence_threshold: Threshold for determining convergence

        Returns:
            Tuple containing:
            - Boolean indicating whether convergence has been reached
            - Float convergence metric value
        """
        if iteration < min_iterations:
            return False, float("inf")

        current_values = current_imp[missing_mask]
        previous_values = previous_imp[missing_mask]

        if len(current_values) == 0:
            return True, 0.0

        if self.determine_column_type(current_imp) == "numeric":
            # For numeric data, use appropriate metric based on implementation
            diff = np.abs(current_values - previous_values)
            # Use relative difference for MICE
            rel_diff = diff / (np.abs(previous_values) + 1e-10)
            metric = np.mean(rel_diff)
            has_converged = metric < convergence_threshold
        else:
            # For categorical data, use change rate
            changes = current_values != previous_values
            change_rate = np.mean(changes)
            has_converged = change_rate < convergence_threshold
            metric = change_rate

        logger.debug(
            f"Iteration {iteration}: Convergence metric = {metric}, Has converged = {has_converged}"
        )
        return has_converged, metric

    def _calculate_weighted_score(
        self, column_scores: Dict[str, float], missing_values: pd.Series
    ) -> float:
        """
        Calculate weighted imputation score based on missing value counts.

        Weights each column's score by the proportion of total missing values
        it represents, giving more importance to columns with more missing data.

        Args:
            column_scores: Dictionary mapping column names to their imputation scores
            missing_values: Series containing count of missing values per column

        Returns:
            Weighted average score across all columns
        """
        score_series = pd.Series(column_scores)
        total_missing = missing_values.sum()

        if total_missing == 0:
            return 0.0

        columns_missing_values = {
            col: missing_values[col] for col in score_series.index
        }
        column_missing_series = pd.Series(columns_missing_values)
        weighted_scores = score_series * column_missing_series / total_missing
        total_score = weighted_scores.sum()

        logger.info(f"Overall weighted imputation score: {total_score:.4f}")
        return total_score

    def _process_columns_with_optimal_params(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        optimal_params: Dict[str, Dict[str, Any]],
        original_dtypes: Dict[str, Any],
        validation_masks: Dict[str, pd.Series],
        imputer_func: Callable,
        country_year: str,
    ) -> pd.DataFrame:
        """
        Process multiple columns using their optimal parameters in parallel.

        Applies the imputation function to each column with missing values
        using the optimal parameters determined during optimization.

        Args:
            df: DataFrame to impute
            columns_with_na: List of columns with missing values
            optimal_params: Dictionary mapping columns to their optimal parameters
            original_dtypes: Dictionary of original column data types
            validation_masks: Dictionary of validation masks for each column
            imputer_func: Function to impute a single column
            country_year: Country-year identifier for logging

        Returns:
            DataFrame with imputed values for all columns
        """
        from joblib import Parallel, delayed

        df_processed = df.copy()

        with tqdm(total=len(columns_with_na), desc="Imputing columns") as pbar:
            results = Parallel(
                n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose
            )(
                delayed(self._process_single_column)(
                    df_processed.copy(),
                    column,
                    validation_masks[column],
                    original_dtypes[column],
                    optimal_params[column],
                    pbar,
                    imputer_func,
                    country_year,
                )
                for column in columns_with_na
            )

        # Update imputed values
        for column, imputed_values in zip(columns_with_na, results):
            df_processed[column] = imputed_values.astype(original_dtypes[column])

        return df_processed

    def _initial_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initially fill missing values to prepare for iterative imputation.

        Fills numeric columns with mean values and categorical columns with
        mode values to provide a starting point for more sophisticated imputation.

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with initially filled values
        """
        df_filled = df.copy()

        for col in df.columns:
            if df[col].isna().any():
                col_type = self.determine_column_type(df[col])

                if col_type == "numeric":
                    # For numeric columns, fill with mean
                    series = pd.to_numeric(df[col], errors="raise")
                    df_filled[col] = series.fillna(series.mean())
                else:
                    # For categorical columns, fill with mode
                    mode_value = (
                        df[col].mode().iloc[0]
                        if not df[col].dropna().empty
                        else "missing"
                    )
                    df_filled[col] = df[col].fillna(mode_value)

        return df_filled

    def _process_single_column(
        self,
        df: pd.DataFrame,
        column: str,
        validation_mask: pd.Series,
        original_dtype: np.dtype,
        optimal_params: Dict[str, Any],
        pbar: tqdm,
        impute_func: Callable,
        country_year: str,
    ) -> pd.Series:
        """
        Process a single column for imputation.

        Template method that handles performance tracking and logging while
        delegating the actual imputation to the provided function.

        Args:
            df: DataFrame containing the column to impute
            column: Name of the column to impute
            validation_mask: Mask for validation data
            original_dtype: Original data type of the column
            optimal_params: Optimal parameters for imputation
            pbar: Progress bar object
            impute_func: Function to perform the imputation
            country_year: Country-year identifier for logging

        Returns:
            Series with imputed values
        """
        self.tracker.start_column(column)

        # Get column type and missing percentage for logging
        column_type = self.determine_column_type(df[column])
        missing_pct = (df[column].isna().sum() / len(df)) * 100

        self.tracker.update_column(
            column,
            {
                "type": column_type,
                "missing_percentage": missing_pct,
                "optimal_params": optimal_params,
            },
        )

        # Perform imputation if impute_func is provided
        imputed_values = None
        if impute_func:
            imputed_values = impute_func(df, column, optimal_params, country_year)
        else:
            raise NotImplementedError(
                "Subclass must override this method or provide impute_func"
            )

        # Restore original dtype
        if column_type == "numeric":
            imputed_values = pd.to_numeric(imputed_values, errors="raise")

        self.tracker.finish_column(column)
        if pbar:
            pbar.update(1)

        return imputed_values

    def _generate_all_column_parameter_combinations(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        validation_masks: Dict[str, pd.Series],
        generate_params: Callable,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate all column and parameter combinations for testing.

        Creates a list of all possible combinations of columns with missing values
        and parameter configurations to evaluate during optimization.

        Args:
            df: DataFrame containing the data
            columns_with_na: List of columns with missing values
            validation_masks: Dictionary of validation masks
            generate_params: Function to generate parameter combinations

        Returns:
            List of tuples containing (column_name, parameter_dict)
        """
        all_combinations = []

        for column in columns_with_na:
            column_type = self.determine_column_type(df[column])
            n_samples = len(df[column])
            missing_pct = (df[column].isna().sum() / n_samples) * 100

            params_list = generate_params(n_samples, column_type, missing_pct, column)

            # Add each parameter combination with its column
            for params in params_list:
                all_combinations.append((column, params))

        return all_combinations

    def _find_optimal_parameters(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        validation_masks: Dict[str, pd.Series],
        country_year: str,
        impute_single_column: Callable,
        generate_params: Callable,
        calc_secondary_scores: Callable,
        get_imputer: Callable,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Find optimal parameters for all columns using parallel processing.

        Tests multiple parameter combinations for each column and selects
        the best configuration based on imputation quality metrics, using
        secondary scoring to break ties.

        Args:
            df: DataFrame containing the data
            columns_with_na: List of columns with missing values
            validation_masks: Dictionary of validation masks
            country_year: Country-year identifier for logging
            impute_single_column: Function to impute a single column
            generate_params: Function to generate parameter combinations
            calc_secondary_scores: Function to calculate secondary scores
            get_imputer: Function to get an imputer object

        Returns:
            Tuple containing:
            - Dictionary mapping columns to their optimal parameters
            - Dictionary mapping columns to their best scores
        """
        # Generate all combinations
        all_combinations = self._generate_all_column_parameter_combinations(
            df, columns_with_na, validation_masks, generate_params
        )

        logger.info(
            f"Generated {len(all_combinations)} total parameter combinations across all columns for {country_year}"
        )
        np.random.shuffle(all_combinations)

        # Process all combinations in parallel
        start_time = time()
        with tqdm(
            total=len(all_combinations), desc=f"Validating parameters {country_year}"
        ) as pbar:
            results = Parallel(
                n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose
            )(
                delayed(self._validate_single_combination)(
                    df.copy(),
                    column,
                    params,
                    validation_masks[column],
                    country_year,
                    impute_single_column,
                    pbar,
                )
                for column, params in all_combinations
            )

        # Group results by column
        column_results = {}
        for column, params, score in results:
            if column not in column_results:
                column_results[column] = []
            column_results[column].append((score, params))

        # Extract best parameters for each column
        optimal_params = {}
        best_primary_scores = {}  # Track the best scores for each column

        for column, result_list in column_results.items():
            # Sort by primary score (lower is better)
            result_list.sort(key=lambda x: x[0])
            best_primary_score = result_list[0][0]
            best_primary_scores[column] = best_primary_score

            # Find all parameter sets that tied for best primary score
            tied_params = [
                params for score, params in result_list if score == best_primary_score
            ]

            if len(tied_params) == 1:
                # No tie - use the best parameters directly
                optimal_params[column] = tied_params[0]
            else:
                # We have a tie - calculate secondary scores
                logger.info(
                    f"Found {len(tied_params)} tied configurations for {column} - using secondary scoring"
                )
                with tqdm(total=len(tied_params), desc="Breaking tie") as pbar:
                    secondary_results = Parallel(
                        n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose
                    )(
                        delayed(self._calculate_single_secondary_score)(
                            df.copy(),
                            column,
                            params,
                            get_imputer,
                            calc_secondary_scores,
                            pbar,
                        )
                        for params in tied_params
                    )
                # Choose the parameters with the best (lowest) secondary score
                optimal_params[column] = min(secondary_results, key=lambda x: x[0])[1]

        logger.info(f"Completed parameter optimization in {time() - start_time:.2f}s")
        return optimal_params, best_primary_scores

    def _validate_single_combination(
        self,
        df: pd.DataFrame,
        column: str,
        params: Dict[str, Any],
        validation_mask: pd.Series,
        country_year: str,
        impute_single_column: Callable,
        pbar: tqdm,
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Validate a single column-parameter combination.

        Tests how well a specific parameter configuration imputes a column
        by comparing imputed values to known values in the validation set.

        Args:
            df: DataFrame containing the data
            column: Name of the column to validate
            params: Parameter configuration to test
            validation_mask: Mask for validation data
            country_year: Country-year identifier for logging
            impute_single_column: Function to perform imputation
            pbar: Progress bar to update

        Returns:
            Tuple containing (column_name, parameter_dict, score)
        """
        score = self._validate_parameters(
            df,
            column,
            params,
            validation_mask,
            country_year,
            impute_single_column,
        )
        pbar.update(1)
        return column, params, score

    def _calculate_single_secondary_score(
        self,
        df: pd.DataFrame,
        column: str,
        params: Dict[str, Any],
        get_imputer: Callable,
        calc_secondary_scores: Callable,
        pbar: tqdm,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate secondary score for a parameter configuration.

        Used to break ties when multiple configurations have the same primary score,
        based on criteria like model complexity or stability.

        Args:
            df: DataFrame containing the data
            column: Name of the column
            params: Parameter configuration
            get_imputer: Function to get an imputer object
            calc_secondary_scores: Function to calculate secondary scores
            pbar: Progress bar to update

        Returns:
            Tuple containing (secondary_score, parameter_dict)
        """
        imputer = get_imputer(df[column], params)
        secondary_score = calc_secondary_scores(df, column, params, imputer)
        pbar.update(1)
        return secondary_score, params

    def _common_imputation_workflow(
        self,
        df: pd.DataFrame,
        imputation_name: str,
        generate_params: Callable,
        impute_single_column: Callable,
        calc_secondary_scores: Callable,
        country_year: str,
        get_imputer: Callable,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Common imputation workflow used by all imputation methods.

        Orchestrates the entire imputation process from data preparation through
        parameter optimization to final imputation and evaluation.

        Args:
            df: DataFrame to impute
            imputation_name: Name of the imputation method for logging
            generate_params: Function to generate parameter combinations
            impute_single_column: Function to impute a single column
            calc_secondary_scores: Function to calculate secondary scores
            country_year: Country-year identifier for logging
            get_imputer: Function to get an imputer object

        Returns:
            Tuple containing:
            - DataFrame with imputed values
            - Overall imputation quality score
        """
        self.tracker.start_column("total")
        for col in df.columns:
            col_type = self.determine_column_type(df[col])
            if col_type == "numeric":
                df[col] = pd.to_numeric(df[col], errors="raise")
            else:
                df[col] = df[col].astype("category")
        logger.info(f"Starting {imputation_name} imputation")

        # Prepare data for imputation
        df_imputed, df_ids = self._prepare_data(df)

        # Preprocess datatypes
        df_processed, column_types = self._preprocess_datatypes(df_imputed)
        original_dtypes = df_processed.dtypes.to_dict()

        # Calculate missing values statistics
        missing_values = df_processed.isna().sum()
        total_missing_values = missing_values.sum()

        # Find columns to impute
        columns_with_na = [
            col for col in df_processed.columns if df_processed[col].isna().any()
        ]

        if not columns_with_na:
            logger.info("No missing values found")
            result_df = (
                pd.concat([df_processed, df_ids], axis=1)
                if not df_ids.empty
                else df_processed
            )
            return result_df, 0.0

        # Create validation masks for parameter optimization
        validation_masks = self._create_validation_masks(df_processed, columns_with_na)

        # Find optimal parameters for all columns
        optimal_params, best_scores = self._find_optimal_parameters(
            df_processed,
            columns_with_na,
            validation_masks,
            country_year,
            impute_single_column,
            generate_params,
            calc_secondary_scores,
            get_imputer,
        )

        # Calculate weighted score
        total_score = self._calculate_weighted_score(best_scores, missing_values)

        # Perform imputation with optimal parameters
        df_processed = self._process_columns_with_optimal_params(
            df_processed,
            columns_with_na,
            optimal_params,
            original_dtypes,
            validation_masks,
            impute_single_column,
            country_year,
        )

        # Combine with ID columns
        if not df_ids.empty:
            df_processed = pd.concat([df_processed, df_ids], axis=1)

        self.tracker.finish_column("total")
        summary = self.tracker.get_summary()

        logger.info(
            f"Completed {imputation_name} imputation in {summary['total_time']:.2f}s\n"
            f"Average time per column: {summary['average_duration']:.2f}s"
        )

        return df_processed, total_score
