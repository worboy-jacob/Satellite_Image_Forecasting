from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from time import time
import logging
import sys
import multiprocessing
from typing import Dict, Any, Tuple, List
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings

###TODO: decide if decreasing the number of combinations would be better for time purposes. Especially with the stronger desktop

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


class MICEImputer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = config.get("verbose", 0)
        self.n_jobs = config.get("n_jobs", multiprocessing.cpu_count() - 1)
        self.skip_optimizing = config.get("skip_optimizing", False)
        self.verbose = config.get("verbose", 0)
        self.tracker = PerformanceTracker()
        setup_logging(config.get("log_level", "INFO"))

    def _generate_parameter_combinations(
        self, n_samples: int, column_type: str, missing_pct: float, column_name: str
    ) -> List[Dict]:
        """
        Generate parameter combinations for a single column.
        """
        logger.info(
            f"Generating parameters for {column_name} ({column_type}, {missing_pct:.1f}% missing)"
        )

        # Determine n_impute values based on missing percentage and sample size
        if missing_pct < 10:
            n_impute_values = [3, 5]
        elif missing_pct < 30:
            n_impute_values = [3, 5, 7]
        elif missing_pct < 50:
            n_impute_values = [3, 5, 7, 9]
        else:
            n_impute_values = [5, 7, 9, 11]  # More imputations for heavily missing data

        # Determine iteration values based on sample size and missing percentage
        if n_samples < 1000:
            if missing_pct > 30:
                iter_values = [5, 10, 15, 20]
            else:
                iter_values = [5, 10, 15]
        elif n_samples < 10000:
            iter_values = [5, 10, 15]
        else:
            # For very large datasets, fewer iterations to manage computational cost
            iter_values = [5, 10, 15]

        # Expanded methods based on column type and data characteristics
        if column_type == "numeric":
            if missing_pct > 30:
                # For high missing rates, include more robust methods
                methods = [
                    "bayesian_ridge",
                    "linear",
                    "bayesian_ridge_warm",
                    "linear_scaled",
                ]
            else:
                methods = ["bayesian_ridge", "linear", "linear_scaled"]
        else:  # categorical
            if missing_pct > 30:
                # For high missing rates in categorical data
                methods = ["logistic", "logistic_balanced", "logistic_multinomial"]
            else:
                methods = ["logistic", "logistic_balanced"]

        # Generate all combinations
        combinations = []
        for n_iter in iter_values:
            for n_impute in n_impute_values:
                for method in methods:
                    # Add parameter variations based on method
                    if method == "bayesian_ridge_warm":
                        params = {
                            "n_iter": n_iter,
                            "n_impute": n_impute,
                            "method": "bayesian_ridge",
                            "warm_start": True,
                        }
                    elif method == "linear_scaled":
                        params = {
                            "n_iter": n_iter,
                            "n_impute": n_impute,
                            "method": "linear",
                            "scale_features": True,
                        }
                    elif method == "logistic_balanced":
                        params = {
                            "n_iter": n_iter,
                            "n_impute": n_impute,
                            "method": "logistic",
                            "class_weight": "balanced",
                        }
                    elif method == "logistic_multinomial":
                        params = {
                            "n_iter": n_iter,
                            "n_impute": n_impute,
                            "method": "logistic",
                            "multi_class": "multinomial",
                        }
                    else:
                        params = {
                            "n_iter": n_iter,
                            "n_impute": n_impute,
                            "method": method,
                        }
                    combinations.append(params)

        logger.info(
            f"Generated {len(combinations)} parameter combinations for {column_name}"
        )
        return combinations

    def _generate_all_column_parameter_combinations(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        validation_masks: Dict[str, pd.Series],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate all column and parameter combinations upfront."""
        all_combinations = []

        for column in columns_with_na:
            column_type = self.determine_column_type(df[column])
            n_samples = len(df[column])
            missing_pct = (df[column].isna().sum() / n_samples) * 100

            params_list = self._generate_parameter_combinations(
                n_samples, column_type, missing_pct, column
            )

            # Add each parameter combination with its column
            for params in params_list:
                all_combinations.append((column, params))

        return all_combinations

    def _validate_single_combination(
        self,
        df: pd.DataFrame,
        column: str,
        params: Dict[str, Any],
        validation_mask: pd.Series,
        pbar: tqdm,
    ) -> Tuple[str, Dict[str, Any], float]:
        """Validate a single column-parameter combination."""
        score = self._validate_parameters(df, column, params, validation_mask)
        pbar.update(1)
        return column, params, score

    def _get_optimal_params_for_all_columns(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        validation_masks: Dict[str, pd.Series],
    ) -> Dict[str, Dict[str, Any]]:
        """Get optimal parameters for all columns using parallel processing."""

        # Generate all combinations
        all_combinations = self._generate_all_column_parameter_combinations(
            df, columns_with_na, validation_masks
        )

        logger.info(
            f"Generated {len(all_combinations)} total parameter combinations across all columns"
        )
        np.random.shuffle(all_combinations)
        # Process all combinations in parallel
        start_time = time()
        with tqdm(total=len(all_combinations), desc="Validating parameters") as pbar:
            results = Parallel(
                n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose
            )(
                delayed(self._validate_single_combination)(
                    df.copy(), column, params, validation_masks[column], pbar
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
        for column, result_list in column_results.items():
            # Sort by primary score
            result_list.sort(key=lambda x: x[0])
            best_primary_score = result_list[0][0]

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
                            df.copy(), column, params, pbar
                        )
                        for params in tied_params
                    )
                # Choose the parameters with the best (lowest) secondary score
                optimal_params[column] = min(secondary_results, key=lambda x: x[0])[1]

        logger.info(f"Completed parameter optimization in {time() - start_time:.2f}s")
        return optimal_params

    def _calculate_single_secondary_score(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any], pbar: tqdm
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate secondary score for a single parameter set."""
        imputer = self._get_imputer(df[column], params["method"])
        secondary_score = self._calculate_secondary_score(df, column, params, imputer)
        pbar.update(1)
        return secondary_score, params

    def _calculate_secondary_score(
        self,
        df: pd.DataFrame,
        target_column: str,
        params: Dict[str, Any],
        imputer: Union[LinearRegression, LogisticRegression, BayesianRidge],
    ) -> float:
        """
        Calculate secondary score for tie-breaking.
        Returns a score between 0 and 1 (lower is better).
        """
        # Calculate prediction confidence if available
        confidence_score = 0.5  # Default neutral score
        if hasattr(imputer, "predict_proba"):
            try:
                # Get non-missing data for confidence calculation
                non_missing_mask = ~df[target_column].isna()
                X_observed = self._prepare_prediction_features(
                    df.loc[non_missing_mask], target_column
                )
                probas = imputer.predict_proba(X_observed)
                confidence_score = 1 - np.mean(
                    np.max(probas, axis=1)
                )  # Lower is better
            except:
                pass  # Keep default confidence score if prediction_proba fails

        # Complexity score based on parameters (normalized to 0-1)
        complexity_score = (
            params["n_iter"] / 20.0
        ) * 0.5 + (  # Penalize high iterations
            params["n_impute"] / 11.0
        ) * 0.5  # Penalize high imputation counts

        # Weighted combination (0.67 for confidence, 0.33 for complexity)
        return 0.67 * confidence_score + 0.33 * complexity_score

    def _prepare_prediction_features(
        self, df: pd.DataFrame, target_column: str
    ) -> np.ndarray:
        """Prepare feature matrix for prediction probability calculation."""
        predictor_columns = [col for col in df.columns if col != target_column]
        numeric_predictors = [
            col
            for col in predictor_columns
            if self.determine_column_type(df[col]) == "numeric"
        ]
        categorical_predictors = [
            col for col in predictor_columns if col not in numeric_predictors
        ]

        X_parts = []
        if numeric_predictors:
            X_numeric = df[numeric_predictors].astype(float).values
            X_parts.append(X_numeric)

        if categorical_predictors:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_categorical = encoder.fit_transform(df[categorical_predictors])
            X_parts.append(X_categorical)

        return np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        target_column: str,
        params: Dict[str, Any],
        validation_mask: pd.Series,
    ) -> float:
        """
        Validate MICE parameters for a single column using artificially created missing values.
        Returns normalized error between 0 and 1 for both categorical and numeric columns.
        """
        df_validation = df.copy()
        df_validation.loc[validation_mask, target_column] = np.nan
        true_values = df.loc[validation_mask, target_column]

        start_time = time()
        imputed_values = self._impute_single_column(
            df_validation, target_column, params
        )
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
            f"Validation for {target_column} - "
            f"Parameters: {params}, "
            f"Normalized Error: {error:.4f}, "
            f"Time: {processing_time:.2f}s"
        )

        return error

    def determine_column_type(self, series: pd.Series) -> str:
        """
        Definitively determine if a column should be numeric or categorical.
        Returns 'numeric' only if all values can be converted to numbers.
        """
        if len(series) == 0:
            return "categorical"

        if pd.api.types.is_numeric_dtype(series):
            if np.any(np.isinf(series.replace([np.inf, -np.inf], np.nan))):
                return "categorical"
            return "numeric"

        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return "categorical"

        try:
            numeric_converted = pd.to_numeric(non_null_values, errors="coerce")
            if numeric_converted.isna().any() or np.any(np.isinf(numeric_converted)):
                return "categorical"
            return "numeric"
        except (ValueError, TypeError):
            return "categorical"

    def _check_convergence(
        self,
        current_imp: pd.Series,
        previous_imp: pd.Series,
        missing_mask: pd.Series,
        iteration: int,
        min_iterations: int = 3,
        convergence_threshold: float = 1e-3,  # Added a parameter to control convergence threshold
    ) -> Tuple[bool, float]:
        """Simplified convergence checking."""
        if iteration < min_iterations:
            return False, float("inf")

        current_values = current_imp[missing_mask]
        previous_values = previous_imp[missing_mask]

        if len(current_values) == 0:
            return True, 0.0

        if self.determine_column_type(current_imp) == "numeric":
            # For numeric data, use relative difference
            diff = np.abs(current_values - previous_values)
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

    def _safe_fill_values(
        self, series: pd.Series, values: np.ndarray, missing_mask: pd.Series
    ) -> pd.Series:
        """
        Safely fill values while respecting the original dtype.

        Args:
            series: Original series to fill
            values: Values to insert
            missing_mask: Boolean mask indicating where to insert values

        Returns:
            Filled series with appropriate dtype
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

        Args:
            series: Original series to fill
            fill_value: Value to use for filling NaN

        Returns:
            Filled series with appropriate dtype
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

    def _impute_single_column(
        self, df: pd.DataFrame, target_column: str, params: Dict[str, Any]
    ) -> pd.Series:
        """Perform MICE imputation for a single column."""
        start_time = time()

        # Initialize encoders once
        categorical_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        label_encoder = LabelEncoder()  # For categorical target variables

        # Get predictor columns
        predictor_columns = [col for col in df.columns if col != target_column]

        # Separate numeric and categorical predictors
        numeric_predictors = [
            col
            for col in predictor_columns
            if self.determine_column_type(df[col]) == "numeric"
        ]
        categorical_predictors = [
            col for col in predictor_columns if col not in numeric_predictors
        ]

        # Initialize imputer
        imputer = self._get_imputer(df[target_column], params["method"])

        # Get missing mask
        missing_mask = df[target_column].isna()
        n_missing = missing_mask.sum()

        if n_missing == 0:
            return df[target_column]

        # Initialize imputations storage
        n_samples = len(df)
        is_numeric_target = self.determine_column_type(df[target_column]) == "numeric"
        imputations = (
            np.zeros((n_samples, params["n_impute"]))
            if is_numeric_target
            else np.empty((n_samples, params["n_impute"]), dtype=object)
        )

        # Prepare working copy
        df_working = df.copy(deep=False)  # Shallow copy

        # Initial fill of missing values in predictors
        for col in predictor_columns:
            if df_working[col].isna().any():
                if col in numeric_predictors:
                    series = pd.to_numeric(df_working[col], errors="coerce")
                    df_working[col] = series.fillna(series.mean())
                else:
                    df_working[col] = df_working[col].fillna(
                        df_working[col].mode().iloc[0]
                    )

        # Fit categorical encoder once
        if categorical_predictors:
            categorical_encoder.fit(df_working[categorical_predictors])

        # Prepare target encoder if needed
        if not is_numeric_target:
            label_encoder.fit(df[target_column].dropna())

        # Multiple imputation loop
        for m in range(params["n_impute"]):
            current_imp = df[target_column].copy()

            # Initialize missing values
            if is_numeric_target:
                current_series = pd.to_numeric(current_imp, errors="coerce")
                current_imp[missing_mask] = current_series.mean()
            else:
                current_imp[missing_mask] = current_imp.mode().iloc[0]

            # Iteration loop
            for iteration in range(params["n_iter"]):
                previous_imp = current_imp.copy()

                # Prepare predictor matrix
                X_parts = []
                if numeric_predictors:
                    X_numeric = df_working[numeric_predictors].astype(float).values
                    X_parts.append(X_numeric)
                if categorical_predictors:
                    X_categorical = categorical_encoder.transform(
                        df_working[categorical_predictors]
                    )
                    X_parts.append(X_categorical)

                X = np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]

                # Prepare target
                y = current_imp
                if not is_numeric_target:
                    y = label_encoder.transform(y)

                # Fit and predict
                observed_mask = ~missing_mask
                if isinstance(imputer, LogisticRegression):
                    imputer = self._safe_fit_logistic(
                        imputer, X[observed_mask], y[observed_mask]
                    )
                else:
                    imputer.fit(X[observed_mask], y[observed_mask])
                predicted = imputer.predict(X[missing_mask])
                if not is_numeric_target:
                    predicted = label_encoder.inverse_transform(predicted.astype(int))

                current_imp[missing_mask] = predicted
                df_working[target_column] = current_imp

                # Check convergence
                has_converged, conv_metric = self._check_convergence(
                    current_imp, previous_imp, missing_mask, iteration
                )

                if has_converged and iteration > 3:
                    break

            imputations[:, m] = current_imp
            gc.collect()

        # Combine imputations
        final_imputation = df[target_column].copy()
        if is_numeric_target:
            final_values = np.mean(imputations[missing_mask], axis=1)
        else:
            final_values = np.array(
                [
                    pd.Series(imputations[i, :]).mode().iloc[0]
                    for i in range(len(missing_mask))
                    if missing_mask[i]
                ]
            )

        final_imputation[missing_mask] = final_values

        if self.verbose:
            logger.info(
                f"Imputed {target_column} in {time() - start_time:.2f}s ({n_missing} missing values)"
            )
        gc.collect()
        return final_imputation

    def _get_imputer(self, series: pd.Series, method: str) -> Union[
        LinearRegression,
        LogisticRegression,
        BayesianRidge,
    ]:
        """
        Get the appropriate sklearn imputer based on data type and method.
        """
        is_numeric = self.determine_column_type(series) == "numeric"

        if is_numeric:
            imputers = {
                "linear": LinearRegression(),
                "bayesian_ridge": BayesianRidge(max_iter=300),
            }
        else:
            imputers = {
                "logistic": LogisticRegression(
                    max_iter=2000, solver="saga", tol=1e-3, n_jobs=1, warm_start=True
                ),
            }

        return imputers.get(method)

    def _safe_fit_logistic(self, imputer, X, y, max_attempts=3):
        """
        Safely fit logistic regression with multiple attempts and parameter adjustment.
        """

        original_max_iter = imputer.max_iter

        for attempt in range(max_attempts):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    imputer.fit(X, y)

                    # Check if there were convergence warnings
                    if not any(
                        issubclass(warn.category, ConvergenceWarning) for warn in w
                    ):
                        return imputer

                    # If we got a convergence warning, adjust parameters for next attempt
                    imputer.max_iter = int(imputer.max_iter * 1.5)
                    imputer.tol *= 1.2

            except Exception as e:
                logger.warning(f"Fitting attempt {attempt + 1} failed: {str(e)}")

                if attempt == max_attempts - 1:
                    # On last attempt, try with very relaxed parameters
                    imputer.max_iter = 5000
                    imputer.tol = 1e-2
                    imputer.fit(X, y)

        # Reset max_iter to original value
        imputer.max_iter = original_max_iter
        return imputer

    def _preprocess_datatypes(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Preprocess dataframe to optimize datatypes for imputation.
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

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Modified impute method to use the new parameter optimization approach."""
        self.tracker.start_column("total")
        logger.info("Starting MICE imputation")

        # Preprocess datatypes
        df_processed, column_types = self._preprocess_datatypes(df)
        original_dtypes = df.dtypes.to_dict()

        # Find columns to impute
        columns_with_na = [
            col for col in df_processed.columns if df_processed[col].isna().any()
        ]
        if not columns_with_na:
            logger.info("No missing values found")
            return df_processed

        # Create validation masks
        validation_masks = self._create_validation_masks(df_processed, columns_with_na)

        # Get optimal parameters for all columns
        optimal_params = self._get_optimal_params_for_all_columns(
            df_processed, columns_with_na, validation_masks
        )

        # Perform imputation with optimal parameters
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
                )
                for column in columns_with_na
            )

        # Update imputed values
        for column, imputed_values in zip(columns_with_na, results):
            df_processed[column] = imputed_values.astype(original_dtypes[column])

        self.tracker.finish_column("total")
        summary = self.tracker.get_summary()

        logger.info(
            f"Completed MICE imputation in {summary['total_time']:.2f}s\n"
            f"Average time per column: {summary['average_duration']:.2f}s"
        )

        return df_processed

    def _create_validation_masks(
        self, df: pd.DataFrame, columns: List[str]
    ) -> Dict[str, pd.Series]:
        """
        Create validation masks for parameter optimization.
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

    def _process_single_column(
        self,
        df: pd.DataFrame,
        column: str,
        validation_mask: pd.Series,
        original_dtype: np.dtype,
        optimal_params: Dict[str, Any],  # Added this parameter
        pbar: tqdm,
    ) -> pd.Series:
        """Process a single column for imputation using pre-computed optimal parameters."""
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

        # Perform imputation
        imputed_values = self._impute_single_column(df, column, optimal_params)

        # Restore original dtype
        if column_type == "numeric":
            imputed_values = pd.to_numeric(imputed_values, errors="coerce")
        imputed_values = imputed_values.astype(original_dtype)

        self.tracker.finish_column(column)
        pbar.update(1)
        return imputed_values


class PerformanceTracker:
    def __init__(self):
        self.timings = {}
        self.column_stats = {}
        self.total_start_time = time()

    def start_column(self, column: str):
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
        Update column statistics with a dictionary of values.

        Args:
            column: Column name
            stats: Dictionary of statistics to update
        """
        if column not in self.column_stats:
            self.column_stats[column] = {}
        self.column_stats[column].update(stats)

    def finish_column(self, column: str):
        if column in self.timings:
            end_time = time()
            self.timings[column]["end"] = end_time
            self.timings[column]["duration"] = end_time - self.timings[column].get(
                "start", end_time
            )

    def get_summary(self) -> Dict:
        """
        Get summary statistics with safe duration calculation.
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
