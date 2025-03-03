# KNN.py
from src.data_processing.imputation.base_imputer import BaseImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from time import time
from sklearn.metrics import mean_squared_error, accuracy_score
import sys

logger = logging.getLogger("wealth_index.imputer")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


class KNNImputer(BaseImputer):
    """K-Nearest Neighbors imputation implementation."""

    def _generate_all_column_parameter_combinations(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        validation_masks: Dict[str, pd.Series],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate all column and parameter combinations upfront."""
        all_combinations = []
        for column in columns_with_na:
            # Get column type and prepare data
            col_type = self.determine_column_type(df[column].copy())
            n_samples = len(df[column])
            missing_pct = (df[column].isna().sum() / n_samples) * 100

            # Generate parameter combinations
            params_list = self._generate_parameter_combinations(
                n_samples,
                col_type,
                missing_pct,
                column,
            )

            # Add each parameter combination with its column
            for params in params_list:
                all_combinations.append((column, params, col_type))

        return all_combinations

    def _validate_single_parameter_combination(
        self,
        df: pd.DataFrame,
        column: str,
        params: Dict[str, Any],
        col_type: str,
        validation_mask: pd.Series,
        country_year: str,
        pbar: tqdm,
    ) -> Tuple[str, Dict[str, Any], float]:
        """Validate a single parameter combination for a column."""
        # Handle None values in parameters that can't be None
        if params["metric"] != "minkowski":
            params = {k: v for k, v in params.items() if k != "p" or v is not None}

        # Evaluate the parameter combination
        score = self._evaluate_knn_parameters(
            params, df[column], validation_mask, df.copy(), column
        )
        logger.info(
            f"Evaluated {country_year}: {column} - Parameters: {params}, Score: {score:.4f}"
        )

        pbar.update(1)
        return column, params, score

    def _find_optimal_params_for_columns(
        self,
        df: pd.DataFrame,
        columns_with_na: List[str],
        validation_masks: Dict[str, pd.Series],
        country_year: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Find optimal parameters for all columns using parallel processing."""
        logger.info(
            f"Generating parameter combinations for all columns of {country_year}"
        )

        # Generate all combinations
        all_combinations = self._generate_all_column_parameter_combinations(
            df.copy(), columns_with_na, validation_masks
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
                delayed(self._validate_single_parameter_combination)(
                    df.copy(),
                    column,
                    params,
                    col_type,
                    validation_masks[column],
                    country_year,
                    pbar,
                )
                for column, params, col_type in all_combinations
            )

        # Group results by column
        column_results = {}
        for column, params, score in results:
            if column not in column_results:
                column_results[column] = []
            column_results[column].append((score, params))

        # Extract best parameters for each column
        optimal_params = {}
        column_scores = {}

        for column in columns_with_na:
            if column in column_results:
                # Sort by error
                result_list = sorted(
                    column_results[column], key=lambda x: x[0], reverse=False
                )
                best_score, best_params = result_list[0]
                column_scores[column] = best_score

                # Find all parameter sets that tied for best score
                tied_params = [
                    params for score, params in result_list if score == best_score
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
                    optimal_params[column] = min(secondary_results, key=lambda x: x[0])[
                        1
                    ]

                logger.info(
                    f"Validation for {column} - "
                    f"Parameters: {optimal_params[column]}, "
                    f"Normalized Error: {column_scores[column]:.4f}"
                )

        logger.info(f"Completed parameter optimization in {time() - start_time:.2f}s")
        return optimal_params, column_scores

    def _calculate_single_secondary_score(
        self, df: pd.DataFrame, column: str, params: Dict[str, Any], pbar: tqdm
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate secondary score for a single parameter set."""
        secondary_score = self._calculate_secondary_score(df, column, params)
        pbar.update(1)
        return secondary_score, params

    def _calculate_secondary_score(self, df, target_column, params):
        """
        Calculate secondary score for tie-breaking.
        Returns a score between 0 and 1 (lower is better).
        """
        # Complexity scores (lower is better)

        # 1. Neighbor count - normalize to 0-1 range
        # Prefer moderate neighbor counts (not too small, not too large)
        n_neighbors = params["n_neighbors"]
        if n_neighbors <= 5:
            n_neighbors_score = 0.3  # Small k is simple but can be noisy
        elif n_neighbors <= 15:
            n_neighbors_score = 0.1  # Moderate k is often optimal
        else:
            n_neighbors_score = (
                0.2 + (n_neighbors - 15) / 50
            )  # Larger k is smoother but can miss patterns

        # 2. Metric complexity
        metric_complexity = {
            "euclidean": 0.1,  # Fast, common
            "manhattan": 0.15,  # Fast, robust to outliers
            "minkowski": 0.3,  # More complex, parameter-dependent
            "chebyshev": 0.25,  # Less common
            "hamming": 0.1,  # Simple for categorical
        }
        metric_score = metric_complexity.get(params["metric"], 0.3)

        # 3. Algorithm choice
        algorithm_complexity = {
            "auto": 0.1,  # Let sklearn decide
            "kd_tree": 0.15,  # Fast for low dimensions
            "ball_tree": 0.2,  # Better for high dimensions
            "brute": 0.4,  # Expensive for large datasets
        }
        algorithm_score = algorithm_complexity.get(params["algorithm"], 0.3)

        # 4. Weighting scheme
        weights_score = 0.2 if params["weights"] == "distance" else 0.1

        # 5. Leaf size (relevant for tree-based algorithms)
        # Larger leaf size means less tree depth, generally faster
        leaf_size = params.get("leaf_size", 30)
        leaf_size_score = 0.1 if leaf_size >= 30 else 0.2

        # Combine all factors with appropriate weights
        return (
            0.35 * n_neighbors_score
            + 0.25 * metric_score
            + 0.2 * algorithm_score
            + 0.15 * weights_score
            + 0.05 * leaf_size_score
        )

    def _generate_parameter_combinations(
        self,
        n_samples: int,
        column_type: str,
        missing_pct: float,
        column_name: str,
    ) -> List[Dict]:
        """Generate parameter combinations for KNN imputation."""
        logger.info(
            f"Generating parameters for {column_name} ({column_type}, {missing_pct:.4f}% missing)"
        )

        # --- K Values Strategy ---
        center_k = int(np.sqrt(n_samples))
        center_k = max(3, min(center_k, 31))
        if center_k % 2 == 0:
            center_k += 1  # Ensure odd number

        k_spread_factor = (
            0.7 if missing_pct > 30 else (0.4 if missing_pct < 10 else 0.5)
        )
        k_spread = max(3, int(center_k * k_spread_factor))

        small_k = list(range(3, min(9, center_k), 2))
        medium_k = [
            max(3, center_k - k_spread),
            center_k,
            min(center_k + k_spread, n_samples // 3),
        ]
        medium_k = [k if k % 2 == 1 else k + 1 for k in medium_k]

        large_k = [min(int(center_k * 1.5), n_samples // 3)] if n_samples > 1000 else []
        if large_k and large_k[0] % 2 == 0:
            large_k[0] += 1

        k_values = sorted(set(small_k + medium_k + large_k))

        # --- Distance Metrics Strategy ---
        if column_type == "numeric":
            metrics = (
                ["euclidean", "manhattan", "minkowski", "chebyshev"]
                if missing_pct > 30
                else ["euclidean", "manhattan"]
            )
        else:
            metrics = ["hamming"] if n_samples > 5000 else ["hamming"]

        # --- Weights Strategy ---
        weights_options = ["uniform", "distance"]

        # --- Algorithm Strategy (Ensure Compatibility) ---
        if any(m in ["hamming"] for m in metrics):
            algorithm_options = ["auto", "ball_tree"]  # Categorical needs BallTree
        elif n_samples > 10000:
            algorithm_options = ["auto", "ball_tree", "kd_tree"]
        elif n_samples > 1000:
            algorithm_options = ["auto", "kd_tree"]
        else:
            algorithm_options = ["auto"]

        # --- Leaf Size Strategy ---
        if n_samples > 10000:
            leaf_size_options = [30, 50]
        elif n_samples > 1000:
            leaf_size_options = [20, 30]
        else:
            leaf_size_options = [10, 30]

        # --- P Parameter for Minkowski ---
        p_values = [1, 2] if "minkowski" in metrics else [None]

        # --- Generate Combinations ---
        combinations = []
        for k in k_values:
            for metric in metrics:
                for weights in weights_options:
                    for algorithm in algorithm_options:
                        if metric in ["hamming", "jaccard"] and algorithm == "kd_tree":
                            continue  # Skip invalid combinations

                        for leaf_size in leaf_size_options:
                            if metric == "minkowski":
                                for p in p_values:
                                    if p is not None:
                                        combinations.append(
                                            {
                                                "n_neighbors": k,
                                                "metric": metric,
                                                "weights": weights,
                                                "algorithm": algorithm,
                                                "leaf_size": leaf_size,
                                                "p": p,
                                                "n_jobs": 1,
                                            }
                                        )
                            else:
                                combinations.append(
                                    {
                                        "n_neighbors": k,
                                        "metric": metric,
                                        "weights": weights,
                                        "algorithm": algorithm,
                                        "leaf_size": leaf_size,
                                        "p": None,
                                        "n_jobs": 1,
                                    }
                                )

        # Limit the number of combinations
        target_combinations = 30 if missing_pct > 20 else 24
        if len(combinations) > target_combinations:
            combinations = self._select_key_combinations(
                combinations,
                target_combinations,
                center_k,
                metrics,
                weights_options,
                algorithm_options,
                column_type,
            )

        # Remove duplicates
        unique_combinations = []
        seen = set()
        for params in combinations:
            params_tuple = tuple(sorted((k, str(v)) for k, v in params.items()))
            if params_tuple not in seen:
                seen.add(params_tuple)
                unique_combinations.append(params)

        logger.info(
            f"Generated {len(unique_combinations)} parameter combinations for {column_name}"
        )
        return unique_combinations

    def _select_key_combinations(
        self,
        combinations: List[Dict],
        target_count: int,
        center_k: int,
        metrics: List[str],
        weights_options: List[str],
        algorithm_options: List[str],
        column_type: str,
    ) -> List[Dict]:
        """Select key parameter combinations strategically."""
        selected_combinations = []
        best_metric = "euclidean" if column_type == "numeric" else "hamming"

        # Include center k with different metrics
        for metric in metrics:
            selected_combinations.append(
                {
                    "n_neighbors": center_k,
                    "metric": metric,
                    "weights": "distance",
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "p": 2 if metric == "minkowski" else None,
                    "n_jobs": 1,
                }
            )

        # Get all k values from combinations
        k_values = sorted(set(params["n_neighbors"] for params in combinations))

        # Include different k values with best metric
        for k in k_values:
            if k != center_k:  # Already included above
                selected_combinations.append(
                    {
                        "n_neighbors": k,
                        "metric": best_metric,
                        "weights": "distance",
                        "algorithm": "auto",
                        "leaf_size": 30,
                        "p": None,
                        "n_jobs": 1,
                    }
                )

        # Include different weights
        for weight in weights_options:
            selected_combinations.append(
                {
                    "n_neighbors": center_k,
                    "metric": best_metric,
                    "weights": weight,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "p": None,
                    "n_jobs": 1,
                }
            )

        # Include algorithm variations
        for algo in algorithm_options:
            if algo != "auto":  # Already included above
                selected_combinations.append(
                    {
                        "n_neighbors": center_k,
                        "metric": best_metric,
                        "weights": "distance",
                        "algorithm": algo,
                        "leaf_size": 30,
                        "p": None,
                        "n_jobs": 1,
                    }
                )

        # Add remaining combinations if needed
        remaining_slots = target_count - len(selected_combinations)
        if remaining_slots > 0:
            # Get combinations not already selected
            remaining_combinations = [
                c for c in combinations if c not in selected_combinations
            ]
            # Sample from remaining combinations
            if len(remaining_combinations) > remaining_slots:
                np.random.shuffle(remaining_combinations)
                selected_combinations.extend(remaining_combinations[:remaining_slots])
            else:
                selected_combinations.extend(remaining_combinations)

        return selected_combinations

    def _evaluate_knn_parameters(
        self,
        params: Dict[str, Any],
        full_column: pd.Series,
        validation_mask: pd.Series,
        df: pd.DataFrame,
        column_name: str,
    ) -> float:
        """Evaluate KNN parameters using validation masks and direct error calculation."""
        col_type = self.determine_column_type(full_column)
        # Create a copy of the data for validation
        df_validation = df.copy()

        # Hide validation data
        df_validation.loc[validation_mask, column_name] = np.nan
        true_values = full_column[validation_mask]

        # Create and fit the model with the given parameters
        if col_type == "numeric":
            # Use KNeighborsRegressor for numeric columns
            knn_params = {
                k: v for k, v in params.items() if k != "metric" or v != "hamming"
            }
            if "p" in knn_params and knn_params["p"] is None:
                del knn_params["p"]
            knn = KNeighborsRegressor(**knn_params)
        else:
            knn = KNeighborsClassifier(**params)

        # Train on non-validation data
        X_for_validation = self._prepare_prediction_features(
            df_validation.copy(), column_name
        )
        observed_mask = ~df_validation[column_name].isna() & ~validation_mask
        knn.fit(X_for_validation[observed_mask], full_column[observed_mask])

        # Predict validation set
        predicted = knn.predict(X_for_validation[validation_mask])

        # Calculate error based on column type
        if col_type == "numeric":
            # For numeric columns, normalize RMSE by the range of the data
            data_range = full_column.max() - full_column.min()
            if data_range == 0:  # Handle constant columns
                data_range = 1.0

            rmse = np.sqrt(mean_squared_error(true_values, predicted))
            # Normalize to 0-1 scale
            error = min(rmse / data_range, 1.0)
            return error
        else:
            # For categorical columns, use error (lower is better)
            return 1 - accuracy_score(true_values, predicted)

    def _impute_column(
        self,
        df: pd.DataFrame,
        column: str,
        validation_mask: pd.Series,
        original_dtype: np.dtype,
        params: Dict[str, Any],
        pbar: tqdm = None,
    ) -> pd.Series:
        """Impute missing values for a single column using KNN with specified parameters."""
        self.tracker.start_column(column)
        start_time = time()
        logger.info(f"Imputing column {column}")

        # Get column type and missing percentage for logging
        column_type = self.determine_column_type(df[column])
        missing_pct = (df[column].isna().sum() / len(df)) * 100

        self.tracker.update_column(
            column,
            {
                "type": column_type,
                "missing_percentage": missing_pct,
                "optimal_params": params,
            },
        )

        # Prepare data
        y = df[column].copy()
        if column_type == "numeric":
            y = pd.to_numeric(y, errors="raise").astype("float64")
        else:
            mask = y.isna()
            y = y.astype(str)
            y[mask] = np.nan

        y_train = y[y.notna()]
        X = self._prepare_prediction_features(df, column)
        X_train = X[y.notna()]

        # Use appropriate model based on column type
        if column_type == "numeric":
            # Use KNeighborsRegressor for numeric columns
            knn_params = {
                k: v for k, v in params.items() if k != "metric" or v != "hamming"
            }
            if "p" in knn_params and knn_params["p"] is None:
                del knn_params["p"]
            knn = KNeighborsRegressor(**knn_params)
        else:
            knn = KNeighborsClassifier(**params)

        # Fit model and predict missing values
        knn.fit(X_train, y_train)
        null_mask = y.isna()

        if null_mask.any():
            imputed_values = knn.predict(X[null_mask])
            y[null_mask] = imputed_values

        logger.info(f"Finished imputing {column} after {time() - start_time:.2f}s")

        if pbar:
            pbar.update(1)

        # Convert to appropriate type
        if column_type == "numeric":
            result = pd.to_numeric(y, errors="raise").astype("float64")
        else:
            result = y.astype("string")

        self.tracker.finish_column(column)
        return result

    def impute(self, df: pd.DataFrame, country_year: str) -> Tuple[pd.DataFrame, float]:
        """Impute missing values using KNN while preserving original data types."""
        logger.info(f"Imputing missing values for dataset: {country_year}")
        self.tracker.start_column("total")
        numerical_columns = []
        for col in df.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 1 and df[col].isna().any():
                logger.info(
                    f"Column {col} has only one unique value: {unique_values[0]} - using constant imputation"
                )
                # Simply fill missing values with the single value
                df[col] = df[col].fillna(unique_values[0])
        for col in df.columns:
            col_type = self.determine_column_type(df[col])
            if col_type == "numeric":
                df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
                numerical_columns.append(col)
            else:
                df[col] = df[col].astype("category")

        # Store original types and prepare data
        original_dtypes = df.dtypes.to_dict()
        df_imputed, df_ids = self._prepare_data(df)

        # Find columns to impute
        columns_with_na = [col for col in df.columns if df[col].isna().any()]
        if not columns_with_na:
            logger.info("No missing values found")
            result_df = df_imputed.copy()
            if not df_ids.empty:
                result_df = pd.concat([result_df, df_ids], axis=1)
            return result_df, 0.0

        # Create validation masks and calculate missing value weights
        validation_masks = self._create_validation_masks(df_imputed, columns_with_na)
        missing_values = df.isna().sum()

        # Find optimal parameters and validate quality
        logger.info("Finding optimal parameters and validating imputation quality")
        optimal_params, column_scores = self._find_optimal_params_for_columns(
            df_imputed, columns_with_na, validation_masks, country_year
        )

        # Calculate weighted score
        total_score = self._calculate_weighted_score(column_scores, missing_values)

        # Perform imputation using the optimal parameters
        logger.info(f"Imputing {len(columns_with_na)} columns")
        with tqdm(total=len(columns_with_na), desc="Imputing columns") as pbar:
            results = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads"
            )(
                delayed(self._impute_column)(
                    df_imputed.copy(),
                    column,
                    validation_masks[column],
                    original_dtypes.get(column),
                    optimal_params[column],
                    pbar,
                )
                for column in columns_with_na
            )

        # Update imputed values in the dataframe
        for column, imputed_values in zip(columns_with_na, results):
            df_imputed[column] = imputed_values

        # Combine with IDs and restore original types
        if not df_ids.empty:
            df_imputed = pd.concat([df_imputed, df_ids], axis=1)

        final_df = self._restore_column_types(
            df_imputed, original_dtypes, numerical_columns
        )

        self.tracker.finish_column("total")
        summary = self.tracker.get_summary()
        logger.info(
            f"Completed KNN imputation in {summary['total_time']:.2f}s\n"
            f"Average time per column: {summary['average_duration']:.2f}s"
        )

        return final_df, total_score
