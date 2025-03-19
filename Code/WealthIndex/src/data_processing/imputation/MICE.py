"""
Multiple Imputation by Chained Equations (MICE) implementation.

Provides an implementation of the MICE algorithm for handling missing data
through an iterative series of predictive models. Each variable with missing
values is modeled as a function of other variables, with appropriate models
selected based on variable types (linear regression for numeric variables,
logistic regression for categorical variables).
"""

from src.data_processing.imputation.base_imputer import BaseImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List
import gc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings
import logging
from time import time
import sys

logger = logging.getLogger("wealth_index.imputer")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


class MICEImputer(BaseImputer):
    """
    Multiple Imputation by Chained Equations implementation.

    Implements the MICE algorithm which handles missing data by modeling
    each variable with missing values conditionally on other variables in
    the data, creating multiple imputations to account for the statistical
    uncertainty in the imputations.
    """

    def _generate_parameter_combinations(
        self, n_samples: int, column_type: str, missing_pct: float, column_name: str
    ) -> List[Dict]:
        """
        Generate parameter combinations for MICE imputation.

        Creates appropriate parameter sets based on data characteristics
        such as sample size, column type, and missing percentage, focusing
        on likely optimal configurations.

        Args:
            n_samples: Number of samples in the dataset
            column_type: Type of the column ('numeric' or 'categorical')
            missing_pct: Percentage of missing values
            column_name: Name of the column for logging

        Returns:
            List of parameter dictionaries for evaluation
        """
        logger.info(
            f"Generating parameters for {column_name} ({column_type}, {missing_pct:.4f}% missing)"
        )

        # Adjust imputation count based on missing data severity
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
            iter_values = [5, 10, 15, 20] if missing_pct > 30 else [5, 10, 15]
        else:
            iter_values = [5, 10, 15]  # For larger datasets, fewer iterations

        # Select appropriate methods based on column type and data characteristics
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

    def _calculate_secondary_score(self, df, target_column, params, imputer):
        """
        Calculate secondary score for tie-breaking between parameter sets.

        Evaluates parameter configurations based on model confidence and
        complexity when multiple configurations have similar imputation quality.
        Lower scores are better, with values between 0 and 1.

        Args:
            df: DataFrame containing the data
            target_column: Column being imputed
            params: Parameter configuration to evaluate
            imputer: The initialized imputation model

        Returns:
            Float score between 0 and 1 (lower is better)
        """
        # 1. Calculate prediction confidence
        confidence_score = 0.5  # Default neutral score
        non_missing_mask = ~df[target_column].isna()

        # Evaluate prediction confidence on observed data
        if non_missing_mask.sum() > 0:
            try:
                X_observed = self._prepare_prediction_features(
                    df.loc[non_missing_mask], target_column
                )
                y_observed = df.loc[non_missing_mask, target_column]

                if hasattr(imputer, "predict_proba"):
                    # For classification models
                    imputer.fit(X_observed, y_observed)
                    probas = imputer.predict_proba(X_observed)
                    confidence_score = 1 - np.mean(np.max(probas, axis=1))
                elif isinstance(imputer, BayesianRidge):
                    # For Bayesian regression
                    imputer.fit(X_observed, y_observed)
                    y_pred, y_std = imputer.predict(X_observed, return_std=True)
                    # Normalize by data standard deviation
                    data_std = np.std(y_observed)
                    if data_std > 0:
                        confidence_score = min(np.mean(y_std) / data_std, 1.0)
                elif hasattr(imputer, "score"):
                    # For other regression models
                    imputer.fit(X_observed, y_observed)
                    r2_score = imputer.score(X_observed, y_observed)
                    confidence_score = max(
                        0, 1 - r2_score
                    )  # Convert R² to error (lower is better)
            except Exception as e:
                logger.debug(f"Error in confidence calculation: {str(e)}")

        # 2. Method-specific complexity
        method_complexity = {
            "linear": 0.1,
            "bayesian_ridge": 0.2,
            "logistic": 0.3,
        }
        method_score = method_complexity.get(params["method"], 0.4)

        # 3. Parameter complexity
        n_iter_score = params["n_iter"] / 20.0
        n_impute_score = params["n_impute"] / 11.0

        # 4. Additional parameter complexity
        additional_complexity = 0.0
        if params.get("warm_start", False):
            additional_complexity += 0.05
        if params.get("scale_features", False):
            additional_complexity += 0.05
        if params.get("class_weight", None) == "balanced":
            additional_complexity += 0.05
        if params.get("multi_class", "auto") == "multinomial":
            additional_complexity += 0.1

        # 5. Combined complexity score
        complexity_score = (
            0.4 * n_iter_score
            + 0.3 * n_impute_score
            + 0.2 * method_score
            + 0.1 * additional_complexity
        )

        # Final weighted score (prioritize confidence over complexity)
        return 0.7 * confidence_score + 0.3 * complexity_score

    def _get_imputer(
        self, series: pd.Series, params: Dict[str, Any]
    ) -> Union[LinearRegression, LogisticRegression, BayesianRidge]:
        """
        Get the appropriate sklearn imputer based on data type and method.

        Selects and configures the proper scikit-learn model based on whether
        the column is numeric or categorical and the specified method.

        Args:
            series: The series to be imputed
            params: Parameters specifying the imputation method

        Returns:
            Configured scikit-learn model ready for fitting
        """

        method = params["method"]
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

    def _safe_fit_logistic(
        self,
        imputer: LogisticRegression,
        X: np.ndarray,
        y: np.ndarray,
        max_attempts: int = 3,
    ) -> LogisticRegression:
        """
        Safely fit logistic regression with multiple attempts and parameter adjustment.

        Handles convergence issues by automatically adjusting parameters and
        making multiple fitting attempts, increasing max_iter and relaxing
        tolerance if needed.

        Args:
            imputer: LogisticRegression model to fit
            X: Feature matrix
            y: Target values
            max_attempts: Maximum number of fitting attempts

        Returns:
            Fitted LogisticRegression model
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

    def _impute_single_column(
        self,
        df: pd.DataFrame,
        target_column: str,
        params: Dict[str, Any],
        country_year: str,
    ) -> pd.Series:
        """
        Perform MICE imputation for a single column.

        Implements the core MICE algorithm for one column, creating multiple
        imputations through an iterative process and combining them into a
        final imputation.

        Args:
            df: DataFrame containing the data
            target_column: Column to impute
            params: Parameters for the imputation process
            country_year: Country-year identifier for logging

        Returns:
            Series with imputed values
        """
        start_time = time()

        # Initialize encoders
        categorical_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        label_encoder = LabelEncoder()  # For categorical target variables

        # Get predictor columns and separate by type
        predictor_columns = [col for col in df.columns if col != target_column]
        numeric_predictors = [
            col
            for col in predictor_columns
            if self.determine_column_type(df[col]) == "numeric"
        ]
        categorical_predictors = [
            col for col in predictor_columns if col not in numeric_predictors
        ]

        # Initialize imputer and get missing mask
        imputer = self._get_imputer(df[target_column], params)
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

        # Prepare working copy and initial fill of missing predictors
        df_working = df.copy(deep=False)  # Shallow copy
        for col in predictor_columns:
            if df_working[col].isna().any():
                if col in numeric_predictors:
                    series = pd.to_numeric(df_working[col], errors="raise")
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

        # Multiple imputation loop - create n_impute different imputed datasets
        for m in range(params["n_impute"]):
            current_imp = df[target_column].copy()

            # Initialize missing values
            if is_numeric_target:
                current_series = pd.to_numeric(current_imp, errors="raise")
                current_imp[missing_mask] = current_series.mean()
            else:
                current_imp[missing_mask] = current_imp.mode().iloc[0]

            # Iteration loop - refine each imputation through multiple iterations
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

        # Combine multiple imputations into final values
        # For numeric data: use mean of imputations
        # For categorical data: use mode (most common value)
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
                f"Imputed {country_year}: {target_column} in {time() - start_time:.2f}s ({n_missing} missing values)"
            )
        gc.collect()
        return final_imputation

    def impute(self, df: pd.DataFrame, country_year: str) -> Tuple[pd.DataFrame, float]:
        """
        Perform MICE imputation on the entire dataset.

        Orchestrates the complete MICE imputation workflow for all columns
        with missing values, handling special cases and delegating to the
        common imputation workflow.

        Args:
            df: DataFrame with missing values to impute
            country_year: String identifier for the country and year

        Returns:
            Tuple containing:
            - DataFrame with imputed values
            - Overall imputation quality score
        """
        logger.info(f"Imputing missing values using MICE for {country_year}")
        for col in df.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 1 and df[col].isna().any():
                logger.info(
                    f"Column {col} has only one unique value: {unique_values[0]} - using constant imputation"
                )
                # Simply fill missing values with the single
                df[col] = df[col].fillna(unique_values[0])
        return self._common_imputation_workflow(
            df=df,
            imputation_name="MICE",
            generate_params=self._generate_parameter_combinations,
            impute_single_column=self._impute_single_column,
            get_imputer=self._get_imputer,
            calc_secondary_scores=self._calculate_secondary_score,
            country_year=country_year,
        )
