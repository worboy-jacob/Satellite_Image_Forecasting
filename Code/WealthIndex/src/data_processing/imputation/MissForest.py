"""
Random Forest based imputation (MissForest) implementation.

Provides an implementation of the MissForest algorithm for handling missing data
through iterative Random Forest models. Each variable with missing values is modeled
using Random Forest regression (for numeric variables) or classification (for categorical
variables) based on observed values in other variables.
"""

from src.data_processing.imputation.base_imputer import BaseImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from sklearn.preprocessing import LabelEncoder
import logging
from time import time
import sys

logger = logging.getLogger("wealth_index.imputer")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


class MissForestImputer(BaseImputer):
    """
    Random Forest based imputation implementation.

    Implements the MissForest algorithm which handles missing data by training
    Random Forest models on observed values to predict missing values. The process
    iterates until convergence, using regression forests for numeric variables
    and classification forests for categorical variables.
    """

    def _generate_parameter_combinations(
        self, n_samples: int, column_type: str, missing_pct: float, column_name: str
    ) -> List[Dict]:
        """
        Generate parameter combinations for MissForest imputation.

        Creates parameter sets based on data characteristics such as sample size,
        column type, and missing percentage, focusing on key Random Forest parameters
        like number of trees, tree depth, and feature selection methods.

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

        # Scale iterations with missing data complexity
        if missing_pct < 10:
            n_iterations_values = [3, 5]
        elif missing_pct < 30:
            n_iterations_values = [3, 5, 7]
        else:
            n_iterations_values = [5, 7, 10]

        # Adjust tree count based on dataset size
        if n_samples < 1000:
            n_estimators_values = [50, 100]
        else:
            n_estimators_values = [100, 200]

        # Feature selection strategies - one of the most impactful parameters
        max_features_values = ["sqrt", "log2", None]  # None means all features

        # Determine max_depth - another impactful parameter
        # Use heuristic: deeper trees for smaller datasets with less missing data
        if n_samples < 1000 and missing_pct < 20:
            max_depth_values = [None]  # Unlimited depth for small datasets
        elif n_samples < 5000:
            max_depth_values = [20, None]
        else:
            max_depth_values = [30, None]  # Limit depth for large datasets

        # Generate combinations focusing on the most impactful parameters
        combinations = []
        for n_iterations in n_iterations_values:
            for n_estimators in n_estimators_values:
                for max_features in max_features_values:
                    for max_depth in max_depth_values:
                        # Fixed values for less impactful parameters
                        min_samples_split = 5 if n_samples > 5000 else 2
                        bootstrap = True  # Enable for OOB score

                        params = {
                            "n_iterations": n_iterations,
                            "n_estimators": n_estimators,
                            "max_features": max_features,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "bootstrap": bootstrap,
                            "warm_start": False,
                            "oob_score": True,
                        }
                        combinations.append(params)

        logger.info(
            f"Generated {len(combinations)} parameter combinations for {column_name}"
        )
        return combinations

    def _calculate_secondary_score(
        self,
        df: pd.DataFrame,
        target_column: str,
        params: Dict[str, Any],
        rf_model: Union[RandomForestClassifier, RandomForestRegressor],
    ) -> float:
        """
        Calculate secondary score for tie-breaking parameter sets.

        Evaluates parameter configurations based on model confidence and
        complexity when multiple configurations have similar imputation quality.
        Lower scores are better, with values between 0 and 1.

        Args:
            df: DataFrame containing the data
            target_column: Column being imputed
            params: Parameter configuration to evaluate
            rf_model: The initialized Random Forest model

        Returns:
            Float score between 0 and 1 (lower is better)
        """
        # Calculate prediction confidence if available
        confidence_score = 0.5  # Default neutral score

        # Get non-missing data for confidence calculation
        non_missing_mask = ~df[target_column].isna()
        try:
            if non_missing_mask.sum() > 0:
                X_observed = self._prepare_prediction_features(
                    df.loc[non_missing_mask], target_column
                )
                y_observed = df.loc[non_missing_mask, target_column]

                # Determine if we're dealing with classification or regression
                is_classifier = hasattr(rf_model, "predict_proba")

                # Sample data for efficiency while preserving distribution
                sample_size = min(1000, len(X_observed))
                if sample_size < len(X_observed):
                    if is_classifier:
                        # For classification, use stratified sampling if possible
                        try:
                            from sklearn.model_selection import StratifiedShuffleSplit

                            splitter = StratifiedShuffleSplit(
                                n_splits=1, train_size=sample_size
                            )
                            for train_idx, _ in splitter.split(X_observed, y_observed):
                                X_sample = X_observed[train_idx]
                                y_sample = y_observed.iloc[train_idx]
                        except Exception:
                            # Fall back to random sampling if stratification fails
                            idx = np.random.RandomState().choice(
                                len(X_observed), sample_size, replace=False
                            )
                            X_sample = X_observed[idx]
                            y_sample = y_observed.iloc[idx]
                    else:
                        # For regression, use random sampling
                        idx = np.random.RandomState().choice(
                            len(X_observed), sample_size, replace=False
                        )
                        X_sample = X_observed[idx]
                        y_sample = y_observed.iloc[idx]
                else:
                    X_sample = X_observed
                    y_sample = y_observed

                # Fit model and calculate confidence
                rf_model.fit(X_sample, y_sample)

                # Calculate model quality score based on available metrics
                if is_classifier:
                    # For classification, use prediction probabilities
                    probas = rf_model.predict_proba(X_sample)
                    confidence_score = 1 - np.mean(
                        np.max(probas, axis=1)
                    )  # Lower is better
                elif hasattr(rf_model, "oob_score_") and rf_model.oob_score_:
                    # For regression with OOB, use OOB score
                    confidence_score = 1 - rf_model.oob_score_  # Lower is better
                else:
                    # For regression without OOB, use R² score
                    from sklearn.metrics import r2_score

                    y_pred = rf_model.predict(X_sample)
                    r2 = r2_score(y_sample, y_pred)
                    confidence_score = max(
                        0, 1 - r2
                    )  # Convert R² to error (lower is better)

        except Exception as e:
            logger.warning(f"Error calculating secondary score: {str(e)}")

        # Model complexity factors (all normalized to 0-1 range, lower is better)

        # 1. Number of estimators (trees)
        n_estimators_score = params["n_estimators"] / 200.0  # Normalize by max value

        # 2. Number of iterations
        n_iterations_score = params["n_iterations"] / 10.0  # Normalize by max value

        # 3. Tree depth complexity
        if params["max_depth"] is None:
            depth_score = 0.5  # Unlimited depth is moderately complex
        else:
            depth_score = min(params["max_depth"] / 50.0, 1.0)

        # 4. Feature selection complexity
        feature_complexity = {
            "sqrt": 0.2,  # Less complex - fewer features
            "log2": 0.3,  # Moderately complex
            None: 0.5,  # Most complex - uses all features
        }
        feature_score = feature_complexity.get(params["max_features"], 0.4)

        # 5. Min samples split (simpler values are larger)
        min_samples_score = 0.5 / max(params["min_samples_split"], 1)

        # Combined complexity score with weighted importance
        complexity_score = (
            0.4 * n_estimators_score  # Most impactful on complexity
            + 0.25 * n_iterations_score  # Second most impactful
            + 0.15 * depth_score  # Third most impactful
            + 0.15 * feature_score  # Also important
            + 0.05 * min_samples_score  # Least impactful
        )

        # Final weighted score (emphasize confidence over complexity)
        # 0.7 for confidence (model quality)
        # 0.3 for complexity (simplicity preference)
        return 0.7 * confidence_score + 0.3 * complexity_score

    def _get_forest_model(
        self, is_numeric: bool, params: Dict[str, Any]
    ) -> Union[RandomForestRegressor, RandomForestClassifier]:
        """
        Get the appropriate Random Forest model based on column type.

        Creates and configures either a RandomForestRegressor for numeric columns
        or a RandomForestClassifier for categorical columns, applying the
        specified parameters.

        Args:
            is_numeric: Whether the column contains numeric data
            params: Parameters for configuring the forest model

        Returns:
            Configured RandomForest model ready for fitting
        """
        forest_params = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "max_features": params["max_features"],
            "bootstrap": params["bootstrap"],
            "warm_start": params["warm_start"],
            "n_jobs": 1,  # Use 1 job per model as we parallelize at a higher level
        }

        # Only add oob_score if bootstrap is True
        if params["bootstrap"]:
            forest_params["oob_score"] = True

        if is_numeric:
            return RandomForestRegressor(**forest_params)
        else:
            return RandomForestClassifier(**forest_params)

    def _get_imputer(
        self, series: pd.Series, params
    ) -> Union[RandomForestRegressor, RandomForestClassifier]:
        """
        Get the appropriate Random Forest imputer for a series.

        Determines the column type and delegates to _get_forest_model to create
        the appropriate model with the given parameters.

        Args:
            series: The series to be imputed
            params: Parameters for the Random Forest model

        Returns:
            Configured RandomForest model for the given data type
        """
        is_numeric_target = self.determine_column_type(series) == "numeric"
        return self._get_forest_model(is_numeric_target, params)

    def _impute_single_column(
        self,
        df: pd.DataFrame,
        target_column: str,
        params: Dict[str, Any],
        country_year: str,
    ) -> pd.Series:
        """
        Perform MissForest imputation for a single column.

        Implements the iterative Random Forest imputation process for one column,
        using other columns as predictors and iterating until convergence or
        maximum iterations is reached.

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

        # Get missing mask
        missing_mask = df[target_column].isna()
        n_missing = missing_mask.sum()

        if n_missing == 0:
            return df[target_column]

        # Determine if target is numeric
        is_numeric_target = self.determine_column_type(df[target_column]) == "numeric"

        # Initialize imputer
        rf_model = self._get_forest_model(is_numeric_target, params)

        # Prepare working copy
        df_working = df.copy(deep=False)  # Shallow copy

        # Initial fill of missing values in predictors
        for col in predictor_columns:
            if df_working[col].isna().any():
                if col in numeric_predictors:
                    series = pd.to_numeric(df_working[col], errors="raise")
                    df_working[col] = series.fillna(series.mean())
                else:
                    df_working[col] = df_working[col].fillna(
                        df_working[col].mode().iloc[0]
                    )

        # Prepare target encoder if needed
        # Only fit the label encoder once on the non-missing values
        if not is_numeric_target:
            non_missing_values = df[target_column].dropna()
            if not non_missing_values.empty:
                label_encoder.fit(non_missing_values)
                # Store if we have a valid encoder
                has_encoder = True
            else:
                has_encoder = False

        # Initialize missing values with statistical estimates
        current_imp = df[target_column].copy()
        if is_numeric_target:
            current_series = pd.to_numeric(current_imp, errors="raise")
            current_imp[missing_mask] = current_series.mean()
        else:
            mode_value = (
                current_imp.mode().iloc[0]
                if not current_imp.dropna().empty
                else "missing"
            )
            current_imp[missing_mask] = mode_value

        df_working[target_column] = current_imp

        # Iterative imputation process
        for iteration in range(params["n_iterations"]):
            previous_imp = current_imp.copy()

            # Prepare features for prediction
            X = self._prepare_prediction_features(df_working, target_column)

            # Prepare target values for model fitting
            observed_mask = ~missing_mask

            # For categorical target, transform to numeric labels if we have an encoder
            if not is_numeric_target and has_encoder:
                # Keep current_imp as Series, but create a transformed version for the model
                y_values = current_imp.values
                # Transform only for model training
                y_transformed = label_encoder.transform(y_values)
                try:
                    # Use transformed values for fitting
                    rf_model.fit(X[observed_mask], y_transformed[observed_mask])
                    # Predict with the model
                    predicted_labels = rf_model.predict(X[missing_mask])
                    # Convert back to original representation
                    predicted_values = label_encoder.inverse_transform(
                        predicted_labels.astype(int)
                    )
                    # Update the Series with predictions
                    current_imp[missing_mask] = predicted_values
                except Exception as e:
                    logger.warning(f"Error during fitting: {str(e)}")
                    # If fitting fails, keep previous imputation
                    current_imp = previous_imp
                    break
            else:
                # For numeric targets, fit directly
                try:
                    rf_model.fit(X[observed_mask], current_imp.values[observed_mask])
                    predicted = rf_model.predict(X[missing_mask])
                    current_imp[missing_mask] = predicted
                except Exception as e:
                    logger.warning(f"Error during fitting: {str(e)}")
                    current_imp = previous_imp
                    break

            # Update working dataframe
            df_working[target_column] = current_imp

            # Check convergence to determine if we can stop early
            has_converged, conv_metric = self._check_convergence(
                current_imp, previous_imp, missing_mask, iteration
            )

            if has_converged and iteration >= 3:
                logger.info(
                    f"Converged after {iteration+1} iterations with metric {conv_metric:.6f}"
                )
                break

        # Return final imputation
        final_imputation = df[target_column].copy()
        final_imputation[missing_mask] = current_imp[missing_mask]

        if self.verbose:
            logger.info(
                f"Imputed {country_year}: {target_column} in {time() - start_time:.2f}s ({n_missing} missing values)"
            )
        gc.collect()
        return final_imputation

    def impute(self, df: pd.DataFrame, country_year: str) -> Tuple[pd.DataFrame, float]:
        """
        Perform MissForest imputation on the entire dataset.

        Orchestrates the complete MissForest imputation workflow for all columns
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
        logger.info(f"Imputing missing values with MissForest for {country_year}")
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
            imputation_name="MissForest",
            generate_params=self._generate_parameter_combinations,
            impute_single_column=self._impute_single_column,
            calc_secondary_scores=self._calculate_secondary_score,
            country_year=country_year,
            get_imputer=self._get_imputer,
        )
