# wealth_index/data_processing/imputer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import logging
from typing import Dict, Any, Tuple, List
from scipy.stats import randint, uniform
from tqdm import tqdm

###TODO: Change this to implement MissForest and MICE, then to go through all 3 and pick the best one
###TODO: make sure to add hyperparameter optimization above

logger = logging.getLogger("wealth_index.imputer")


class KNNImputer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = config.get("verbose", 0)

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
            df_imputed[column] = self._impute_column(df_imputed, column)
            logger.info(f"Successfully imputed column: {column}")

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
            col_type = self.determine_column_type(features[col])
            if col_type == "numeric":
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

    def _get_parameter_space(
        self, X_train: np.ndarray, y_train: pd.Series, col_type: str
    ) -> Dict[str, Any]:
        """
        Define parameter space for randomized search based on data characteristics.
        """
        n_samples = len(y_train)
        # Calculate k range
        min_k = 3
        max_k = min(int(np.sqrt(n_samples)), 15, n_samples // 4)

        if n_samples < 100:
            leaf_range = (10, 20)
        else:
            leaf_range = (20, 30)
        # Base parameter space
        param_space = {
            "n_neighbors": randint(min_k, max_k + 1),
            "weights": ["uniform", "distance"],
            "leaf_size": randint(*leaf_range),
        }

        # Metric space based on column type
        if col_type == "numeric":
            param_space.update(
                {
                    "metric": ["minkowski", "euclidean", "manhattan", "cosine"],
                    "p": randint(1, 3),  # 1 for manhattan, 2 for euclidean
                }
            )
        else:
            param_space.update(
                {"metric": ["hamming"], "p": [2]}  # Fixed for categorical data
            )

        return param_space

    def _simple_parameter_search(
        self, X_train: np.ndarray, y_train: pd.Series, param_space: Dict
    ) -> Dict[str, Any]:
        """
        Simple parameter search for cases with severe class imbalance.
        """
        from sklearn.model_selection import train_test_split

        # Split data with stratification if possible
        try:
            X_t, X_v, y_t, y_v = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                stratify=y_train if len(np.unique(y_train)) > 1 else None,
            )
        except ValueError:
            # If stratification fails, do regular split
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2)

        best_score = float("-inf")
        best_params = None
        no_improvement_count = 0
        max_no_improvement = 5
        n_iter = max(5, min(10, len(y_train) // 10)) * 2
        pbar = tqdm(
            range(n_iter),
            desc="Parameter search",
            unit="trial",
            ncols=80,  # Fixed width
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        # Number of random combinations to try

        for _ in pbar:
            # Sample random parameters
            params = {
                "n_neighbors": np.random.randint(3, 21),
                "weights": np.random.choice(["uniform", "distance"]),
                "leaf_size": np.random.randint(10, 50),
                "metric": param_space["metric"][0],  # Use first metric
                "p": param_space["p"][0],  # Use first p value
            }

            # Ensure n_neighbors is odd
            if params["n_neighbors"] % 2 == 0:
                params["n_neighbors"] += 1

            try:
                knn = KNeighborsClassifier(**params, algorithm="auto")
                knn.fit(X_t, y_t)
                score = knn.score(X_v, y_v)

                if score > best_score:
                    best_score = score
                    best_params = params
                    no_improvement_count = 0
                    pbar.set_description(f"Best score: {best_score:.4f}")
                else:
                    no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    logger.info("Early stopping triggered - no significant improvement")
                    break

            except Exception as e:
                logger.warning(f"Trial failed with parameters {params}: {e}")
                continue

        logger.info(f"Simple search completed. Best score: {best_score:.4f}")
        return best_params or param_space  # Return best params or defaults

    def _find_optimal_parameters(
        self, X_train: np.ndarray, y_train: pd.Series, col_type: str
    ) -> Dict[str, Any]:
        """
        Find optimal parameters using randomized search with early stopping.
        """
        param_space = self._get_parameter_space(X_train, y_train, col_type)

        # Calculate number of iterations based on dataset size
        n_iter = max(5, min(10, len(y_train) // 10))
        value_counts = pd.Series(y_train).value_counts()

        cv = 2
        if value_counts.min() < 2:
            logger.info("Using simple parameter search due to class imbalance")
            return self._simple_parameter_search(X_train, y_train, param_space)
        else:
            logger.info(f"Using {cv}-fold cross-validation based on class distribution")

            # Create base classifier
            logger.info(f"Starting randomized search with {n_iter} iterations")
            logger.info(f"Parameter space: {param_space}")

            # Initialize randomized search
            random_search = RandomizedSearchCV(
                estimator=KNeighborsClassifier(algorithm="auto"),
                param_distributions=param_space,
                n_iter=n_iter,
                cv=cv,
                scoring="accuracy",
                error_score="raise",
                n_jobs=-1,
                verbose=self.verbose,
            )

            try:
                # Fit randomized search
                random_search.fit(X_train, y_train)

                # Get best parameters
                best_params = random_search.best_params_

                # Ensure n_neighbors is odd
                if best_params["n_neighbors"] % 2 == 0:
                    best_params["n_neighbors"] += 1

                logger.info(
                    f"Best parameters found: {best_params} "
                    f"with score={random_search.best_score_:.4f}"
                )

                return best_params

            except Exception as e:
                logger.error(f"Parameter optimization failed: {e}")
                # Return default parameters if optimization fails
                return {
                    "n_neighbors": 3,
                    "weights": "uniform",
                    "metric": "hamming" if col_type == "categorical" else "minkowski",
                    "p": 2,
                    "leaf_size": 30,
                    "algorithm": "auto",
                }

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
        optimal_params = self._find_optimal_parameters(X_train, y_train, col_type)

        knn = KNeighborsClassifier(**optimal_params)

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
