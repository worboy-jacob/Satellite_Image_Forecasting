"""
Factor Analysis of Mixed Data (FAMD) implementation for wealth index calculation.

This module provides tools to calculate wealth indices from household survey data
using FAMD, with support for parallel processing, component significance testing,
and visualization of results.
"""

import pandas as pd
import numpy as np
from prince import FAMD
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.paths import get_results_dir
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
import sys
import multiprocessing
from time import time
import gc
import psutil
from scipy import sparse

logger = logging.getLogger("wealth_index.famd")
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


class FAMDAnalyzer:
    """Factor Analysis of Mixed Data implementation."""

    def __init__(self, config: dict):
        self.config = config
        self.n_jobs = config.get("n_jobs", mp.cpu_count() - 1)
        self.verbose = config.get("verbose", 0)
        setup_logging(config.get("log_level", "INFO"))

    def _get_effective_n_jobs(self) -> int:
        """
        Convert n_jobs parameter to actual number of jobs.

        Handles negative values according to joblib convention, where
        -1 means using all processors, -2 means using all but one, etc.

        Returns:
            Number of jobs to use for parallel processing
        """
        if self.n_jobs < 0:
            # Convert negative n_jobs to positive
            effective_n_jobs = max(mp.cpu_count() + 1 + self.n_jobs, 1)
        else:
            effective_n_jobs = self.n_jobs
        return effective_n_jobs

    def _run_parallel_analysis(
        self, famd, df: pd.DataFrame, n_simulations: int
    ) -> list:
        """
        Run parallel analysis to determine statistically significant components.

        Performs Monte Carlo simulations to identify components with eigenvalues
        greater than those from randomly generated data.

        Args:
            famd: Fitted FAMD model
            df: Input data frame
            n_simulations: Number of simulations to perform

        Returns:
            List of indices for significant components
        """
        n_samples, n_features = df.shape
        real_eigenvalues = pd.to_numeric(famd.eigenvalues_summary["eigenvalue"]).astype(
            np.float32
        )
        effective_n_jobs = self._get_effective_n_jobs()
        # Prepare column information once
        column_info = []
        memory_per_category = 0
        for col in df.columns:
            if df[col].dtype in ["float64", "float32", "int64"]:
                column_info.append(
                    {
                        "name": col,
                        "type": "numeric",
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                    }
                )
            else:
                # Optimize categorical storage by using integers
                value_counts = df[col].value_counts(normalize=True)
                categories = value_counts.index.tolist()
                memory_per_category += len(categories) * 8
                column_info.append(
                    {
                        "name": col,
                        "type": "categorical",
                        "categories": categories,
                        "probabilities": value_counts.values.astype(np.float32),
                        "n_categories": len(categories),
                    }
                )

        # Estimate memory requirements to determine optimal batch size
        memory_per_row = (
            n_features * 4  # float32 for numeric
            + memory_per_category  # categorical overhead
            + n_features * 2  # DataFrame overhead
        )
        available_memory = psutil.virtual_memory().available * 0.8
        memory_per_simulation = memory_per_row * n_samples
        batch_size = max(
            1,
            min(
                25,
                int(available_memory / (memory_per_simulation * effective_n_jobs * 3)),
            ),
        )
        logger.info(
            f"Using {effective_n_jobs} processes with batch size of {batch_size}"
        )

        # Process simulations in batches to manage memory usage
        simulated_eigenvalues = []
        n_batches = (n_simulations + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_simulations)
            batch_size_current = batch_end - batch_start

            # Clear memory before each batch
            gc.collect()

            with parallel_backend("multiprocessing", n_jobs=effective_n_jobs):
                batch_results = Parallel(verbose=self.verbose)(
                    delayed(self._simulation_wrapper_optimized)(
                        n_features, n_samples, column_info
                    )
                    for _ in range(batch_size_current)
                )

            # Filter out None results from failed simulations
            valid_results = [r for r in batch_results if r is not None]
            simulated_eigenvalues.extend(valid_results)

            logger.info(f"Completed batch {batch_idx + 1}/{n_batches}")
        # Calculate significant components
        percentile_95 = np.percentile(simulated_eigenvalues, 95, axis=0)
        significant_components = [
            i
            for i, (real, simulated) in enumerate(zip(real_eigenvalues, percentile_95))
            if real > simulated
        ]

        return significant_components

    def _plot_contributions(self, famd: FAMD, components: list):
        """
        Visualize how variables contribute to principal components.

        Creates a heatmap showing the contribution of each variable to the
        selected components and saves it to the results directory.

        Args:
            famd: Fitted FAMD model
            components: List of component indices to include in the plot
        """
        contributions = famd.column_contributions_

        # Create heatmap
        plt.figure(figsize=(15, len(contributions.index) / 2))
        sns.heatmap(
            contributions.iloc[:, components],
            cmap="YlOrRd",
            cbar_kws={"label": "Contribution"},
        )
        plt.title("Column Contributions to Components")
        plt.xlabel("Components")
        plt.ylabel("Variables")
        plt.tight_layout()
        output_dir = get_results_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "famd_contributions.png")
        plt.close()

    def calculate_wealth_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate household wealth index using Factor Analysis of Mixed Data.

        Processes household survey data, applying FAMD to extract components,
        then combines them into a weighted wealth index correlated with
        standard wealth indicators.

        Args:
            df: DataFrame containing household survey data

        Returns:
            DataFrame with household IDs and calculated wealth index
        """
        logger.info("Starting wealth index calculation")

        # Prepare data
        df_processed = df.copy()

        # Identify household ID columns
        id_cols = ["hv000", "hv001", "hv005", "hv007"]
        df_ids = df_processed[id_cols]
        df_processed = df_processed.drop(columns=id_cols)

        # Convert potential numeric columns stored as strings
        numeric_cols = df_processed.select_dtypes(include=["object"]).columns
        for col in numeric_cols:
            # Try to convert to numeric, identifying actual numeric columns
            try:
                numeric_data = pd.to_numeric(df_processed[col])
                # Determine most efficient numeric dtype

                df_processed[col] = numeric_data.astype(np.float32)
            except (ValueError, TypeError):
                # If conversion to numeric fails, it's truly categorical
                continue

        # Now handle categorical columns (those that couldn't be converted to numeric)
        cat_cols = df_processed.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df_processed[col] = pd.Categorical(df_processed[col])

        # Standardize numerical columns
        num_cols = df_processed.select_dtypes(include=["number"]).columns
        if num_cols.size > 0:
            df_processed[num_cols] = (
                StandardScaler()
                .fit_transform(df_processed[num_cols])
                .astype(np.float32)
            )
        # Perform FAMD
        famd = FAMD(n_components=df_processed.shape[1])
        famd.fit(df_processed)

        # Determine components to use
        if self.config.get("run_parallel_analysis", False):
            start_time = time()
            n_simulations = self.config.get("n_simulations", 1000)
            components = self._run_parallel_analysis(
                famd, df_processed.copy(), n_simulations
            )
            logger.info(
                f"Parallel analysis selected {len(components)} components after {time()-start_time}s"
            )
        else:
            components = list(range(len(famd.eigenvalues_)))
            logger.info("Using all components")

        # Plot contributions if requested
        if self.config.get("plot_contributions", False):
            self._plot_contributions(famd, components)

        # Define indicators that correlate with household wealth
        wealth_indicators = {
            "hv206": ["yes"],  # Has electricity
            "hv207": ["yes"],  # Has radio
            "hv208": ["yes"],  # Has television
            "hv209": ["yes"],  # Has refrigerator
            "hv211": ["yes"],  # Has motorcycle
            "hv212": ["yes"],  # Has car
        }

        # Calculate a simple wealth score to determine component directions
        wealth_score = np.zeros(len(df_processed))
        for col, values in wealth_indicators.items():
            if col in df_processed.columns:
                for val in values:
                    try:
                        wealth_score += (df_processed[col] == val).astype(int)
                    except Exception as e:
                        logger.warning(f"Error processing indicator {col}: {e}")
        row_coords = famd.row_coordinates(df_processed)

        # Determine which components correlate negatively with wealth
        # These need to be inverted to ensure consistent wealth index direction
        invert_components = []
        component_correlations = []
        for comp in components:
            correlation = np.corrcoef(wealth_score, row_coords[comp])[0, 1]
            component_correlations.append(correlation)
            logger.debug(
                f"Component {comp} correlation with wealth score: {correlation:.4f}"
            )

            if correlation < 0:
                invert_components.append(comp)
                logger.info(
                    f"Component {comp} will be inverted (correlation: {correlation:.4f})"
                )
        abs_correlations = np.abs(component_correlations)
        correlation_weights = abs_correlations / np.sum(abs_correlations)
        wealth_index = np.zeros(len(df_processed))
        for i, comp_idx in enumerate(components):
            component_value = row_coords[comp_idx].copy()

            # Invert if needed
            if comp_idx in invert_components:
                component_value = -component_value

            wealth_index += component_value * correlation_weights[i]

        multi = self.config.get("iqr_multiplier", 1.5)
        wealth_index_copy = wealth_index.copy()
        q1, q3 = np.percentile(wealth_index, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (multi * iqr)
        upper_bound = q3 + (multi * iqr)
        wealth_index_copy = np.clip(wealth_index_copy, lower_bound, upper_bound)
        # Create result DataFrame
        result = pd.concat(
            [df_ids, pd.Series(wealth_index_copy, name="wealth_index")], axis=1
        )

        # Normalize wealth index to 0-1 range
        if self.config.get("normalize_index", True):
            min_val = result["wealth_index"].min()
            max_val = result["wealth_index"].max()
            result["wealth_index"] = (result["wealth_index"] - min_val) / (
                max_val - min_val
            )
            logger.info("Wealth index normalized to 0-1 range")
        return result

    def _simulation_wrapper_optimized(self, n_features, n_samples, column_info):
        """
        Generate random data and calculate FAMD eigenvalues for parallel analysis.

        Creates synthetic data matching the statistical properties of the original
        dataset, then performs FAMD to get eigenvalues for significance testing.

        Args:
            n_features: Number of features/columns
            n_samples: Number of samples/rows
            column_info: List of dictionaries with column metadata

        Returns:
            Series of eigenvalues from FAMD on random data
        """
        start_time = time()
        data_dict = {}
        rng = np.random.default_rng()

        # Process numeric columns first
        numeric_columns = [col for col in column_info if col["type"] == "numeric"]
        if numeric_columns:
            means = np.array([col["mean"] for col in numeric_columns], dtype=np.float32)
            stds = np.array([col["std"] for col in numeric_columns], dtype=np.float32)
            numeric_data = (
                rng.standard_normal((n_samples, len(numeric_columns)), dtype=np.float32)
                * stds
                + means
            )

            # Assign to data_dict
            for idx, col in enumerate(numeric_columns):
                data_dict[col["name"]] = numeric_data[:, idx]

            del numeric_data

        # Process categorical columns efficiently
        categorical_columns = [
            col for col in column_info if col["type"] == "categorical"
        ]
        for col in categorical_columns:
            probs = col["probabilities"].astype(np.float32)
            probs /= probs.sum()  # Normalize to ensure sum is exactly 1

            # Handle categorical data generation with memory-efficient approach
            if len(col["categories"]) > np.iinfo(np.int32).max:
                # For extremely large category counts, use cumulative probability approach
                cumprobs = np.cumsum(probs)
                random_values = rng.random(n_samples, dtype=np.float32)
                cat_indices = np.searchsorted(cumprobs, random_values)
            else:
                # For typical category counts, use direct sampling
                cat_indices = rng.choice(
                    len(col["categories"]),
                    size=n_samples,
                    p=probs,
                )

            # Convert indices to categories efficiently
            categories = np.array(col["categories"])
            cat_values = categories[cat_indices]

            # Create categorical data efficiently
            data_dict[col["name"]] = pd.Categorical(
                cat_values, categories=col["categories"], ordered=False
            )

            # Clean up temporary arrays
            del cat_indices, cat_values

        # Create DataFrame with optimized memory usage
        random_df = pd.DataFrame(data_dict)

        # Ensure proper types
        for col in column_info:
            if col["type"] == "numeric" and not np.issubdtype(
                random_df[col["name"]].dtype, np.float32
            ):
                random_df[col["name"]] = random_df[col["name"]].astype(np.float32)

        # Fit FAMD
        famd_random = FAMD(
            n_components=n_features, random_state=rng.integers(0, 2**32 - 1)
        ).fit(random_df)

        result = famd_random.eigenvalues_summary["eigenvalue"].astype(np.float32)
        logger.debug(f"Simulation completed in {time() - start_time:.2f}s")
        for name in ["random_df", "famd_random", "data_dict"]:
            if name in locals():
                del locals()[name]
        gc.collect()
        return result
