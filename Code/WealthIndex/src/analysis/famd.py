import pandas as pd
import numpy as np
from prince import FAMD
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from pathlib import Path
from src.utils.paths import get_results_dir

logger = logging.getLogger("wealth_index.famd")


class FAMDAnalyzer:
    """Factor Analysis of Mixed Data implementation."""

    def __init__(self, config: dict):
        self.config = config

    def _run_parallel_analysis(
        self, famd, df: pd.DataFrame, n_simulations: int
    ) -> list:
        """Run parallel analysis to determine significant components."""
        n_samples, n_features = df.shape
        real_eigenvalues = pd.to_numeric(famd.eigenvalues_summary["eigenvalue"])

        # Store simulated eigenvalues
        simulated_eigenvalues = np.zeros((n_simulations, n_features))

        for i in range(n_simulations):
            print(
                f"running simulation {i+1} of {n_simulations}"
            )  ###TODO: change to log
            # Create random dataset with same structure
            random_df = df.copy()
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    random_df[col] = np.random.normal(size=n_samples)
                else:
                    random_df[col] = np.random.choice(df[col].unique(), size=n_samples)

            # Get eigenvalues from random data
            famd_random = FAMD(n_components=n_features).fit(random_df)
            simulated_eigenvalues[i] = pd.to_numeric(
                famd_random.eigenvalues_summary["eigenvalue"]
            )

        # Get 95th percentile of simulated eigenvalues
        percentile_95 = np.percentile(simulated_eigenvalues, 95, axis=0)

        # Keep components where real eigenvalues > 95th percentile
        ###TODO: double check that this formula works
        significant_components = [
            i
            for i, (real, simulated) in enumerate(zip(real_eigenvalues, percentile_95))
            if real > simulated
        ]

        return significant_components

    def _plot_contributions(self, famd: FAMD, components: list):
        """Plot column contributions for selected components."""
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
        """Calculate wealth index using FAMD."""
        logger.info("Starting wealth index calculation")

        # Prepare data
        df_processed = df.copy()
        id_cols = ["hv000", "hv001", "hv005", "hv007"]
        df_ids = df_processed[id_cols]
        df_processed = df_processed.drop(columns=id_cols)

        # Standardize numerical columns
        num_cols = df_processed.select_dtypes(include=["float64", "int64"]).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

        # Perform FAMD
        famd = FAMD(n_components=df_processed.shape[1])
        famd.fit(df_processed)

        # Determine components to use
        if self.config.get("run_parallel_analysis", False):
            n_simulations = self.config.get("n_simulations", 1000)
            components = self._run_parallel_analysis(famd, df_processed, n_simulations)
            logger.info(f"Parallel analysis selected {len(components)} components")
        else:
            components = [0]  # Just use first component
            logger.info("Using only first component")

        # Plot contributions if requested
        if self.config.get("plot_contributions", False):
            self._plot_contributions(famd, components)

        # Calculate wealth index
        var_explained = (
            pd.to_numeric(famd.eigenvalues_summary["% of variance"].str.rstrip("%"))
            / 100
        )
        component_weights = var_explained[components] / var_explained[components].sum()

        wealth_index = np.zeros(len(df_processed))
        for i, comp in enumerate(components):
            wealth_index += (
                famd.row_coordinates(df_processed)[comp] * component_weights[i]
            )

        # Create result DataFrame
        result = pd.concat(
            [df_ids, pd.Series(wealth_index, name="wealth_index")], axis=1
        )  ###TODO: Normalize from 0 to 1

        return result
