import pandas as pd
import numpy as np
from prince import FAMD
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger("wealth_index.famd")


class FAMDAnalyzer:
    """Factor Analysis of Mixed Data implementation."""

    def __init__(self, config: dict):
        self.config = config

    def calculate_wealth_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate wealth index using FAMD."""
        logger.info("Starting wealth index calculation")

        # Prepare data
        df_processed = df.copy()
        id_cols = ["hv000", "hv001", "hv005"]
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

        # Calculate wealth index
        transformed_data = famd.transform(df_processed)
        explained_variance = famd.explained_inertia_ratio_

        wealth_index = np.sum(
            transformed_data * explained_variance.reshape(1, -1), axis=1
        )

        # Create result DataFrame
        result = pd.concat(
            [df_ids, pd.Series(wealth_index, name="wealth_index")], axis=1
        )

        return result
