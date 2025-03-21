"""
Imputation manager module for handling multiple imputation strategies.

Provides a unified interface to different imputation techniques including KNN,
MICE, and MissForest, allowing for easy switching or comparison between methods
when handling missing data in household surveys.
"""

from src.data_processing.imputation.KNN import KNNImputer
from src.data_processing.imputation.MICE import MICEImputer
from src.data_processing.imputation.MissForest import MissForestImputer
from src.data_processing.imputation.base_imputer import BaseImputer
import pandas as pd
from typing import Dict, Any, Tuple, Union
import logging
from time import time

logger = logging.getLogger("wealth_index.imputer")


class ImputerManager:
    """
    Manages different imputation methods and provides a unified interface.

    Serves as a facade for various imputation techniques, allowing clients
    to easily switch between methods or compare results from multiple
    approaches without changing their code.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize imputation manager with configuration settings.

        Instantiates the required imputation objects based on the specified
        method in the configuration. Supports single method execution or
        comparison mode.

        Args:
            config: Dictionary containing configuration parameters including:
                - imputation: Method to use ('knn', 'mice', 'missforest', or 'compare')
                - Various method-specific parameters passed to the imputers
        """
        self.config = config
        self.method = config.get("imputation", "mice").lower()

        # Initialize required imputers based on method
        if self.method == "knn" or self.method == "compare":
            self.knn_imputer = KNNImputer(config)

        if self.method == "mice" or self.method == "compare":
            self.mice_imputer = MICEImputer(config)

        if self.method == "missforest" or self.method == "compare":
            self.missforest_imputer = MissForestImputer(config)

        logger.info(f"Initialized imputer manager with method: {self.method}")

    def impute(self, df: pd.DataFrame, country_year: str) -> Union[pd.DataFrame, Tuple]:
        """
        Impute missing values using the selected method.

        Delegates to the appropriate imputation implementation based on the
        configured method. When in 'compare' mode, runs all implemented methods
        and returns their results for comparison.

        Args:
            df: Input dataframe with missing values
            country_year: String identifier for the country and year being processed

        Returns:
            For single methods: Tuple of (imputed_dataframe, quality_score)
            For compare method: Tuple of (knn_df, knn_score, mice_df, mice_score,
                                        missforest_df, missforest_score)

        Raises:
            ValueError: If an unknown imputation method is specified
        """
        if self.method == "knn":
            return self.knn_imputer.impute(df.copy(), country_year)
        elif self.method == "mice":
            return self.mice_imputer.impute(df.copy(), country_year)
        elif self.method == "missforest":
            return self.missforest_imputer.impute(df.copy(), country_year)
        elif self.method == "compare":
            # Run all imputation methods to allow comparison of their effectiveness
            start_time = time()
            knn_df, knn_score = self.knn_imputer.impute(df.copy(), country_year)
            mice_df, mice_score = self.mice_imputer.impute(df.copy(), country_year)
            missforest_df, missforest_score = self.missforest_imputer.impute(
                df.copy(), country_year
            )
            logger.info(
                f"Imputation comparison of {country_year} took {time()-start_time:.2f} seconds"
            )
            return (
                knn_df,
                knn_score,
                mice_df,
                mice_score,
                missforest_df,
                missforest_score,
            )
        else:
            logger.error(f"Unknown imputation method: {self.method}")
            raise ValueError(f"Unknown imputation method: {self.method}")
