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
    """Manages different imputation methods and provides a unified interface."""

    def __init__(self, config: Dict[str, Any]):
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

        Args:
            df: Input dataframe with missing values

        Returns:
            For single methods: Tuple of (imputed_dataframe, score)
            For compare method: Tuple of (knn_df, knn_score, mice_df, mice_score, missforest_df, missforest_score)
        """
        if self.method == "knn":
            return self.knn_imputer.impute(df.copy(), country_year)
        elif self.method == "mice":
            return self.mice_imputer.impute(df.copy(), country_year)
        elif self.method == "missforest":
            return self.missforest_imputer.impute(df.copy(), country_year)
        elif self.method == "compare":
            # Run all imputation methods and return their results
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
