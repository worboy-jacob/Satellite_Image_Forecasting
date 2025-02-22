# imputer_selector.py

from src.data_processing.imputation.KNN import KNNImputer
from src.data_processing.imputation.MICE import MICEImputer
import pandas as pd
from typing import Dict, Any
import logging
import sys

logger = logging.getLogger("wealth_index.imputer")


class ImputerManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knn_imputer = KNNImputer(config)
        self.mice_imputer = MICEImputer(config)
        self.method = config.get("imputation", "mice")
        print(self.method)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method.lower() == "knn":
            return self.knn_imputer.impute(df)
        elif self.method.lower() == "mice":
            return self.mice_imputer.impute(df)
        else:
            logger.error(f"No method {self.method}")
            sys.exit()
