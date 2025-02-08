import pandas as pd
import numpy as np
import json
import logging
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from prince import FAMD
import matplotlib.pyplot as plt
from collections import Counter
import time
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module=".*", append=True)


###Load Configuration
class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path) as f:
            return json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)


###Process data
class DataProcessor:
    def __init__(self, config):
        self.config = config

    ###Loading data
    def load_data(self, file_path):
        logging.info(f"Loading data from {file_path}")
        return pd.read_stata(file_path)

    ###Saving csv files
    def save_csv(self, df, file_path):
        logging.info(f"Saving DataFrame to {file_path}")
        df.to_csv(file_path, index=False)

    ###Preprocessing removing known columns unwanted
    def preprocess_data(self, df):
        logging.info("Preprocessing data")
        columns_to_include = self.config.get("columns_to_include")
        df = df[columns_to_include]
        df = df.dropna(axis=1, how="all")
        logging.debug(f"Columns after preprocessing: {df.columns}")
        return df

    ###Adding Na where related column is no
    def na_replace(self, df):
        logging.info("Replacing NA values")
        na_columns = self.config.get("na_columns")
        for target, condition in na_columns.items():
            df[target] = df[target].where(
                ~df[condition].str.lower().str.contains("no", na=False), "NotApplicable"
            )
        logging.debug(f"Columns after NA replacement: {df.columns}")
        return df

    ###Replacing values with easier to handle
    def replace_val(self, df):
        logging.info("Replacing values")
        replace_val = self.config.get("replace_val")
        for value, replace in replace_val.items():
            df = df.replace(value, replace)
        logging.debug(f"Columns after value replacement: {df.columns}")
        return df

    ###Dropping all columns with less than 50% filled
    def drop_less50(self, df):
        logging.info("Dropping columns with more than 50% missing values")
        threshold = 0.5 * len(df)
        df = df.loc[:, df.isnull().sum() <= threshold]
        logging.debug(f"Columns after dropping: {df.columns}")
        return df

    ###Removing all columns that were removed from the processing of any dataframe for consistency
    def handle_partially_missing_columns(self, dfs):
        all_columns = [set(df.columns) for df in dfs.values()]
        common_columns = set.intersection(*all_columns)
        dfs_cleaned = {name: df[list(common_columns)] for name, df in dfs.items()}
        return dfs_cleaned, common_columns

    ###Overall process_data
    def process_data(self, dfs):
        logging.info("Processing data")
        for df in dfs:
            df = self.preprocess_data(df)
            df = self.na_replace(df)
            df = self.replace_val(df)
            df = self.drop_less50(df)
        dfs, common_columns = self.handle_partially_missing_columns(dfs)
        logging.info("Data processing complete")
        logging.debug(f"Columns after processing: {common_columns}")
        return dfs


###Imputing using KNN
class KNN_Imputer:
    def __init__(self, config):
        self.config = config

    ###FIXME: Need to remove certain columns (cluster ID and household weight for ex), from here maybe? Or is it worth keeping?
    ###Overall imputation
    def knn_imputations(self, df):
        start_time = time.time()
        logging.info("Imputing missing values using KNN")
        dropped_columns = df[["hv000", "hv001", "hv005"]]
        df = df.drop(columns=["hv000", "hv001", "hv005"])
        k_values = self.config.get("k_values")
        best_k_dict = self.best_ks(df, k_values)  ###Calculating best k_values
        df_imputed = df.copy()
        for column in df.columns:
            if df[column].isna().sum() > 0 and column in best_k_dict:
                logging.info(f"Imputing missing values for column {column}")
                best_k = best_k_dict[column]
                knn = KNeighborsClassifier(n_neighbors=best_k)  ###KNN using best k

                ###Checking for missing data
                X_train = df.drop(column, axis=1).astype(str).fillna("missing")

                ###Convering to a dataset that can be handled
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

                ##Training
                X_train_encoded = encoder.fit_transform(X_train)
                y_train = df[column]
                non_nan_mask = y_train.notna()
                missing_mask = y_train.isna()  ###Checking for missing data
                knn.fit(
                    X_train_encoded[non_nan_mask], y_train[non_nan_mask].astype(str)
                )

                ###Estimating missing data
                if missing_mask.sum() > 0:
                    X_missing = X_train_encoded[missing_mask]
                    imputed_values = knn.predict(X_missing)
                    if df_imputed[column].dtype == "category":
                        current_categories = df_imputed[column].cat.categories
                        new_categories = pd.Index(imputed_values).unique()
                        updated_categories = current_categories.union(new_categories)
                        df_imputed[column] = df_imputed[column].cat.set_categories(
                            updated_categories
                        )
                    df_imputed.loc[missing_mask, column] = imputed_values
        logging.info(
            f"Imputation complete, remaining missing values: {df_imputed.isna().sum().sum()}"
        )
        df_imputed[["hv000", "hv001", "hv005"]] = dropped_columns
        logging.debug(f"Columns after imputation: {df_imputed.columns}")
        logging.info(f"Time taken for imputation: {time.time() - start_time} seconds")
        return df_imputed

    ###Calculating best ks for each column
    def best_ks(self, df, k_values):
        logging.info("Finding the best k values for KNN imputation")
        best_k = {}
        for column_name in df.columns:
            test_data = df[df[column_name].isna()]
            if test_data.empty:
                continue
            train_data = df.dropna(subset=[column_name]).astype(str)
            X_train = train_data.drop(column_name, axis=1)
            y_train = train_data[column_name]
            X_test = test_data.drop(column_name, axis=1)

            ###Converting to data type easier to interpolate
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_train_encoded = encoder.fit_transform(X_train)
            X_test_encoded = encoder.transform(X_test)
            mode_scores = []

            ###Picking k based on mode score
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_encoded, y_train)
                imputed_values = knn.predict(X_test_encoded)
                mode_scores.append(imputed_values)
            mode_counts = [
                Counter(scores).most_common(1)[0][1] for scores in mode_scores
            ]
            best_k[column_name] = k_values[np.argmax(mode_counts)]
        logging.debug(f"Best k values: {best_k}")
        return best_k


###For generating FAMD
class FAMDAnalyzer:
    def __init__(self, config):
        self.config = config

    ###Parallel analysis as a monte carlo simulation to determine usable components
    def parallel_analysis(self, n_simulations, actual_eigenvalues, df):
        start_time = time.time()
        logging.info("Performing parallel analysis")
        simulated_eigenvalues = np.zeros((n_simulations, len(actual_eigenvalues)))
        categorical_cols = df.select_dtypes(include=["category", "object"]).columns
        numerical_cols = df.select_dtypes(include=["number"]).columns
        for i in range(n_simulations):
            simulated_df = pd.DataFrame()

            ###Normal distirbution for numerical
            for col in numerical_cols:
                simulated_df[col] = np.random.normal(
                    df[col].mean(), df[col].std(), size=len(df)
                )

            ###Picking at random for categorical
            for col in categorical_cols:
                simulated_df[col] = np.random.choice(
                    df[col].unique(),
                    size=len(df),
                    p=df[col].value_counts(normalize=True).values,
                )
            famd_sim = FAMD(n_components=df.shape[1])
            famd_sim.fit(simulated_df)

            ###Stroing eigen values
            simulated_eigenvalues[i, :] = famd_sim.eigenvalues_summary[
                "eigenvalue"
            ].astype(float)

        ###threshold the actual eigenvalues must beat to be kept
        threshold = np.percentile(simulated_eigenvalues, 95, axis=0)
        components_to_keep = np.where(actual_eigenvalues > threshold)[0]
        logging.info("Parallel analysis complete")
        logging.debug(f"Components to keep: {len(components_to_keep)}")
        logging.info(
            f"Time taken for parallel analysis: {time.time() - start_time} seconds"
        )
        return len(components_to_keep)

    ###Displaying the scree plot
    def display_scree_plot(self, explained_variance):
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(explained_variance) + 1),
                explained_variance,
                marker="o",
                linestyle="--",
            )
            plt.title("Scree Plot")
            plt.xlabel("Principal Components")
            plt.ylabel("Explained Variance")
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error while displaying scree plot: {e}")

    def save_csv(self, df, file_path):
        logging.info(f"Saving df to {file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
        df.to_csv(file_path)

    ###Printing column contributions
    def print_loadings(self, famd):
        try:
            logging.info("Printing loadings")
            repo_root = os.path.dirname(os.path.abspath(__file__))
            loadings_file = "Loadings.csv"
            loadings_file_path = os.path.join(
                repo_root, self.config["results_path"], loadings_file
            )
            self.save_csv(famd.column_contributions_, loadings_file_path)
        except Exception as e:
            logging.error(f"Error while printing loadings: {e}")

    ###Calculating the FAMD
    def FAMD_calc(
        self, df, n_simulations=10, print_loadings=False, display_scree_plot=False
    ):
        logging.info("Calculating Wealth Index using FAMD")
        df = df.apply(
            lambda x: x.astype("float64") if pd.api.types.is_numeric_dtype(x) else x
        )

        ###Columns that will be needed but shouldn't be used for the FAMD
        dropped_columns = df[["hv000", "hv001", "hv005"]]
        df = df.drop(columns=["hv000", "hv001", "hv005"])

        ###Standardizing types
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].astype("float64")
            else:
                df[column] = df[column].astype("category")
                df[column] = df[column].cat.codes.astype(str)

        ###Scaling continuous data
        continuous_columns = df.select_dtypes(include="float64").columns
        scaler = StandardScaler()
        df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
        famd = FAMD(n_components=df.shape[1])
        famd.fit(df)

        ###Explained variance for scree plot or the weighted average
        explained_variance = famd.eigenvalues_summary["% of variance"].apply(
            lambda x: float(x.strip("%")) / 100
        )
        cum_explained_variance = famd.eigenvalues_summary[
            "% of variance (cumulative)"
        ].apply(lambda x: float(x.strip("%")) / 100)

        ###Eigenvalues to compre to monte carlo result
        actual_eigenvalues = famd.eigenvalues_summary["eigenvalue"].astype(float)
        num_components = self.parallel_analysis(n_simulations, actual_eigenvalues, df)
        cum_explained_variance_threshold = cum_explained_variance[num_components - 1]
        principal_components = famd.transform(df)

        ###wealth index calculated as a weighted sum of explained variances
        wealth_index = sum(
            principal_components[i]
            * explained_variance[i]
            / cum_explained_variance_threshold
            for i in range(num_components)
        )
        if display_scree_plot:
            self.display_scree_plot(explained_variance)
        if print_loadings:
            self.print_loadings(famd)
        df[["hv000", "hv001", "hv005"]] = dropped_columns
        df["wealth_index"] = wealth_index
        logging.info("Wealth Index calculation complete")
        logging.debug(f"Wealth index results: {df['wealth_index']}")
        return df


class WealthIndexCalculator:
    ###Initial class
    def __init__(self, config_path):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.data_processor = DataProcessor(self.config)
        self.imputer = KNN_Imputer(self.config)
        self.famd_analyzer = FAMDAnalyzer(self.config)

    def calculate_wealth_index(self):
        ###main calculator function
        repo_root = os.path.dirname(os.path.abspath(__file__))
        dfs = {}
        log_level = self.config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level, logging.INFO)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("wealth_index_creation.log"),
                logging.StreamHandler(),
            ],
        )
        for country, years in self.config["country_year"].items():
            for year in years:
                file_path = os.path.join(repo_root, "data", "DHS", country, year)
                file_name = next(
                    (f for f in os.listdir(file_path) if f.endswith(".DTA")), None
                )
                if file_name:
                    dfs[f"{country}_{year}"] = pd.read_stata(
                        os.path.join(file_path, file_name)
                    )
                else:
                    logging.info(f"Could not find {country} {year} data.")

        dfs = self.data_processor.process_data(dfs)

        for df in dfs:
            if self.config["imputation"] == "KNN":
                df = self.imputer.knn_imputations(df)
        check = 0
        for name, df in dfs.items():
            if df.isna().sum().sum() > 0:  # Check if the DataFrame has any NaN values
                logging.info(f"Missing data found in {name}:")
                check += 1
            if check != 0:
                logging.info("Missing data found, exiting.")
                exit()
        merged_df = pd.concat(dfs.values(), axis=0, ignore_index=True)
        merged_df = self.famd_analyzer.FAMD_calc(
            merged_df,
            self.config["n_simulations"],
            self.config["print_loadings"],
            self.config["display_scree_plot"],
        )
        csv_file = "Wealth_Index_DF.csv"
        csv_file_path = os.path.join(repo_root, self.config["results_path"], csv_file)
        self.data_processor.save_csv(merged_df, csv_file_path)


if __name__ == "__WealthIndex__":
    ###Calling the needed classes
    calculator = WealthIndexCalculator("Configs/config.json")
    calculator.calculate_wealth_index()
