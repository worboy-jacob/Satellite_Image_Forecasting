"""
Main entry point for the wealth index calculation pipeline.

This module orchestrates the end-to-end process of loading household survey data,
performing imputation of missing values, calculating wealth indices using FAMD,
and saving the results to disk in multiple formats.
"""

import sys
import time
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    get_project_root,
    get_configs_dir,
    get_dhs_dir,
    get_results_dir,
)
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.data_processing.processor import DataProcessor
from src.data_processing.imputation.imputer import ImputerManager
from src.analysis.famd import FAMDAnalyzer
from src.utils.validation import validate_data_structure

# Add project root to Python path to ensure imports work correctly
project_root = get_project_root()
sys.path.insert(0, str(project_root))


def process_data(config, logger):
    """
    Process data according to configuration settings.

    Orchestrates the full data processing pipeline including data loading,
    missing value imputation, and wealth index calculation.

    Args:
        config: Configuration dictionary with processing parameters
        logger: Logger for status and error messages

    Returns:
        DataFrame containing calculated wealth indices

    Raises:
        ValueError: If no data files were successfully loaded
    """
    # Initialize components
    imputer_manager = ImputerManager(config)
    processor = DataProcessor(config)
    analyzer = FAMDAnalyzer(config)

    # Load data
    logger.info("Loading data files")
    data_frames = processor.load_all_data()
    if not data_frames:
        raise ValueError("No data files were loaded successfully")

    # Process data with appropriate imputation method
    imputation_method = config.get("imputation", "mice").lower()
    if imputation_method == "compare":
        processed_frames = compare_imputation_methods(
            data_frames, imputer_manager, logger
        )
    else:
        processed_frames = impute_datasets(data_frames, imputer_manager, logger)

    # Merge processed data
    logger.info("Merging processed datasets")
    merged_df = pd.concat(processed_frames.values(), axis=0, ignore_index=True)

    # Calculate wealth index
    logger.info("Calculating wealth index")
    return analyzer.calculate_wealth_index(merged_df)


def impute_datasets(data_frames, imputer_manager, logger):
    """
    Impute missing values in each dataset using the configured method.

    Processes each country-year dataset independently with the same
    imputation method specified in the configuration.

    Args:
        data_frames: Dictionary of DataFrames to process
        imputer_manager: ImputerManager configured with imputation method
        logger: Logger for status messages

    Returns:
        Dictionary of processed DataFrames with imputed values
    """
    logger.info("Processing datasets")
    processed_frames = {}

    for country_year, df in data_frames.items():
        logger.info(f"Processing dataset: {country_year}")
        imputed_df, _ = imputer_manager.impute(df, country_year)
        processed_frames[country_year] = imputed_df

    return processed_frames


def compare_imputation_methods(data_frames, imputer_manager, logger):
    """
    Compare different imputation methods and select the best performing one.

    Runs KNN, MICE, and MissForest imputation on each dataset, computes
    weighted quality scores, and selects the method with the lowest error.

    Args:
        data_frames: Dictionary of DataFrames to process
        imputer_manager: ImputerManager configured for comparison
        logger: Logger for status messages

    Returns:
        Dictionary of DataFrames processed with the best imputation method
    """
    logger.info("Comparing imputation methods")

    # Track missing value counts to weight the importance of each dataset
    df_missing_count = {}
    knn_frames, mice_frames, missforest_frames = {}, {}, {}
    knn_scores, mice_scores, missforest_scores = {}, {}, {}
    total_missing = 0

    # Process each dataset with different imputation methods
    for country_year, df in data_frames.items():
        logger.info(f"Processing dataset: {country_year}")
        df_missing_count[country_year] = df.isnull().sum().sum()
        total_missing += df_missing_count[country_year]

        knn_df, knn_score, mice_df, mice_score, missforest_df, missforest_score = (
            imputer_manager.impute(df, country_year)
        )

        # Store results
        knn_frames[country_year] = knn_df
        mice_frames[country_year] = mice_df
        missforest_frames[country_year] = missforest_df
        knn_scores[country_year] = knn_score
        mice_scores[country_year] = mice_score
        missforest_scores[country_year] = missforest_score

    # Calculate weighted scores
    weighted_scores = calculate_weighted_imputation_scores(
        data_frames.keys(),
        knn_scores,
        mice_scores,
        missforest_scores,
        df_missing_count,
        total_missing,
    )

    # Select best imputation method
    return select_best_imputation_method(
        weighted_scores, knn_frames, mice_frames, missforest_frames, logger
    )


def calculate_weighted_imputation_scores(
    country_years,
    knn_scores,
    mice_scores,
    missforest_scores,
    df_missing_count,
    total_missing,
):
    """
    Calculate weighted imputation scores based on missing data proportions.

    Weights each dataset's imputation score by its proportion of total
    missing values to prioritize datasets with more missing data.

    Args:
        country_years: List of country-year identifiers
        knn_scores: Dictionary of KNN imputation scores by country-year
        mice_scores: Dictionary of MICE imputation scores by country-year
        missforest_scores: Dictionary of MissForest scores by country-year
        df_missing_count: Dictionary of missing value counts by country-year
        total_missing: Total count of missing values across all datasets

    Returns:
        Dictionary with weighted scores for each imputation method
    """
    knn_score, mice_score, missforest_score = 0, 0, 0

    for country_year in country_years:
        weight = df_missing_count[country_year] / total_missing
        knn_score += knn_scores[country_year] * weight
        mice_score += mice_scores[country_year] * weight
        missforest_score += missforest_scores[country_year] * weight

    return {"knn": knn_score, "mice": mice_score, "missforest": missforest_score}


def select_best_imputation_method(
    scores, knn_frames, mice_frames, missforest_frames, logger
):
    """
    Select the best imputation method based on quality scores.

    Compares the weighted scores of each imputation method and returns
    the set of DataFrames processed with the best-performing method.

    Args:
        scores: Dictionary of weighted scores for each method
        knn_frames: Dictionary of KNN-imputed DataFrames
        mice_frames: Dictionary of MICE-imputed DataFrames
        missforest_frames: Dictionary of MissForest-imputed DataFrames
        logger: Logger for status messages

    Returns:
        Dictionary of DataFrames processed with the best method
    """
    logger.info(
        f"KNN score: {scores['knn']:.2f}, MICE score: {scores['mice']:.2f}, "
        f"MissForest score: {scores['missforest']:.2f}"
    )

    if scores["knn"] < scores["mice"] and scores["knn"] < scores["missforest"]:
        logger.info("KNN imputation method selected")
        return knn_frames
    elif scores["missforest"] < scores["knn"] and scores["missforest"] < scores["mice"]:
        logger.info("MissForest imputation method selected")
        return missforest_frames
    else:
        logger.info("MICE imputation method selected")
        return mice_frames


def save_results(result_df, config, logger):
    """
    Save results to disk in multiple formats.

    Writes the wealth index results to both Parquet and CSV formats
    in the configured output location.

    Args:
        result_df: DataFrame containing calculated wealth indices
        config: Configuration dictionary with output settings
        logger: Logger for status messages
    """
    output_path = get_results_dir() / config.get("output_file", "wealth_index")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save in multiple formats
    result_df.to_parquet(output_path.with_suffix(".parquet"), engine="pyarrow")
    result_df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"Results saved to {output_path}")


def main():
    """
    Main entry point for the wealth index calculation pipeline.

    Orchestrates the end-to-end process from configuration loading
    through data processing to result saving, with comprehensive
    error handling and execution time tracking.
    """

    # Measure execution time for performance monitoring
    start_time = time.time()

    try:
        # Validate environment and data structure before proceeding
        validate_data_structure()

        # Load configuration
        config_path = get_configs_dir() / "config.yaml"
        config = Config(config_path).config
        logger = setup_logging(log_level=config.get("log_level", "INFO"))
        logger.info("Starting wealth index calculation")

        # Process data and calculate wealth index
        result_df = process_data(config, logger)

        # Save results
        save_results(result_df, config, logger)

        # Log completion with timing information for performance analysis
        execution_time = time.time() - start_time
        logger.info(f"Processing completed in {execution_time:.2f} seconds")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        logger.info("Process finished")


if __name__ == "__main__":
    main()
