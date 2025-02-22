# main.py
import os
import sys
from pathlib import Path
import pandas as pd
import logging
import time


from src.utils.paths import get_project_root, get_logs_dir, get_configs_dir

project_root = get_project_root()
sys.path.insert(0, str(project_root))
###TODO: Update log levels to be more inline with what we want
# Local imports
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.data_processing.processor import DataProcessor
from src.data_processing.imputation.imputer import ImputerManager
from src.analysis.famd import FAMDAnalyzer


def create_directory_structure():
    """Create necessary directory structure if it doesn't exist."""
    directories = ["logs", "data/Results/WealthIndex"]
    for directory in directories:
        Path(project_root / directory).mkdir(parents=True, exist_ok=True)


def validate_data_structure():
    """Validate that required data directories and files exist."""
    from src.utils.paths import get_dhs_dir

    required_paths = [
        str(get_configs_dir() / "config.yaml"),
        str(get_dhs_dir() / "Ghana_Data"),  ###TODO: make this automatic
        str(get_dhs_dir() / "Senegal_Data"),
    ]
    missing_paths = []
    for path in required_paths:
        if not (project_root / path).exists():
            missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(f"Required paths not found: {', '.join(missing_paths)}")


def main():
    start_time = time.time()

    try:
        # Initial setup
        create_directory_structure()
        validate_data_structure()

        # Load configuration
        config_path = get_configs_dir() / "config.yaml"
        config = Config(config_path).config
        logger = setup_logging(
            log_level=config.get("log_level", "INFO")
        )  ###TODO: add level from config
        logger.info("Starting wealth index calculation")
        imputer_manager = ImputerManager(config)
        # Initialize processors
        processor = DataProcessor(config)
        analyzer = FAMDAnalyzer(config)

        # Load and process data
        logger.info("Loading data files")
        data_frames = processor.load_all_data()

        if not data_frames:
            raise ValueError("No data files were loaded successfully")

        # Process each dataset
        logger.info("Processing datasets")
        processed_frames = {}
        for country_year, df in data_frames.items():
            logger.info(f"Processing dataset: {country_year}")

            # Impute missing values
            imputed_df = imputer_manager.impute(df)
            processed_frames[country_year] = imputed_df

        # Merge all processed dataframes
        logger.info("Merging processed datasets")
        merged_df = pd.concat(processed_frames.values(), axis=0, ignore_index=True)

        # Calculate wealth index
        logger.info("Calculating wealth index")
        result_df = analyzer.calculate_wealth_index(merged_df)

        # Save results
        output_path = project_root / config["results_path"] / "wealth_index"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path.with_suffix(".parquet"), engine="pyarrow")
        result_df.to_csv(output_path.with_suffix(".csv"), index=False)

        execution_time = time.time() - start_time
        logger.info(f"Processing completed in {execution_time:.2f} seconds")
        logger.info(f"Results saved to {output_path}")

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
