# Modified main.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
from label import (
    load_grid_cells,
    load_wealth_data,
    calculate_coverage,
    analyze_thresholds,
    plot_results,
)
from logging_config import setup_logging
from paths import get_config_dir, get_grid_path, get_wealthindex_paths, get_output_dir
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_by_year(grid_gdf_year, min_threshold, max_threshold, year, steps=20):
    """Analyze thresholds for a single year's data."""
    from label import analyze_thresholds

    results_df = analyze_thresholds(grid_gdf_year, min_threshold, max_threshold, steps)
    # Add year column to results
    results_df["year"] = year
    return results_df


def aggregate_results(year_results_dfs):
    """Aggregate results from multiple years."""
    # Create an empty DataFrame to store aggregated results
    all_years = pd.concat(year_results_dfs)

    # Group by threshold to calculate aggregate statistics
    thresholds = sorted(all_years["threshold"].unique())
    aggregated_results = []

    for threshold in thresholds:
        threshold_data = all_years[all_years["threshold"] == threshold]

        # Calculate average and range for percent cells labelled
        avg_percent = threshold_data["percent_cells_labelled"].mean()
        min_percent = threshold_data["percent_cells_labelled"].min()
        max_percent = threshold_data["percent_cells_labelled"].max()

        # Calculate average and range for Moran's I (if available)
        morans_i_values = [
            row["spatial_autocorr"]["I"]
            for _, row in threshold_data.iterrows()
            if row["spatial_autocorr"] is not None
        ]

        if morans_i_values:
            avg_morans_i = np.mean(morans_i_values)
            min_morans_i = min(morans_i_values)
            max_morans_i = max(morans_i_values)
            # Get p-values too
            p_values = [
                row["spatial_autocorr"]["p_value"]
                for _, row in threshold_data.iterrows()
                if row["spatial_autocorr"] is not None
            ]
            avg_p_value = np.mean(p_values)
            spatial_autocorr = {
                "I": avg_morans_i,
                "I_min": min_morans_i,
                "I_max": max_morans_i,
                "p_value": avg_p_value,
            }
        else:
            spatial_autocorr = None

        # Calculate average and range for confidence interval width (if available)
        ci_widths = [
            row["confidence_interval"]["width"]
            for _, row in threshold_data.iterrows()
            if row["confidence_interval"] is not None
        ]

        if ci_widths:
            avg_ci_width = np.mean(ci_widths)
            min_ci_width = min(ci_widths)
            max_ci_width = max(ci_widths)
            confidence_interval = {
                "width": avg_ci_width,
                "width_min": min_ci_width,
                "width_max": max_ci_width,
            }
        else:
            confidence_interval = None

        # Store aggregated results
        aggregated_results.append(
            {
                "threshold": threshold,
                "percent_cells_labelled": avg_percent,
                "percent_cells_min": min_percent,
                "percent_cells_max": max_percent,
                "num_years": len(threshold_data),
                "spatial_autocorr": spatial_autocorr,
                "confidence_interval": confidence_interval,
            }
        )

    return pd.DataFrame(aggregated_results)


def main():
    """Run the analysis using configuration from config.yaml."""
    logger = setup_logging("INFO")
    logger.info("Starting wealth coverage threshold analysis for multiple years...")
    config_path = get_config_dir() / "config.yaml"

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Extract configuration values
    grid_gpkg_path = get_grid_path()
    wealth_gpkg_paths = (
        get_wealthindex_paths()
    )  # This should now return a list of paths
    output_dir = get_output_dir()
    min_threshold = config.get("min_threshold", 20)
    max_threshold = config.get("max_threshold", 40)
    steps = config.get("threshold_steps", 20)

    # Validate required configuration
    if not grid_gpkg_path or not wealth_gpkg_paths:
        logger.error(
            "Error: grid_gpkg_path and wealth_gpkg_paths must be specified in config file"
        )
        sys.exit(1)

    # Ensure wealth_gpkg_paths is a list
    if not isinstance(wealth_gpkg_paths, list):
        wealth_gpkg_paths = [wealth_gpkg_paths]

    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            sys.exit(1)

    # Load grid cells (only once)
    try:
        logger.info("Loading grid cells...")
        grid_gdf = load_grid_cells(grid_gpkg_path)
        grid_gdf.to_crs("EPSG:32628", inplace=True)
        logger.info(f"Grid crs: {grid_gdf.crs}")
    except Exception as e:
        logger.error(f"Fatal error loading grid data: {e}")
        sys.exit(1)

    # Store results for each year
    year_grid_gdfs = {}
    year_results = []

    # Process each wealth index file
    for i, wealth_path in enumerate(wealth_gpkg_paths):
        try:
            # Extract year from filename (adjust this based on your actual filename pattern)
            year = os.path.basename(wealth_path).split("_")[1]
            logger.info(f"Processing wealth index for {year}...")

            # Load wealth data
            wealth_gdf = load_wealth_data(wealth_path)
            wealth_gdf.to_crs("EPSG:32628", inplace=True)
            logger.info(f"Wealth index crs: {wealth_gdf.crs}")

            # Calculate coverage
            logger.info(f"Calculating coverage percentages for {year}...")
            grid_gdf_year = calculate_coverage(grid_gdf.copy(), wealth_gdf.copy())

            # Store grid_gdf for this year
            year_grid_gdfs[year] = grid_gdf_year

            # Analyze thresholds for this year
            logger.info(f"Analyzing thresholds for {year}...")
            year_result = analyze_by_year(
                grid_gdf_year, min_threshold, max_threshold, year, steps
            )
            year_results.append(year_result)

            # Save year-specific results
            if output_dir:
                year_results_path = os.path.join(
                    output_dir, f"{year}_threshold_analysis.csv"
                )
                year_result.to_csv(year_results_path, index=False)
                logger.info(f"Saved {year} threshold analysis to {year_results_path}")

                # Generate year-specific plots
                year_plot_path = os.path.join(
                    output_dir, f"{year}_threshold_analysis_plots.png"
                )
                plot_results(year_result, year_plot_path)

        except Exception as e:
            logger.error(f"Error processing wealth data for {wealth_path}: {e}")
            continue

    if not year_results:
        logger.error("No valid wealth data processed. Exiting.")
        sys.exit(1)

    # Aggregate results across years
    logger.info("Aggregating results across all years...")
    aggregated_results = aggregate_results(year_results)

    # Save aggregated results
    if output_dir:
        agg_results_path = os.path.join(output_dir, "aggregated_threshold_analysis.csv")
        try:
            aggregated_results.to_csv(agg_results_path, index=False)
            logger.info(f"Saved aggregated threshold analysis to {agg_results_path}")
        except Exception as e:
            logger.error(f"Error saving aggregated results: {e}")
            sys.exit(1)

    # Create a custom plotting function for aggregated results with error bars
    def plot_aggregated_results(agg_results, output_path=None):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Plot 1: Percentage of cells labelled vs threshold with min/max range
        axs[0].errorbar(
            agg_results["threshold"],
            agg_results["percent_cells_labelled"],
            yerr=[
                agg_results["percent_cells_labelled"]
                - agg_results["percent_cells_min"],
                agg_results["percent_cells_max"]
                - agg_results["percent_cells_labelled"],
            ],
            fmt="o-",
            capsize=5,
        )
        axs[0].set_xlabel("Coverage Threshold (%)")
        axs[0].set_ylabel("Cells Labelled (%) - Avg across years")
        axs[0].set_title("Percentage of Cells Labelled vs Coverage Threshold")
        axs[0].grid(True)

        # Plot 2: Spatial autocorrelation vs threshold with min/max range
        valid_indices = [
            i
            for i, row in agg_results.iterrows()
            if row["spatial_autocorr"] is not None
        ]
        if valid_indices:
            valid_thresholds = agg_results.iloc[valid_indices]["threshold"]
            morans_i = [
                row["spatial_autocorr"]["I"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]
            morans_i_min = [
                row["spatial_autocorr"]["I_min"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]
            morans_i_max = [
                row["spatial_autocorr"]["I_max"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]
            p_values = [
                row["spatial_autocorr"]["p_value"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]

            axs[1].errorbar(
                valid_thresholds,
                morans_i,
                yerr=[
                    np.array(morans_i) - np.array(morans_i_min),
                    np.array(morans_i_max) - np.array(morans_i),
                ],
                fmt="o-",
                capsize=5,
                label="Moran's I (avg)",
            )
            axs[1].set_xlabel("Coverage Threshold (%)")
            axs[1].set_ylabel("Moran's I")
            axs[1].set_title("Spatial Autocorrelation vs Coverage Threshold")
            axs[1].grid(True)

            # Add p-value annotation
            ax2 = axs[1].twinx()
            ax2.plot(valid_thresholds, p_values, "r--", label="p-value (avg)")
            ax2.set_ylabel("p-value")

            # Combine legends
            lines1, labels1 = axs[1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axs[1].legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            axs[1].text(
                0.5,
                0.5,
                "No valid spatial autocorrelation data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[1].transAxes,
            )

        # Plot 3: CI width vs threshold with min/max range
        valid_indices = [
            i
            for i, row in agg_results.iterrows()
            if row["confidence_interval"] is not None
        ]
        if valid_indices:
            valid_thresholds = agg_results.iloc[valid_indices]["threshold"]
            ci_widths = [
                row["confidence_interval"]["width"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]
            ci_min = [
                row["confidence_interval"]["width_min"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]
            ci_max = [
                row["confidence_interval"]["width_max"]
                for i, row in agg_results.iterrows()
                if i in valid_indices
            ]

            axs[2].errorbar(
                valid_thresholds,
                ci_widths,
                yerr=[
                    np.array(ci_widths) - np.array(ci_min),
                    np.array(ci_max) - np.array(ci_widths),
                ],
                fmt="o-",
                capsize=5,
            )
            axs[2].set_xlabel("Coverage Threshold (%)")
            axs[2].set_ylabel("CI Width (avg)")
            axs[2].set_title("Confidence Interval Width vs Coverage Threshold")
            axs[2].grid(True)
        else:
            axs[2].text(
                0.5,
                0.5,
                "No valid confidence interval data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[2].transAxes,
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved aggregated plot to {output_path}")

        return fig

    # Plot aggregated results
    if output_dir:
        try:
            logger.info("Generating aggregated plots...")
            agg_plot_path = os.path.join(
                output_dir, "aggregated_threshold_analysis_plots.png"
            )
            plot_aggregated_results(aggregated_results, agg_plot_path)
        except Exception as e:
            logger.error(f"Error generating aggregated plots: {e}")
            sys.exit(1)

    # Print summary statistics
    logger.info("Summary of Results:")

    for year, grid_gdf_year in year_grid_gdfs.items():
        cells_with_coverage = len(grid_gdf_year[grid_gdf_year["coverage_percent"] > 0])
        logger.info(
            f"\nYear {year}:\n"
            f"  Total grid cells: {len(grid_gdf_year)}\n"
            f"  Grid cells with any coverage: {cells_with_coverage} "
            f"({cells_with_coverage/len(grid_gdf_year)*100:.2f}%)"
        )

        for threshold in range(min_threshold, max_threshold + 1, 5):
            cells_at_threshold = len(
                grid_gdf_year[
                    (grid_gdf_year["coverage_percent"] >= threshold)
                    & grid_gdf_year["weighted_wealth"].notna()
                ]
            )
            percent = (cells_at_threshold / len(grid_gdf_year)) * 100
            logger.info(
                f"  Grid cells with ≥{threshold}% coverage: {cells_at_threshold} ({percent:.2f}%)"
            )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
