import os
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import matplotlib.pyplot as plt
from esda.moran import Moran
from libpysal.weights import Queen
import scipy.stats as stats
from tqdm import tqdm
from src.utils.logging_config import setup_logging
import sys


logger = setup_logging("INFO")
for handler in logger.handlers:
    handler.flush = sys.stdout.flush


def load_grid_cells(grid_gpkg_path):
    """Load grid cells from existing geopackage."""
    try:
        grid_gdf = gpd.read_file(grid_gpkg_path)
        logger.info(f"Loaded {len(grid_gdf)} grid cells with CRS: {grid_gdf.crs}")
        return grid_gdf
    except Exception as e:
        logger.error(f"Error loading grid cells: {e}")
        raise


def load_wealth_data(wealth_gpkg_path):
    """Load wealth index data from GPKG file."""
    try:
        wealth_gdf = gpd.read_file(wealth_gpkg_path)
        logger.info(
            f"Loaded {len(wealth_gdf)} wealth data points with CRS: {wealth_gdf.crs}"
        )
        return wealth_gdf
    except Exception as e:
        logger.error(f"Error loading wealth data: {e}")
        raise


def calculate_coverage(grid_gdf, wealth_gdf):
    """calculate_coverage Calculate the coverage percentage for each grid cell.

    Args:
        grid_gdf: grid of all the cells of a country with no labels
        wealth_gdf: the calculated wealth indices with their geometry

    Returns:
        GeoDataFrame: grid_gdf for the country now labeled with the wealth index
    """
    # Ensure both GDFs have the same CRS for intersection area
    if grid_gdf.crs != wealth_gdf.crs:
        logger.info(f"Converting CRS: {grid_gdf.crs} to {wealth_gdf.crs}")
        grid_gdf = grid_gdf.to_crs(wealth_gdf.crs)

    # Calculate area of each grid cell
    grid_gdf["total_area"] = grid_gdf.geometry.area

    # Initialize coverage columns
    grid_gdf["covered_area"] = 0.0
    grid_gdf["coverage_percent"] = 0.0
    grid_gdf["weighted_wealth"] = np.nan

    # For each grid cell, calculate intersection with wealth data
    for idx, grid_cell in tqdm(
        grid_gdf.iterrows(), total=len(grid_gdf), desc="Calculating coverage"
    ):
        # Find intersecting wealth cells
        try:
            intersections = wealth_gdf[wealth_gdf.intersects(grid_cell.geometry)]

            if len(intersections) == 0:
                continue

            # Calculate intersection areas and weights
            intersection_areas = []
            weights = []
            wealth_values = []

            for _, wealth_cell in intersections.iterrows():
                intersection = grid_cell.geometry.intersection(wealth_cell.geometry)
                area = intersection.area
                intersection_areas.append(area)
                weights.append(area)
                wealth_values.append(wealth_cell["wealth_index"])

            # Store results
            grid_gdf.at[idx, "covered_area"] = sum(intersection_areas)
            grid_gdf.at[idx, "coverage_percent"] = (
                sum(intersection_areas) / grid_cell.total_area
            ) * 100

            # Calculate weighted wealth index for every cell
            if sum(weights) > 0:
                weighted_wealth = sum(
                    weights[i] * wealth_values[i] for i in range(len(weights))
                ) / sum(weights)
                grid_gdf.at[idx, "weighted_wealth"] = weighted_wealth

        except Exception as e:
            logger.error(f"Error processing grid cell {idx}: {e}")
            continue

    return grid_gdf


def analyze_thresholds(grid_gdf, min_threshold, max_threshold, steps=30):
    """analyze_thresholds calculates the autocorrelation, threshold, and confidence for the different thresholds

    Args:
        grid_gdf: labeled grid for a country that includes wealth index for every cell
        min_threshold: smallest threshold to test
        max_threshold: largest threshold to test
        steps: number of steps to do between min and max threshold. Defaults to 30.

    Returns:
        DataFrame: DataFrame including all the statistics calculated
    """
    results = []
    threshold_range = np.linspace(min_threshold, max_threshold, steps).tolist()

    # Looping through thresholds
    for threshold in threshold_range:
        # Filter cells that meet the threshold
        filtered_cells = grid_gdf[
            (grid_gdf["coverage_percent"] >= threshold)
            & grid_gdf["weighted_wealth"].notna()
        ].copy()

        # Calculate percentage of cells that meet this threshold
        percent_cells = (len(filtered_cells) / len(grid_gdf)) * 100

        # Calculate spatial autocorrelation if we have enough cells
        spatial_autocorr = None
        if len(filtered_cells) > 1:
            try:
                # Create spatial weights matrix
                w = Queen.from_dataframe(filtered_cells)
                # Check if weights are valid
                if w.n > 0 and w.s0 > 0:
                    # Calculate Moran's I
                    moran = Moran(filtered_cells["weighted_wealth"], w)
                    spatial_autocorr = {"I": moran.I, "p_value": moran.p_sim}
                else:
                    logger.warning(
                        f"Warning: Invalid weights for threshold {threshold}. Skipping Moran's I calculation."
                    )
            except Exception as e:
                logger.error(
                    f"Error calculating spatial autocorrelation for threshold {threshold}: {e}"
                )

        # Calculate confidence intervals for the wealth index for each threshold
        confidence_interval = None
        if len(filtered_cells) > 1:
            try:
                mean = filtered_cells["weighted_wealth"].mean()
                std = filtered_cells["weighted_wealth"].std()
                n = len(filtered_cells)

                if n > 1 and std > 0:
                    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
                    confidence_interval = {
                        "lower": ci[0],
                        "upper": ci[1],
                        "width": ci[1] - ci[0],
                    }
            except Exception as e:
                logger.error(
                    f"Error calculating confidence interval for threshold {threshold}: {e}"
                )

        results.append(
            {
                "threshold": threshold,
                "percent_cells_labelled": percent_cells,
                "num_cells_labelled": len(filtered_cells),
                "spatial_autocorr": spatial_autocorr,
                "confidence_interval": confidence_interval,
            }
        )

    return pd.DataFrame(results)


def plot_results(results_df, output_path=None):
    """Plot the analysis results."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Percentage of cells labelled vs threshold
    axs[0].plot(results_df["threshold"], results_df["percent_cells_labelled"], "o-")
    axs[0].set_xlabel("Coverage Threshold (%)")
    axs[0].set_ylabel("Cells Labelled (%)")
    axs[0].set_title("Percentage of Cells Labelled vs Coverage Threshold")
    axs[0].grid(True)

    # Plot 2: Spatial autocorrelation vs threshold
    morans_i = []
    p_values = []
    valid_thresholds = []

    for idx, row in results_df.iterrows():
        if row["spatial_autocorr"] is not None:
            morans_i.append(row["spatial_autocorr"]["I"])
            p_values.append(row["spatial_autocorr"]["p_value"])
            valid_thresholds.append(row["threshold"])

    if len(valid_thresholds) > 0:
        axs[1].plot(valid_thresholds, morans_i, "o-", label="Moran's I")
        axs[1].set_xlabel("Coverage Threshold (%)")
        axs[1].set_ylabel("Moran's I")
        axs[1].set_title("Spatial Autocorrelation vs Coverage Threshold")
        axs[1].grid(True)

        # Add p-value annotation
        ax2 = axs[1].twinx()
        ax2.plot(valid_thresholds, p_values, "r--", label="p-value")
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

    # Plot 3: Confidence interval width vs threshold
    ci_widths = []
    ci_thresholds = []

    for idx, row in results_df.iterrows():
        if row["confidence_interval"] is not None:
            ci_widths.append(row["confidence_interval"]["width"])
            ci_thresholds.append(row["threshold"])

    if len(ci_thresholds) > 0:
        axs[2].plot(ci_thresholds, ci_widths, "o-")
        axs[2].set_xlabel("Coverage Threshold (%)")
        axs[2].set_ylabel("CI Width")
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
        logger.info(f"Saved plot to {output_path}")

    return fig
