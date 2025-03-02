# src/main.py
import logging
from pathlib import Path
import psutil
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logging_config import setup_logging
from src.utils.config import Config
from src.data_processing.wealth_processor import WealthProcessor
from src.core.grid_processor import GridProcessor
from src.visualization.plots import WealthMapVisualizer
from src.utils.paths import get_config_path, get_results_dir
from src.utils.data_loader import load_all_data
import sys
import pandas as pd

###TODO: move some of this code to separate files


def cleanup_processes():
    """Kill any potential leftover Python processes from previous runs."""
    current_process = psutil.Process(os.getpid())
    current_pid = current_process.pid

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Check if it's a Python process but not our current process
            if (
                proc.info["name"] == "python" or proc.info["name"] == "python3"
            ) and proc.pid != current_pid:
                cmdline = proc.info["cmdline"]
                if cmdline and any("grid_processor.py" in arg for arg in cmdline):
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def create_output_directories():
    """Create necessary output directories if they don't exist."""
    # Create main results directory
    results_dir = get_results_dir()
    results_dir.mkdir(exist_ok=True, parents=True)

    # Create WealthMap directory for images
    wealth_map_dir = results_dir / "GPSResults" / "WealthMap"
    wealth_map_dir.mkdir(exist_ok=True, parents=True)

    # Create WealthGPS directory for data files
    wealth_gps_dir = results_dir / "GPSResults" / "WealthGPS"
    wealth_gps_dir.mkdir(exist_ok=True, parents=True)

    return wealth_map_dir, wealth_gps_dir


def main():
    try:
        # Initialize configuration
        cleanup_processes()
        config_path = get_config_path() / "config.yaml"
        config = Config(config_path).config
        logger = setup_logging(log_level=config.get("log_level", "INFO"))
        logger.info("Starting wealth index mapping")

        # Create output directories
        wealth_map_dir, wealth_gps_dir = create_output_directories()

        # Load all country-year data
        country_year_data = load_all_data(config)
        default_crs = config.get("default_crs")

        # Initialize processors
        wealth_processor = WealthProcessor(config)
        grid_processor = GridProcessor(config)

        # Initialize a single visualizer instance to be used for all maps
        visualizer = WealthMapVisualizer(
            config=config,
            figsize=config.get("visualization", {}).get("figsize", [15, 15]),
            dpi=config.get("visualization", {}).get("dpi", 300),
        )

        # Lists to store all results and metadata for combined visualization
        all_grids = []  # Store all processed grids
        all_boundaries = {}  # Store boundaries by country code
        grid_info = []  # Store metadata about each grid

        # Process each country-year pair
        for country_iso2, country_data in country_year_data.items():
            logger.info(f"Processing country: {country_iso2}")
            country_results = {}

            # Convert all boundary data to default_crs
            for year in country_data:
                country_data[year]["boundary"].to_crs(default_crs, inplace=True)
                # Store boundary for visualization
                if country_iso2 not in all_boundaries:
                    all_boundaries[country_iso2] = country_data[year]["boundary"]

            # Process each year for this country
            for year, data_package in country_data.items():
                logger.info(f"Processing year: {year}")

                # Extract data for this country-year pair
                boundary_data = data_package["boundary"]
                gps_data = data_package["gps"]
                wealth_data = data_package["wealth"]
                country_crs = data_package["crs"]

                try:
                    # Process wealth data
                    logger.info(f"Processing wealth data for {country_iso2}-{year}")
                    processed_wealth = wealth_processor.process(
                        gps_data.copy(),
                        wealth_data.copy(),
                        default_crs=default_crs,
                        country_crs=country_crs,
                    )

                    # Process grid
                    logger.info(f"Processing grid for {country_iso2}-{year}")
                    final_grid = grid_processor.process(
                        boundary_data.copy(),
                        processed_wealth,
                        country_crs=country_crs,
                        default_crs=default_crs,
                    )

                    # Add country and year columns to the grid
                    final_grid["country_iso2"] = country_iso2
                    final_grid["year"] = year

                    # Store results for this country-year pair
                    country_results[year] = final_grid

                    # Store for combined visualization
                    all_grids.append(final_grid)
                    grid_info.append({"country_iso2": country_iso2, "year": year})

                    # Save individual result to WealthGPS directory
                    output_filename = f"{country_iso2}_{year}_output.gpkg"
                    output_path = wealth_gps_dir / output_filename
                    final_grid.to_file(output_path, driver="GPKG")
                    logger.info(
                        f"Results for {country_iso2}-{year} saved to {output_path}"
                    )

                    # Create visualization for this country-year pair
                    title = f"Wealth Distribution - {country_iso2} ({year})"
                    visualizer.create_map(final_grid, boundary_data, title=title)
                    vis_filename = f"{country_iso2}_{year}_wealth_map.png"
                    visualizer.save_map(wealth_map_dir / vis_filename)

                except Exception as e:
                    logger.error(f"Error processing {country_iso2}-{year}: {str(e)}")
                    # Continue with next year even if this one fails
                    continue

        # Create combined data file
        if all_grids:
            # Save combined result to WealthGPS directory
            combined_grid = gpd.GeoDataFrame(pd.concat(all_grids, ignore_index=True))
            combined_output_filename = "all_countries_all_years.gpkg"
            combined_output_path = wealth_gps_dir / combined_output_filename
            combined_grid.to_file(combined_output_path, driver="GPKG")
            logger.info(f"Combined results saved to {combined_output_path}")

            # Create combined visualization with subplots
            logger.info("Creating combined visualization with all country-year pairs")

            # Determine grid layout for subplots
            total_plots = len(all_grids)
            if total_plots <= 3:
                n_cols = total_plots
                n_rows = 1
            else:
                n_cols = min(3, total_plots)  # Max 3 columns
                n_rows = (total_plots + n_cols - 1) // n_cols  # Ceiling division

            # Create figure for combined plot
            fig_size = config.get("visualization", {}).get("figsize", [15, 15])
            # Scale figsize based on number of subplots
            combined_figsize = (fig_size[0], fig_size[1] * n_rows / 2)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=combined_figsize, constrained_layout=True
            )

            # Handle case where axes is not an array (single subplot)
            if total_plots == 1:
                axes = np.array([axes])

            # Flatten axes array for easy iteration
            if total_plots > 1:
                axes = axes.flatten()

            # Calculate global min and max for standardized colormap
            global_min = min(grid_df["wealth_index"].min() for grid_df in all_grids)
            global_max = max(grid_df["wealth_index"].max() for grid_df in all_grids)
            norm = plt.Normalize(global_min, global_max)

            # Create individual plots in each subplot
            for i, (grid_df, info) in enumerate(zip(all_grids, grid_info)):
                if i < total_plots:  # Safety check
                    country_iso2 = info["country_iso2"]
                    year = info["year"]
                    boundary = all_boundaries[country_iso2]

                    ax = axes[i]

                    # Plot boundary
                    boundary.boundary.plot(ax=ax, color="black", linewidth=1)

                    # Plot wealth data with standardized scale
                    grid_df.plot(
                        column="wealth_index",
                        ax=ax,
                        legend=True,
                        legend_kwds={"label": "Wealth Index", "shrink": 0.7},
                        cmap="RdYlBu_r",
                        norm=norm,  # Use the standardized normalization
                    )

                    ax.set_title(f"{country_iso2} ({year})")
                    ax.axis("off")

            # Hide any unused subplots
            if total_plots > 1:  # Only needed if axes is an array
                for j in range(total_plots, len(axes)):
                    axes[j].set_visible(False)

            # Add overall title
            fig.suptitle(
                "Wealth Distribution Across Countries and Years", fontsize=16, y=1.02
            )

            # Save the combined figure
            combined_vis_path = wealth_map_dir / "combined_wealth_maps.png"
            fig.savefig(
                combined_vis_path,
                dpi=config.get("visualization", {}).get("dpi", 300),
                bbox_inches="tight",
            )
            plt.close(fig)

            logger.info(f"Combined visualization saved to {combined_vis_path}")

        logger.info("Wealth mapping calculation completed successfully")

    except Exception as e:
        logger.error(f"Error in wealth mapping calculation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
