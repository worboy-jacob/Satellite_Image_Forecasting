import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import geopandas as gpd
from PIL import Image

# Define paths - use absolute path based on script location
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "Results" / "LabelledGrids"
SHAPEFILE_DIR = DATA_DIR / "ShapeFiles" / "Senegal"
OUTPUT_DIR = DATA_DIR / "Results" / "WealthMaps"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Define the target CRS for all spatial data
TARGET_CRS = "EPSG:32628"


# Function to find all available gpkg files and extract years
def find_gpkg_files():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    all_files = list(RESULTS_DIR.glob("*.gpkg"))

    if not all_files:
        print(f"Error: No GPKG files found in {RESULTS_DIR}")
        sys.exit(1)

    # Extract years and types from filenames
    years_and_types = {}
    pattern = re.compile(r"Senegal_(\d{4})_wealthindex_labelled(_inferred)?")

    for file_path in all_files:
        filename = file_path.name
        match = pattern.search(filename)
        if match:
            year = match.group(1)
            is_inferred = bool(match.group(2))

            if year not in years_and_types:
                years_and_types[year] = {"original": None, "inferred": None}

            if is_inferred:
                years_and_types[year]["inferred"] = file_path
            else:
                years_and_types[year]["original"] = file_path

    # Sort years
    sorted_years = sorted(years_and_types.keys())

    return years_and_types, sorted_years


# Function to load shapefile
def load_shapefile():
    try:
        # Find the shapefile
        shapefiles = list(SHAPEFILE_DIR.glob("*.shp"))

        if not shapefiles:
            print(f"Error: No shapefile found in {SHAPEFILE_DIR}")
            sys.exit(1)

        shapefile_path = shapefiles[0]
        print(f"Loading shapefile: {shapefile_path}")

        # Load and explicitly set CRS to TARGET_CRS
        gdf = gpd.read_file(shapefile_path)

        # Explicitly set the CRS to TARGET_CRS
        gdf = gdf.set_crs("EPSG:4326", allow_override=False)
        gdf = gdf.to_crs(TARGET_CRS)

        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {str(e)}")
        sys.exit(1)


# Function to load gpkg data
def load_gpkg_data(file_path):
    try:
        if not file_path:
            return None

        # Load the file
        gdf = gpd.read_file(file_path)

        # Check if 'geom' is in the columns
        if "geom" in gdf.columns:
            # Set 'geom' as the active geometry column
            print("Setting 'geom' as active geometry column")
            gdf = gdf.set_geometry("geom")

        # Ensure weighted_wealth exists
        if "weighted_wealth" not in gdf.columns:
            print(f"Error: No weighted_wealth column found in {file_path}")
            sys.exit(1)

        # Explicitly set the CRS to TARGET_CRS
        gdf = gdf.set_crs(TARGET_CRS, allow_override=True)

        # Check for empty or invalid geometries
        invalid_geoms = gdf[~gdf.is_valid]
        if len(invalid_geoms) > 0:
            print(f"Warning: {len(invalid_geoms)} invalid geometries found")
            # We could try to fix them: gdf = gdf.buffer(0)

        empty_geoms = gdf[gdf.is_empty]
        if len(empty_geoms) > 0:
            print(f"Warning: {len(empty_geoms)} empty geometries found")
            # Remove empty geometries
            gdf = gdf[~gdf.is_empty]

        # Verify we still have data
        if len(gdf) == 0:
            print("Error: No valid geometries found after cleaning")
            return None

        return gdf
    except Exception as e:
        print(f"Error loading GPKG data from {file_path}: {str(e)}")
        sys.exit(1)


# Function to calculate statistics
def calculate_statistics(gdf):
    if gdf is None or gdf.empty:
        return None

    stats = {
        "median": gdf["weighted_wealth"].median(),
        "mean": gdf["weighted_wealth"].mean(),
        "min": gdf["weighted_wealth"].min(),
        "max": gdf["weighted_wealth"].max(),
        "std": gdf["weighted_wealth"].std(),
    }
    return stats


# Function to find global min/max across all datasets
def find_global_min_max(years_and_types):
    global_min = float("inf")
    global_max = float("-inf")

    for year_data in years_and_types.values():
        for data_type in ["original", "inferred"]:
            if year_data[data_type]:
                gdf = load_gpkg_data(year_data[data_type])
                if gdf is not None and not gdf.empty:
                    current_min = gdf["weighted_wealth"].min()
                    current_max = gdf["weighted_wealth"].max()

                    global_min = min(global_min, current_min)
                    global_max = max(global_max, current_max)

    # If no data was found
    if global_min == float("inf"):
        global_min = 0
    if global_max == float("-inf"):
        global_max = 1

    return global_min, global_max


# Function to generate a map for a specific year and data type
def generate_map(
    gdf, year, data_type, global_min, global_max, senegal_boundary, output_path=None
):
    if gdf is None or gdf.empty:
        print(f"No data available for {year} - {data_type}")
        return None

    print(f"Generating map for {year} - {data_type}")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the title
    data_type_label = (
        "Original Data" if data_type == "original" else "Satellite Image Predicted Data"
    )
    plt.title(f"Senegal Wealth Index - {year} - {data_type_label}", fontsize=14)

    # First, plot the data with a colormap
    # Create a diverging colormap from blue to white to red
    cmap = plt.cm.RdBu_r

    # Plot the wealth data
    gdf.plot(
        column="weighted_wealth",
        cmap=cmap,
        linewidth=0.2,
        ax=ax,
        edgecolor="gray",
        alpha=0.8,  # Increased alpha for better visibility
        vmin=global_min,
        vmax=global_max,
    )

    # Then plot the Senegal boundary on top
    senegal_boundary.boundary.plot(ax=ax, color="black", linewidth=1.0)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=global_min, vmax=global_max)
    )
    sm._A = []  # Empty array for the data range
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Wealth Index")

    # Calculate statistics
    stats = calculate_statistics(gdf)

    # Add statistics to the plot
    if stats:
        stats_text = (
            f"Statistics:\n"
            f"Median: {stats['median']:.4f}\n"
            f"Mean: {stats['mean']:.4f}\n"
            f"Std Dev: {stats['std']:.4f}\n"
            f"Min: {stats['min']:.4f}\n"
            f"Max: {stats['max']:.4f}"
        )
        plt.figtext(
            0.15,
            0.02,
            stats_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved map to {output_path}")

    plt.close(fig)
    return output_path


# Function to create composite image
def create_composite_image(image_paths, output_path, title):
    if not image_paths:
        print(f"No images to create composite for {title}")
        return

    # Determine grid dimensions
    n_images = len(image_paths)
    n_cols = min(3, n_images)  # Maximum 3 columns
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division

    # Create a figure with subplots
    fig = plt.figure(figsize=(n_cols * 7, n_rows * 8))
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # Add the title to the figure
    fig.suptitle(title, fontsize=16, y=0.98)

    # Add each image to the grid
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue

        row = i // n_cols
        col = i % n_cols

        # Open the image
        img = plt.imread(img_path)

        # Create subplot
        ax = fig.add_subplot(gs[row, col])

        # Extract year from filename
        year = os.path.basename(img_path).split("_")[1]

        # Set title for this subplot
        ax.set_title(f"Year: {year}", fontsize=12)

        # Display the image
        ax.imshow(img)
        ax.axis("off")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the title

    # Save the composite image
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved composite image to {output_path}")
    plt.close(fig)


def main():
    print("Loading data...")
    years_and_types, sorted_years = find_gpkg_files()
    senegal_boundary = load_shapefile()

    global_min, global_max = find_global_min_max(years_and_types)

    print(f"Found years: {sorted_years}")
    print(f"Global wealth index range: {global_min:.4f} - {global_max:.4f}")

    # Lists to store paths of generated images
    original_images = []
    inferred_images = []

    # Generate maps for each year and data type
    for year in sorted_years:
        for data_type in ["original", "inferred"]:
            file_path = years_and_types[year][data_type]
            if file_path:
                print(f"Processing {year} - {data_type}...")
                gdf = load_gpkg_data(file_path)

                if gdf is None:
                    print(f"Skipping {year} - {data_type} due to missing data")
                    continue

                # Define output path
                output_path = OUTPUT_DIR / f"senegal_wealth_{year}_{data_type}.png"

                # Generate the map
                image_path = generate_map(
                    gdf,
                    year,
                    data_type,
                    global_min,
                    global_max,
                    senegal_boundary,
                    output_path,
                )

                # Add to appropriate list
                if image_path:
                    if data_type == "original":
                        original_images.append(str(output_path))
                    else:
                        inferred_images.append(str(output_path))

    # Create composite images
    if original_images:
        create_composite_image(
            original_images,
            OUTPUT_DIR / "composite_original_wealth_maps.png",
            "Senegal Wealth Index - Original Survey Data",
        )

    if inferred_images:
        create_composite_image(
            inferred_images,
            OUTPUT_DIR / "composite_inferred_wealth_maps.png",
            "Senegal Wealth Index - Satellite Image Predicted Data",
        )

    print("Image generation complete.")


if __name__ == "__main__":
    main()
