# preprocess.py
import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
import time
import geopandas as gpd
import pickle
import json

# Record start time
start_time = time.time()
print("Starting preprocessing script...")

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "Results" / "LabelledGrids"
SHAPEFILE_DIR = DATA_DIR / "ShapeFiles" / "Senegal"
OUTPUT_DIR = DATA_DIR / "Results" / "DashboardData"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Shapefile directory: {SHAPEFILE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


def find_gpkg_files():
    """Find all available GPKG files and extract years"""
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


def load_shapefile():
    """Load and process the Senegal shapefile"""
    try:
        # Find the shapefile
        shapefiles = list(SHAPEFILE_DIR.glob("*.shp"))

        if not shapefiles:
            print(f"Error: No shapefile found in {SHAPEFILE_DIR}")
            sys.exit(1)

        shapefile_path = shapefiles[0]
        print(f"Loading shapefile: {shapefile_path}")

        # Load and reproject if needed
        gdf = gpd.read_file(shapefile_path)

        # Check if we need to reproject
        if gdf.crs is None:
            print("Warning: Shapefile has no CRS information. Assuming EPSG:4326.")
            gdf.crs = "EPSG:4326"

        if gdf.crs != "EPSG:32628":
            gdf = gdf.to_crs("EPSG:32628")

        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {str(e)}")
        sys.exit(1)


def load_gpkg_data(file_path):
    """Load GPKG data from file"""
    try:
        if not file_path:
            return None

        # Load the file
        gdf = gpd.read_file(file_path)

        # Check if 'geom' is in the columns
        if "geom" in gdf.columns:
            # Set 'geom' as the active geometry column
            gdf = gdf.set_geometry("geom")

        # Ensure weighted_wealth exists
        if "weighted_wealth" not in gdf.columns:
            print(f"Error: No weighted_wealth column found in {file_path}")
            sys.exit(1)

        return gdf
    except Exception as e:
        print(f"Error loading GPKG data from {file_path}: {str(e)}")
        sys.exit(1)


def calculate_statistics(gdf):
    """Calculate statistics for a GeoDataFrame"""
    if gdf is None or gdf.empty:
        return None

    stats = {
        "median": float(gdf["weighted_wealth"].median()),
        "mean": float(gdf["weighted_wealth"].mean()),
        "min": float(gdf["weighted_wealth"].min()),
        "max": float(gdf["weighted_wealth"].max()),
        "std": float(gdf["weighted_wealth"].std()),
    }
    return stats


def find_global_min_max(years_and_types):
    """Find global min/max across all datasets"""
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

    return float(global_min), float(global_max)


def save_to_geojson(gdf, output_path):
    """Save GeoDataFrame to GeoJSON"""
    if gdf is None or gdf.empty:
        return

    # Ensure geometry column is properly set
    if not hasattr(gdf, "crs"):
        print(f"Warning: GeoDataFrame has no CRS. Cannot save to {output_path}")
        return

    # Convert to EPSG:4326 for GeoJSON
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Save to GeoJSON
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved GeoJSON to {output_path}")


# Main preprocessing function
def preprocess_data():
    # Step 1: Load basic data
    years_and_types, sorted_years = find_gpkg_files()
    senegal_boundary = load_shapefile()
    global_min, global_max = find_global_min_max(years_and_types)

    print(f"Found years: {sorted_years}")
    print(f"Global wealth index range: {global_min:.4f} - {global_max:.4f}")

    # Step 2: Pre-calculate statistics for all years and data types
    all_stats = {}
    for year in sorted_years:
        all_stats[year] = {}
        for type_key in ["original", "inferred"]:
            file_path = years_and_types[year][type_key]
            if file_path:
                data = load_gpkg_data(file_path)
                all_stats[year][type_key] = calculate_statistics(data)

    # Step 3: Prepare data for the time series chart
    chart_data = {
        "Year": [],
        "Original Data": [],
        "Satellite Image Predicted Data": [],
    }

    for year in sorted_years:
        # Convert year to integer for proper plotting
        chart_data["Year"].append(int(year))

        if "original" in all_stats[year] and all_stats[year]["original"]:
            chart_data["Original Data"].append(all_stats[year]["original"]["median"])
        else:
            chart_data["Original Data"].append(None)

        if "inferred" in all_stats[year] and all_stats[year]["inferred"]:
            chart_data["Satellite Image Predicted Data"].append(
                all_stats[year]["inferred"]["median"]
            )
        else:
            chart_data["Satellite Image Predicted Data"].append(None)

    # Step 4: PRE-PROCESS ALL GEOSPATIAL DATA
    print("Pre-processing all geospatial data...")
    processed_data = {}

    for year in sorted_years:
        processed_data[year] = {}

        for data_type in ["original", "inferred"]:
            file_path = years_and_types[year][data_type]
            if not file_path:
                processed_data[year][data_type] = None
                continue

            print(f"Processing {year} - {data_type}...")

            # Load data
            gdf = load_gpkg_data(file_path)
            if gdf is None or gdf.empty:
                processed_data[year][data_type] = None
                continue

            # Clip to boundary
            try:
                clipped_gdf = gpd.overlay(gdf, senegal_boundary, how="intersection")

                if clipped_gdf.empty:
                    print(
                        f"Warning: Clipping resulted in empty GeoDataFrame for {year} - {data_type}"
                    )
                    plot_gdf = gdf.copy()
                else:
                    plot_gdf = clipped_gdf

                # Convert to lat/lon for visualization
                if plot_gdf.crs != "EPSG:4326":
                    plot_gdf_4326 = plot_gdf.to_crs("EPSG:4326")
                else:
                    plot_gdf_4326 = plot_gdf.copy()

                # Calculate centroids in a projected CRS to avoid the warning
                temp_gdf = plot_gdf.to_crs("EPSG:32628")
                centroids = temp_gdf.geometry.centroid
                centroids_4326 = centroids.to_crs("EPSG:4326")
                center_lat = float(centroids_4326.y.mean())
                center_lon = float(centroids_4326.x.mean())

                # Save GeoJSON file
                geojson_path = OUTPUT_DIR / f"{year}_{data_type}.geojson"
                save_to_geojson(plot_gdf_4326, geojson_path)

                # Store metadata
                processed_data[year][data_type] = {
                    "geojson_path": str(geojson_path),
                    "center": {"lat": center_lat, "lon": center_lon},
                }

            except Exception as e:
                print(f"Error processing {year} - {data_type}: {str(e)}")
                processed_data[year][data_type] = None

    # Step 5: Save Senegal boundary
    boundary_path = OUTPUT_DIR / "senegal_boundary.geojson"
    save_to_geojson(senegal_boundary.to_crs("EPSG:4326"), boundary_path)

    # Step 6: Save all metadata and processed data
    metadata = {
        "years": sorted_years,
        "global_min": global_min,
        "global_max": global_max,
        "stats": all_stats,
        "chart_data": chart_data,
        "processed_data": processed_data,
        "boundary_path": str(boundary_path),
    }

    # Save metadata to JSON
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    preprocess_data()
