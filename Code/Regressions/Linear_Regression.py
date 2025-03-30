import os
import re
from glob import glob
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =============================================================================
# Set a Professional Plot Style
# =============================================================================
sns.set(style="whitegrid", context="talk")

# =============================================================================
# Section 1: Process Only "labelled_inferred" Files & Per-Year Regression
# =============================================================================

# Define folder path to the GPKG files
folder_path = Path(__file__).parent.parent.parent / "data" / "Results" / "LabelledGrids"
all_files = glob(os.path.join(folder_path, "*.gpkg"))

# Filter files to include only those with "labelled_inferred" in the name (case insensitive)
inferred_files = [f for f in all_files if "labelled_inferred" in f.lower()]

# Dictionaries to hold per-year data and aggregated results for the combined regression
data_by_year = {}  # Format: {year: [ {file info, model, data, means}, ... ]}
agg_results = []  # Each element is a dict: {'year': year, 'wealth_index': mean_wealth}

# Loop through each inferred file, extract year, read the file and perform regression
for file in inferred_files:
    # Extract year from the filename (expecting a pattern like 20XX)
    match = re.search(r"(20\d{2})", file)
    if not match:
        print(f"No year found in {file}. Skipping file.")
        continue
    year = int(match.group(0))

    try:
        gdf = gpd.read_file(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    # Check for required columns
    required_columns = {"coverage_percent", "weighted_wealth"}
    if not required_columns.issubset(gdf.columns):
        print(
            f"File {file} is missing required columns {required_columns}. Skipping file."
        )
        continue

    # Use only the required columns and drop any missing values
    df = gdf[["coverage_percent", "weighted_wealth"]].dropna()
    if df.empty:
        continue

    # Perform linear regression: weighted_wealth ~ coverage_percent
    X = df[["coverage_percent"]]
    y = df["weighted_wealth"]
    model = LinearRegression()
    model.fit(X, y)

    # Compute mean values for coverage and wealth in the file
    mean_coverage = df["coverage_percent"].mean()
    mean_wealth = df["weighted_wealth"].mean()

    # Save the file's data, regression model, and computed means into our dictionary
    data_by_year.setdefault(year, []).append(
        {
            "file": file,
            "data": df,
            "model": model,
            "mean_coverage": mean_coverage,
            "mean_wealth": mean_wealth,
        }
    )

    # Store the aggregated result (one point per file) for the combined regression
    agg_results.append({"year": year, "wealth_index": mean_wealth})

# =============================================================================
# Section 2: Create Per-Year Regression Plots for Inferred Files
# =============================================================================
for year, items in sorted(data_by_year.items()):
    plt.figure(figsize=(10, 6))
    plt.title(f"Labelled Inferred Regression for Year {year}")
    plt.xlabel("Coverage Percent")
    plt.ylabel("Weighted Wealth")

    # For each file in this year, plot data points and the regression line
    for item in items:
        df = item["data"]
        model = item["model"]
        # Plot scatter points (black color)
        plt.scatter(
            df["coverage_percent"],
            df["weighted_wealth"],
            color="black",
            s=5,
            alpha=0.6,
            label=os.path.basename(item["file"]).replace(".gpkg", ""),
        )
        # Create and plot the regression line (green)
        x_min, x_max = df["coverage_percent"].min(), df["coverage_percent"].max()
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = model.predict(x_vals.reshape(-1, 1))
        plt.plot(x_vals, y_vals, color="green", linewidth=2, alpha=0.7)

    plt.legend(title="Files", fontsize="small")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Section 3: Combined Regression Across Years to Predict 2024
# =============================================================================

# Create a DataFrame from the aggregated results and sort by year
agg_df = pd.DataFrame(agg_results).sort_values(by="year")

# Build the regression model: wealth_index ~ year
X_big = agg_df[["year"]]
y_big = agg_df["wealth_index"]
big_model = LinearRegression()
big_model.fit(X_big, y_big)

# Predict wealth_index for the year 2024 using the combined model
pred_2024 = big_model.predict(np.array([[2024]]))[0]
print(f"Predicted wealth_index for 2024: {pred_2024:.2f}")

# Plot the aggregated regression
plt.figure(figsize=(10, 6))
plt.title("Combined Regression Across Years (Labelled Inferred Files)")
plt.xlabel("Year")
plt.ylabel("Aggregated Weighted Wealth")

# Plot the aggregated data points in blue
plt.scatter(
    agg_df["year"], agg_df["wealth_index"], color="blue", s=50, label="Aggregated Data"
)

# Plot the regression line from the minimum year to 2024
year_range = np.linspace(agg_df["year"].min(), 2024, 100)
plt.plot(
    year_range,
    big_model.predict(year_range.reshape(-1, 1)),
    color="green",
    linewidth=3,
    label="Combined Regression",
)

# Highlight the predicted 2024 point in red
plt.scatter(2024, pred_2024, color="red", s=100, label="Prediction 2024")

plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# Section 4: Compute R² for Each File and Overall Combined Data
# =============================================================================

# Compute and print R² for each file's regression
print("Per-file R² values:")
for year, items in sorted(data_by_year.items()):
    for item in items:
        X = item["data"][["coverage_percent"]]
        y = item["data"]["weighted_wealth"]
        r2 = item["model"].score(X, y)
        print(f"File: {os.path.basename(item['file'])} (Year {year}) - R²: {r2:.3f}")

# Combine all data from different files for a global regression
all_data = pd.concat(
    [item["data"] for year in data_by_year for item in data_by_year[year]],
    ignore_index=True,
)
X_all = all_data[["coverage_percent"]]
y_all = all_data["weighted_wealth"]
combined_model = LinearRegression().fit(X_all, y_all)
overall_r2 = combined_model.score(X_all, y_all)
print(f"\nOverall R² for combined inferred data: {overall_r2:.3f}")

# =============================================================================
# Section 5: Cell-Level Predictions for 2024 Using "cell_id" & Saving Results
# =============================================================================

# Filter inferred files again for cell-level prediction
data_list = []  # List to store per-file data with cell-level info

for file in inferred_files:
    # Extract year from filename
    match = re.search(r"(20\d{2})", file)
    if not match:
        print(f"No year found in {file}. Skipping file.")
        continue
    year = int(match.group(0))

    try:
        gdf = gpd.read_file(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    # Check for required columns: cell_id and weighted_wealth
    required_columns = {"cell_id", "weighted_wealth"}
    if not required_columns.issubset(gdf.columns):
        print(
            f"File {file} is missing required columns {required_columns}. Skipping file."
        )
        continue

    # Drop missing values for the required columns and add the year column
    df = gdf[["cell_id", "weighted_wealth"]].dropna()
    if df.empty:
        continue
    df["year"] = year

    data_list.append(df)

# Combine all the cell-level data into one DataFrame
if data_list:
    combined_df = pd.concat(data_list, ignore_index=True)
else:
    raise ValueError("No data available from the inferred files.")

# Prepare a list to store predictions per cell_id
predictions = []

# Group by cell_id and perform regression if there are at least 2 time points
for cell_id, group in combined_df.groupby("cell_id"):
    if len(group) < 2:
        continue  # Skip if insufficient data for regression

    # Use year as predictor and weighted_wealth as response variable
    X = group["year"].values.reshape(-1, 1)
    y = group["weighted_wealth"].values
    model = LinearRegression().fit(X, y)
    pred_2024_cell = model.predict(np.array([[2024]]))[0]

    predictions.append(
        {"cell_id": cell_id, "predicted_weighted_wealth_2024": pred_2024_cell}
    )

# Convert predictions list to a DataFrame and save as CSV
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions_by_cell_2024.csv", index=False)

print("Predictions for 2024 by cell_id:")
print(predictions_df.head())

# Plot a histogram of the predicted values
plt.figure(figsize=(8, 6))
plt.hist(
    predictions_df["predicted_weighted_wealth_2024"],
    bins=30,
    color="skyblue",
    edgecolor="black",
)
plt.title("Histogram of Predicted Weighted Wealth for 2024 (by cell_id)")
plt.xlabel("Predicted Weighted Wealth")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("predictions_histogram_2024.png", dpi=300)
plt.show()

# =============================================================================
# Section 6: Combined Panel Figure with Multiple Subplots and CSV Preview
# =============================================================================

# Define the five selected years for plotting
selected_years = [2016, 2017, 2018, 2019, 2023]

# Create a figure with a grid: 2 rows and 5 columns
fig = plt.figure(figsize=(20, 10))
grid = fig.add_gridspec(nrows=2, ncols=5)

# Upper row: One subplot per selected year
axes_years = [fig.add_subplot(grid[0, i]) for i in range(5)]
# Lower row: Aggregated regression plot and CSV preview
ax_agg = fig.add_subplot(grid[1, 0:3])
ax_csv = fig.add_subplot(grid[1, 3:5])


def plot_selected_year(ax, year):
    """
    Plots coverage_percent vs. weighted_wealth for a specified year.
    """
    items = data_by_year.get(year, [])
    if not items:
        ax.text(0.5, 0.5, f"No data for {year}", ha="center", va="center")
        ax.set_title(str(year))
        return
    ax.set_title(f"Year {year}")
    ax.set_xlabel("Coverage Percent")
    ax.set_ylabel("Weighted Wealth")
    for item in items:
        df = item["data"]
        model = item["model"]
        # Scatter plot of the data points
        ax.scatter(
            df["coverage_percent"], df["weighted_wealth"], color="black", s=5, alpha=0.6
        )
        # Plot regression line for the file
        x_min, x_max = df["coverage_percent"].min(), df["coverage_percent"].max()
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = model.predict(x_vals.reshape(-1, 1))
        ax.plot(x_vals, y_vals, color="green", linewidth=2, alpha=0.7)


# Plot for each of the selected years
for ax, year in zip(axes_years, selected_years):
    plot_selected_year(ax, year)

# Aggregated regression plot (lower left)
ax_agg.set_title("Combined Regression Across Years")
ax_agg.set_xlabel("Year")
ax_agg.set_ylabel("Aggregated Weighted Wealth")
ax_agg.scatter(
    agg_df["year"], agg_df["wealth_index"], color="blue", s=50, label="Aggregated Data"
)
year_range = np.linspace(agg_df["year"].min(), 2024, 100)
ax_agg.plot(
    year_range,
    big_model.predict(year_range.reshape(-1, 1)),
    color="green",
    linewidth=3,
    label="Combined Regression",
)
ax_agg.scatter(2024, pred_2024, color="red", s=100, label="Prediction 2024")
ax_agg.legend()

# CSV preview plot (lower right)
ax_csv.axis("off")
csv_file = "cell_wealth_predictions_2024.csv"
try:
    df_csv = pd.read_csv(csv_file)
    sample_csv = df_csv.head(10)  # Display the first 10 rows
    csv_text = sample_csv.to_string(index=False)
except Exception as e:
    csv_text = f"Error loading CSV: {e}"

ax_csv.text(
    0.05,
    0.95,
    f"CSV Preview (Top 10 rows):\n\n{csv_text}",
    transform=ax_csv.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
filename = "combined_panel.png"
fig.savefig(filename, dpi=300)
plt.close(fig)
print(f"Saved combined panel figure as {filename}")

# =============================================================================
# Section 7: Cell-Level 2024 Prediction Comparison
# =============================================================================
"""
This section performs the following:
1. Loads cell-level predictions from "predictions_by_cell_2024.csv".
2. Loads the actual 2024 data from the specified GPKG file.
3. Merges both datasets on 'cell_id'.
4. Computes error metrics:
   - MAPE (Mean Absolute Percentage Error)
   - R² (Coefficient of Determination)
Interpretation:
- A MAPE of ~9% suggests predictions are within 9% on average.
- An R² of ~0.607 indicates about 60% of the variability is explained by the model.
"""

# Load cell-level predictions from CSV
predictions_df = pd.read_csv("predictions_by_cell_2024.csv")

# Load actual 2024 data from the GPKG file
actual_2024_file = (
    Path(__file__).parent.parent.parent
    / "data"
    / "LabelledGrids"
    / "Senegal_2024_wealthindex_labelled_inferred.gpkg"
)
gdf_2024 = gpd.read_file(actual_2024_file)

# Prepare actual data: select cell_id and weighted_wealth, then rename column
actual_df = gdf_2024[["cell_id", "weighted_wealth"]].rename(
    columns={"weighted_wealth": "weighted_wealth_actual"}
)

# Merge predictions with actual data on cell_id
merged = pd.merge(predictions_df, actual_df, on="cell_id", how="inner")

# Calculate error metrics: MAPE and R²
errors = np.abs(
    merged["weighted_wealth_actual"] - merged["predicted_weighted_wealth_2024"]
)
ape = errors / np.abs(merged["weighted_wealth_actual"])
mape = np.mean(ape) * 100
r2 = r2_score(
    merged["weighted_wealth_actual"], merged["predicted_weighted_wealth_2024"]
)

print("Cell-Level 2024 Prediction Comparison:")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.3f}")
