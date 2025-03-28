# main.py
import os
import sys
import time
import json
from pathlib import Path

# Record start time
start_time = time.time()
print("Starting dashboard application...")

# Dash and Plotly imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

import webbrowser
import threading
import time

# Geospatial imports for visualization
import geopandas as gpd
import pandas as pd

# Set the app theme - using a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

# Define paths - use absolute path based on script location
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DASHBOARD_DATA_DIR = DATA_DIR / "Results" / "DashboardData"

# Check if preprocessed data exists
metadata_path = DASHBOARD_DATA_DIR / "metadata.json"
if not metadata_path.exists():
    print(f"Error: Preprocessed data not found at {metadata_path}")
    print("Please run the preprocessing script first.")
    sys.exit(1)

# Load metadata
print(f"Loading metadata from {metadata_path}")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Extract data from metadata
sorted_years = metadata["years"]
global_min = metadata["global_min"]
global_max = metadata["global_max"]
all_stats = metadata["stats"]
chart_data = metadata["chart_data"]
processed_data = metadata["processed_data"]
boundary_path = metadata["boundary_path"]

print(f"Found years: {sorted_years}")
print(f"Global wealth index range: {global_min:.4f} - {global_max:.4f}")

# Define the app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Senegal Wealth Index Dashboard", className="text-center my-4"),
                width=12,
            )
        ),
        dbc.Row(
            [
                # Left column for controls
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Controls", className="card-title"),
                                    html.Hr(),
                                    html.P("Select Year:"),
                                    dcc.Slider(
                                        id="year-slider",
                                        min=0,
                                        max=len(sorted_years) - 1,
                                        step=1,
                                        value=0,
                                        marks={
                                            i: year
                                            for i, year in enumerate(sorted_years)
                                        },
                                    ),
                                    html.Br(),
                                    html.P("Data Type:"),
                                    dbc.RadioItems(
                                        id="data-type-radio",
                                        options=[
                                            {
                                                "label": "Original Data",
                                                "value": "original",
                                            },
                                            {
                                                "label": "Satellite Image Predicted Data",
                                                "value": "inferred",
                                            },
                                        ],
                                        value="original",
                                        inline=True,
                                    ),
                                    html.Br(),
                                    html.Div(id="stats-output"),
                                ]
                            ),
                            className="mb-4",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Median Wealth Over Time",
                                        className="card-title",
                                    ),
                                    dcc.Graph(id="time-series-chart"),
                                ]
                            )
                        ),
                    ],
                    md=4,
                ),
                # Right column for map
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Wealth Index Map", className="card-title"),
                                dcc.Graph(id="wealth-map", style={"height": "70vh"}),
                            ]
                        )
                    ),
                    md=8,
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                html.P(
                    f"Satellite Image Predicted Data is based on machine learning models trained on the original data. ",
                    className="text-muted text-center mt-4",
                ),
                width=12,
            )
        ),
    ],
    fluid=True,
    className="dbc",
)


# Update just the map creation part in the callback function
@app.callback(
    [
        Output("wealth-map", "figure"),
        Output("stats-output", "children"),
        Output("time-series-chart", "figure"),
    ],
    [Input("year-slider", "value"), Input("data-type-radio", "value")],
)
def update_map(year_index, data_type):
    callback_start = time.time()

    # Get the selected year
    selected_year = sorted_years[year_index]
    print(f"Updating display for {selected_year} - {data_type}")

    # Get pre-processed data metadata
    data_package = processed_data[selected_year][data_type]

    # Create the map figure
    if data_package is not None:
        try:
            # Load the GeoJSON file
            geojson_path = data_package["geojson_path"]
            center = data_package["center"]

            # Load GeoDataFrame from GeoJSON
            gdf = gpd.read_file(geojson_path)

            # Load the Senegal boundary for map borders
            boundary_gdf = gpd.read_file(boundary_path)

            # Create a continuous color scale from blue to white to red
            colorscale = [
                [0, "blue"],
                [0.5, "white"],
                [1, "red"],
            ]

            # Create a base figure
            fig = go.Figure()

            # Add wealth index choropleth (without hover)
            fig.add_choroplethmapbox(
                geojson=gdf.__geo_interface__,
                locations=gdf.index,
                z=gdf["weighted_wealth"],
                colorscale=colorscale,
                zmin=global_min,
                zmax=global_max,
                marker_opacity=0.7,
                marker_line_width=0,
                colorbar=dict(title="Wealth Index", tickformat=".2f"),
                hoverinfo="none",  # Disable hover on the choropleth
            )

            # Add Senegal boundary line
            fig.add_choroplethmapbox(
                geojson=boundary_gdf.__geo_interface__,
                locations=boundary_gdf.index,
                z=[1] * len(boundary_gdf),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_color="black",
                marker_line_width=2,
                showscale=False,
                hoverinfo="none",
            )

            # Update layout
            fig.update_layout(
                mapbox_style="carto-darkmatter",
                mapbox=dict(center=dict(lat=center["lat"], lon=center["lon"]), zoom=6),
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                # More explicit hover settings
                hovermode="closest",
                hoverdistance=2,  # Increase hover detection distance
                uirevision="constant",  # Maintain state when figure updates
            )

            # Add major cities in Senegal
            cities = {
                "Dakar": [14.7167, -17.4677],
                "Thiès": [14.7833, -16.9167],
                "Kaolack": [14.1333, -16.0667],
                "Saint-Louis": [16.0333, -16.5000],
                "Ziguinchor": [12.5833, -16.2667],
            }

            for city, coords in cities.items():
                fig.add_scattermapbox(
                    lat=[coords[0]],
                    lon=[coords[1]],
                    mode="markers+text",
                    marker=dict(size=8, color="white"),
                    text=[city],
                    textposition="top right",
                    hoverinfo="text",
                    showlegend=False,
                )
        except Exception as e:
            print(f"Error creating map: {str(e)}")
            # Create empty figure if error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating map: {str(e)}",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
    else:
        # Create empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data available for this selection",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )

    # [Rest of the callback function remains the same]

    # Create statistics display
    data_type_label = (
        "Original Data" if data_type == "original" else "Satellite Image Predicted Data"
    )
    current_stats = all_stats[selected_year][data_type]

    if current_stats:
        stats_display = [
            html.H5(f"Statistics for {selected_year} - {data_type_label}"),
            html.Table(
                [
                    html.Tr(
                        [html.Td("Median:"), html.Td(f"{current_stats['median']:.4f}")]
                    ),
                    html.Tr(
                        [html.Td("Mean:"), html.Td(f"{current_stats['mean']:.4f}")]
                    ),
                    html.Tr(
                        [html.Td("Std Dev:"), html.Td(f"{current_stats['std']:.4f}")]
                    ),
                    html.Tr([html.Td("Min:"), html.Td(f"{current_stats['min']:.4f}")]),
                    html.Tr([html.Td("Max:"), html.Td(f"{current_stats['max']:.4f}")]),
                ],
                className="table table-sm",
            ),
        ]
    else:
        stats_display = html.P(
            f"No statistics available for {selected_year} - {data_type_label}"
        )

    # Create time series chart
    time_series_df = pd.DataFrame(
        {
            "Year": chart_data["Year"],
            "Original Data": chart_data["Original Data"],
            "Satellite Image Predicted Data": chart_data[
                "Satellite Image Predicted Data"
            ],
        }
    )

    time_fig = go.Figure()

    # Add original data line
    time_fig.add_trace(
        go.Scatter(
            x=time_series_df["Year"],
            y=time_series_df["Original Data"],
            mode="lines+markers",
            name="Original Data",
            line=dict(color="#3366CC"),
        )
    )

    # Add inferred data line
    time_fig.add_trace(
        go.Scatter(
            x=time_series_df["Year"],
            y=time_series_df["Satellite Image Predicted Data"],
            mode="lines+markers",
            name="Satellite Image Predicted Data",
            line=dict(color="#DC3912"),
        )
    )

    # Highlight the selected year
    time_fig.add_vline(
        x=int(selected_year), line_width=2, line_dash="dash", line_color="white"
    )

    # Update layout
    time_fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Median Wealth Index",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin={"r": 10, "t": 30, "l": 10, "b": 10},
    )

    print(f"Display updated in {time.time() - callback_start:.2f} seconds")

    return fig, stats_display, time_fig


if __name__ == "__main__":
    print(f"Dashboard loaded in {time.time() - start_time:.2f} seconds")
    print("Starting Dash app...")

    # Function to open browser after a short delay
    def open_browser():
        time.sleep(1)  # Give the server a second to start
        webbrowser.open_new("http://127.0.0.1:8050")

    # Start browser in a new thread
    threading.Thread(target=open_browser).start()

    # Run the app
    app.run(debug=False)
