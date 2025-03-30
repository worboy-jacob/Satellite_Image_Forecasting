# WealthMapping

A Python-based spatial analysis tool for mapping wealth distribution across geographic areas using survey data. This tool processes demographic and health survey data to create detailed geospatial visualizations of wealth indices.

## Overview

The Wealth Mapping tool combines household survey data with GPS coordinates to create high-resolution maps of wealth distribution. It processes DHS (Demographic and Health Survey) data with corresponding GPS cluster points to generate gridded wealth maps that can be used for socioeconomic analysis, policy planning, and research.

## Features

* Processes household wealth indices from DHS surveys
* Combines survey data with GPS coordinates to create spatial representations
* Creates appropriate buffers around survey points based on urban/rural classification
* Generates uniform grid cells for standardized analysis
* Calculates wealth indices for each grid cell using weighted averages
* Produces high-quality visualizations of wealth distribution
* Supports multi-country and multi-year analysis
* Exports results in GeoPackage format for GIS applications

## Installation

### Prerequisites

* Python 3.12 or higher
* Required Python packages (listed in requirements.txt)

### Setup

1. Clone the repository:

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/WealthMapping
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Data Structure

The tool expects a specific data structure:

```javascript
data/
├── GPS/
│   ├── [Country]/
│   │   ├── [Year]/
│   │   │   └── *.shp (GPS cluster points)
├── ShapeFiles/
│   ├── [Country]/
│   │   └── *.shp (Administrative boundaries)
├── WealthIndex/
│   └── wealth_index.parquet (Wealth index data)
└── Results/
    └── (Output directory, created automatically)
```

## Configuration

The tool is configured using a YAML file (`config.yaml`). Key configuration parameters include:

* Country and year specifications
* Coordinate reference systems
* Grid cell size
* Buffer sizes for urban and rural areas
* Visualization parameters

Example configuration:

```yaml
countries:
 -name: "Senegal"
  iso2: "SN"
  crs: "EPSG:32628"
  years: [2023, 2019, 2018, 2017, 2016, 2015]
 -name: "Ghana"
  iso2: "GH"
  crs: "EPSG:32630"
  years: [2022]default_crs: "EPSG:4326"

cell_size: 500 # meters
buffer:
  urban: 2000 # meters
  rural: 5050 # meters
```

## Usage

Run the wealth mapping process:

```bash
python main.py
```

This will:

1. Load data for all specified countries and years
2. Process wealth indices and GPS data
3. Create grid cells and calculate wealth indices
4. Generate visualizations
5. Save results to the output directory

## Output

The tool produces two main types of output:

1. **Geospatial data files** (GeoPackage format)

* Individual country-year wealth grids
* Combined multi-country, multi-year grid

2. **Visualizations** (PNG format)

* Individual wealth maps for each country-year
* Combined visualization with all country-year pairs

## Development

### Project Structure

```javascript
WealthMapping/
├── src/
│   ├── core/
│   │   └── grid_processor.py
│   ├── data_processing/
│   │   └── wealth_processor.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── logging_config.py
│   │   └── paths.py
│   └── visualization/
│       └── plots.py
├── config/
│   └── config.yaml
├── main.py
├── requirements.txt
└── setup.py
```

## Acknowledgements

* Spatial processing is powered by GeoPandas and Shapely
* Visualization components use Matplotlib

## Contact

For questions or support, please open an issue on the GitLab repository.

# WealthIndex

A Python-based tool for calculating household wealth indices from Demographic and Health Survey (DHS) data. This tool provides a comprehensive pipeline for data loading, missing value imputation, Factor Analysis of Mixed Data (FAMD), and wealth index calculation.

## Overview

The WealthIndex tool processes household survey data to create standardized wealth indices that can be used for socioeconomic analysis, policy planning, and research. It implements a sophisticated pipeline that handles data from multiple countries and survey years, employs advanced imputation techniques for missing values, and uses Factor Analysis of Mixed Data (FAMD) to calculate composite wealth indices.

## Features

* **Data Processing**
* Loads and processes DHS survey data from multiple countries and years
* Handles missing values with sophisticated imputation techniques
* Standardizes data types and values across different surveys
* **Missing Value Imputation**
* Implements multiple imputation techniques:
* * K-Nearest Neighbors (KNN)
  * Multiple Imputation by Chained Equations (MICE)
  * Random Forest based imputation (MissForest)
* Provides automatic comparison and selection of optimal imputation methods
* Optimizes imputation parameters through validation testing
* **Wealth Index Calculation**
* Uses Factor Analysis of Mixed Data (FAMD) for handling mixed numeric and categorical variables
* Performs parallel analysis to determine statistically significant components
* Weights components based on correlation with standard wealth indicators
* Normalizes indices for consistent interpretation
* **Performance Optimization**
* Implements parallel processing for computationally intensive operations
* Optimizes memory usage for large datasets
* Provides performance tracking and logging

## Installation

### Prerequisites

* Python 3.12 or higher
* Required Python packages (listed in requirements.txt)

### Setup

1. Clone the repository:

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/WealthIndex
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Data Structure

The tool expects a specific data structure:

```javascript
data/
├── DHS/
│   ├── [Country]_Data/
│   │   ├── [Year]/
│   │   │   └── *.DTA (Household survey data)
├── Results/
│   └── WealthIndex/
│       └── (Output directory, created automatically)
└── logs/
    └── (Log files, created automatically)
```

## Configuration

The tool is configured using a YAML file (`config.yaml`). Key configuration parameters include:

* Country and year specifications
* Columns to include in analysis
* Missing value handling settings
* Imputation method selection
* FAMD analysis parameters

Example configuration:

```yaml
country_year:
  Senegal:
    - "2023"
    - "2019"
  Ghana:
    - "2022"

imputation: "compare" # Options: "knn", "mice", "missforest", "compare"

#FAMD analysis configuration
n_simulations: 1000
plot_contributions: True
run_parallel_analysis: True
```

## Usage

Run the wealth index calculation:

```bash
python main.py
```

This will:

1. Load data for all specified countries and years
2. Process and standardize the data
3. Perform missing value imputation
4. Calculate wealth indices using FAMD
5. Save results to the output directory

## Output

The tool produces two main outputs:

1. **Wealth Index Data**

* Parquet file with calculated wealth indices for all households
* CSV file with the same data for broader compatibility

2. **Analysis Visualizations** (if enabled)

* FAMD component contribution plots
* Diagnostic visualizations for data quality assessment

## Architecture

The project is organized into several modules:

* `main.py`: Entry point and orchestration
* `src/data_processing/`: Data loading and preprocessing
* `src/data_processing/imputation/`: Missing value imputation implementations
* `src/analysis/`: FAMD and wealth index calculation
* `src/utils/`: Configuration, logging, and utility functions

## Advanced Features

### Imputation Method Comparison

When configured with `imputation: "compare"`, the tool runs all three imputation methods (KNN, MICE, MissForest) and selects the best performing one based on validation metrics. This ensures optimal handling of missing values for each dataset.

### Parallel Analysis

The FAMD implementation includes parallel analysis to determine the statistically significant components, which provides a data-driven approach to component selection rather than arbitrary cutoffs.

### Component Weighting

The tool weights FAMD components based on their correlation with standard wealth indicators (electricity, refrigerator, television, etc.), ensuring that the calculated wealth index aligns with established socioeconomic patterns.

## Acknowledgements

* This tool uses DHS survey data
* FAMD implementation uses the Prince Python library
* Imputation techniques are based on scikit-learn implementations

## Contact

For questions or support, please open an issue on the GitLab repository.

# Visualization

A Python-based visualization tool for generating wealth index maps from geopackage data. This tool creates individual and composite maps showing wealth distribution across Senegal for different years, using both original survey data and satellite image predictions.

## Overview

This visualization tool processes geopackage (GPKG) files containing wealth index data, generates standardized maps with consistent color scales, and compiles multi-year composite images for easy comparison of wealth distribution patterns over time. It handles both original survey-based wealth indices and those predicted from satellite imagery.

## Features

* **Data Visualization**
* Creates wealth distribution maps with consistent color scales
* Visualizes both original survey data and satellite-predicted wealth indices
* Generates comparative multi-year composite images
* **Spatial Data Processing**
* Handles geopackage (GPKG) files with wealth index data
* Manages coordinate reference systems for consistent mapping
* Validates and cleans spatial geometries
* **Statistical Analysis**
* Calculates key statistics for each dataset (mean, median, min, max, std)
* Displays statistics directly on generated maps
* Determines global min/max values for consistent color scaling

## Data Structure

The tool expects a specific data structure:

```javascript
data/
├── Results/
│   ├── LabelledGrids/
│   │   └── Senegal_[YEAR]_wealthindex_labelled[_inferred].gpkg
│   └── WealthMaps/
│       └── (Output maps, created automatically)
└── ShapeFiles/
    └── Senegal/
        └── *.shp (Administrative boundary shapefile)
```

## File Naming Convention

The tool expects GPKG files with the following naming pattern:

* `Senegal_[YEAR]_wealthindex_labelled.gpkg` - Original survey data
* `Senegal_[YEAR]_wealthindex_labelled_inferred.gpkg` - Satellite-predicted data

## Usage

Run the visualization script:

```bash
python visualization.py
```

This will:

1. Load all available geopackage files from the LabelledGrids directory
2. Process both original and inferred wealth index data for each year
3. Generate individual maps for each dataset with consistent color scales
4. Create composite images for both original and inferred data types
5. Save all outputs to the WealthMaps directory

## Output

The tool produces two types of output:

1. **Individual Maps**

* Filename format: `senegal_wealth_[YEAR]_[TYPE].png`
* Each map includes:
* * Color-coded wealth distribution
  * Country boundary overlay
  * Statistical summary (median, mean, min, max, std)
  * Consistent color scale across all years and types

2. **Composite Images**

* `composite_original_wealth_maps.png`- Combines all years of original survey data
* `composite_inferred_wealth_maps.png` - Combines all years of satellite-predicted data

## Technical Details

* Uses matplotlib for visualization with a diverging color map (RdBu_r)
* Implements GeoPandas for spatial data processing
* Sets standardized coordinate reference system (EPSG:32628 - UTM Zone 28N)
* Handles invalid or empty geometries through validation and cleaning
* Dynamically adjusts grid layout for composite images based on available data

## Requirements

* Python 3.x
* NumPy
* Pandas
* GeoPandas
* Matplotlib
* Pillow (PIL)
* pathlib

## Notes

* The tool automatically determines available years from the filenames
* Color scales are normalized across all datasets for consistent comparison
* Empty or invalid geometries are detected and reported during processing

# SatelliteImages

A Python toolkit for downloading, processing, and combining satellite imagery data from Sentinel-2 and VIIRS sources. This package creates standardized grid cells across countries and processes multi-year satellite data for use in spatial analysis and machine learning applications.

## Overview

The SatelliteImages toolkit orchestrates the end-to-end process of generating standardized satellite imagery datasets. It downloads optical imagery from Sentinel-2 and nighttime lights data from VIIRS, processes them into consistent formats, and combines them to create comprehensive multi-band datasets for each grid cell.

## Features

* **Grid Generation**
* Creates uniform grid cells over country boundaries
* Filters cells based on configurable overlap percentage
* Assigns unique IDs for consistent reference
* **Sentinel-2 Processing**
* Downloads multi-spectral imagery with cloud filtering
* Processes key spectral bands (RGB, NIR, SWIR)
* Calculates derived indices (NDVI, built-up index)
* Handles different Sentinel-2 collections based on year
* **VIIRS Nightlights Processing**
* Downloads nighttime light intensity data
* Calculates spatial gradient of nightlights
* Normalizes values using global statistics
* **Data Fusion**
* Combines Sentinel-2 and VIIRS data into unified arrays
* Standardizes pixel dimensions and coordinate systems
* Creates comprehensive metadata for each cell
* **Robust Error Handling**
* Implements multi-level retry mechanisms
* Logs and tracks failures for repair
* Recovers from Earth Engine API issues

## Installation

### Prerequisites

* Python 3.12 or higher
* Google Earth Engine account with API access
* Required Python packages (listed in requirements.txt)

### Setup

1. Clone the repository:

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/SatelliteImages
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package in development mode:

```bash
pip install -e .
```

4. Authenticate with Earth Engine:

```bash
earthengine authenticate
```

## Configuration

The toolkit is configured using a YAML file (`config.yaml`). Key configuration parameters include:

* Country specifications and years to process
* Grid cell size and overlap requirements
* Sentinel-2 and VIIRS band selection
* Cloud filtering thresholds
* Composite methods
* Processing parameters

Example configuration:

```yaml
countries:
-name: "Senegal"
 iso: "SEN"
 years: [2023, 2019, 2018, 2017, 2016]
 crs: "EPSG:32628"

cell_size_km: 5
min_area_percent: 40.0

sentinel:  
  bands: ["B2", "B3", "B4", "B8", "B11", "B12"]  
  cloud_threshold: 20  
  composite_method: "median"
viirs:
  bands: ["avg_rad"]
  composite_method: "median"
```

## Usage

Run the main pipeline:

```bash
python main.py
```

This will:

1. Create grid cells for each configured country
2. Download and process Sentinel-2 data for each year and cell
3. Download and process VIIRS data for each year and cell
4. Repair any failed downloads
5. Combine Sentinel-2 and VIIRS data for each cell
6. Generate comprehensive metadata

## Output

The toolkit produces a hierarchical directory structure containing processed satellite data for each grid cell, organized by country, year, and data type.

## Data Format

Each processed cell contains:

* **Sentinel-2 data:**
* RGB composite (3-band, 8-bit)
* NIR, SWIR1, SWIR2 bands (16-bit)
* NDVI and built-up indices (8-bit)
* **VIIRS data:**
* Nightlights intensity (16-bit)
* Nightlights gradient (16-bit)
* **Metadata:**
* Spatial information (coordinates, bounds, CRS)
* Processing parameters
* Array statistics
* Source information

## Advanced Features

### Parallel Processing

The toolkit uses parallel processing to maximize efficiency:

* Dynamically calculates optimal worker count based on CPU, memory, and API limits
* Processes cells in batches to manage memory usage
* Uses thread pools for I/O-bound operations

### Error Recovery

Comprehensive error handling includes:

* Multi-level retry mechanisms with exponential backoff
* Tracking and logging of failures
* Automated repair of failed downloads
* Session management to handle API issues

### Memory Management

Optimized memory usage through:

* Dynamic calculation of batch sizes based on available memory
* Garbage collection between processing steps
* Memory usage tracking per cell
* Optimized array storage formats

## Acknowledgements

* Earth Engine API for satellite data access
* Sentinel-2 data provided by ESA/Copernicus
* VIIRS nightlights data provided by NOAA

## Contact

For questions or support, please open an issue on the GitLab repository.

# LabellingTests

A Python toolkit for analyzing wealth index coverage thresholds across grid cells, supporting multi-year analysis with statistical evaluation of spatial autocorrelation and confidence intervals.

## Overview

This toolkit analyzes how different spatial coverage thresholds affect the quality and quantity of labeled grid cells when working with wealth index data. It processes multiple years of data, calculates key statistical metrics, and generates comprehensive visualizations to help determine optimal coverage thresholds for labeling.

## Features

* **Grid Cell Analysis**
* Loads standardized grid cells and wealth index data
* Calculates coverage percentages for each cell
* Computes weighted wealth indices based on intersection areas
* **Threshold Testing**
* Tests multiple coverage thresholds from a configurable range
* Calculates percentage of cells labeled at each threshold
* Supports both single-year and multi-year analysis
* **Statistical Evaluation**
* Computes spatial autocorrelation (Moran's I) to measure clustering
* Calculates confidence intervals for wealth indices
* Aggregates statistics across multiple years
* **Visualization**
* Generates plots showing how metrics change across thresholds
* Creates both year-specific and aggregated visualizations
* Includes error bars showing variation across years

## Installation

### Prerequisites

* Python 3.12 or higher
* Required Python packages (listed in requirements.txt)

### Setup

1. Clone the repository:

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/LabellingTests
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Configuration

The toolkit is configured using a YAML file (`config.yaml`). Key configuration parameters include:

* `min_threshold`: Minimum coverage threshold percentage to analyze
* `max_threshold`: Maximum coverage threshold percentage to analyze
* `threshold_steps`: Number of steps between min and max threshold

Example configuration:

```yaml
min_threshold: 30
max_threshold: 35
threshold_steps: 20
```

## Data Structure

The toolkit expects data in the following locations:

```javascript
data/Results/
├── SatelliteImageData/
│   └── grids/
│       └── Senegal_grid_5km.gpkg
├── GPSResults/
│   └── WealthGPS/
│       ├── SN_2023_output.gpkg
│       ├── SN_2019_output.gpkg
│       ├── SN_2018_output.gpkg
│       └── ...
└── LabelTestResults/
    └── (Output directory for analysis results)
```

## Usage

Run the analysis script:

```bash
python main.py
```

This will:

1. Load grid cells and wealth index data for multiple years
2. Calculate coverage percentages for each cell
3. Analyze the effect of different coverage thresholds
4. Generate statistical metrics and visualizations
5. Save results to the output directory

## Output

The toolkit produces the following outputs:

* **CSV files**
* Year-specific threshold analysis results
* Aggregated threshold analysis across all years
* **Visualization plots**
* Percentage of cells labeled vs. threshold
* Spatial autocorrelation (Moran's I) vs. threshold
* Confidence interval width vs. threshold
* Both year-specific and aggregated plots with error ranges

## Key Metrics

### Percentage of Cells Labeled

Shows how many grid cells meet the coverage threshold criteria. Higher thresholds result in fewer labeled cells but potentially higher quality labels.

### Spatial Autocorrelation (Moran's I)

Measures how clustered or dispersed the wealth values are. Higher positive values indicate stronger spatial clustering, which is typically desirable for robust spatial predictions.

### Confidence Interval Width

Indicates the statistical uncertainty in the wealth index estimates. Narrower confidence intervals suggest more precise estimates.

## Advanced Features

### Multi-Year Aggregation

The toolkit can process multiple years of wealth data and aggregate the results, showing:

* Average metrics across years
* Min/max ranges to indicate year-to-year variability
* Consistent trends that persist across different survey years

### Spatial Weights

Uses Queen contiguity for spatial weight matrix construction, considering all neighboring cells that share a boundary or corner point when calculating spatial autocorrelation.

## Acknowledgements

* PySAL and ESDA for spatial analysis tools
* GeoPandas for geospatial data processing
* Matplotlib for visualization

## Contact

For questions or support, please open an issue on the GitLab repository.

# LabelImagery

A Python toolkit for applying wealth index data to geospatial grid cells with configurable coverage thresholds. This package processes wealth index data from multiple countries and years, calculates area-weighted wealth indices, and applies coverage thresholds to create labeled grid datasets.

## Overview

The LabelImagery toolkit generates labeled grid cells by calculating the overlap between wealth index polygons and a regular grid. It computes area-weighted wealth indices for each cell and applies a configurable coverage threshold to ensure data quality. The resulting labeled grids can be used for machine learning, visualization, and spatial analysis.

## Features

* **Grid Cell Processing**
* Loads standardized grid cells for multiple countries
* Handles different coordinate reference systems automatically
* Maintains geospatial integrity throughout processing
* **Wealth Index Integration**
* Calculates intersection between grid cells and wealth index polygons
* Computes area-weighted wealth indices for overlapping regions
* Determines coverage percentage for quality filtering
* **Threshold Application**
* Applies configurable coverage thresholds to ensure data quality
* Filters out cells with insufficient coverage
* Preserves spatial relationships in resulting datasets
* **Multi-Country Support**
* Processes data for different countries with appropriate CRS handling
* Supports country-specific coordinate reference systems
* Maintains consistent processing across different geographies

## Installation

### Prerequisites

* Python 3.12 or higher
* Required Python packages (listed in requirements.txt)

### Setup

1. Clone the repository:

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/LabelImagery
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Configuration

The toolkit is configured using a YAML file (`config.yaml`). Key configuration parameters include:

* `threshold`: Coverage percentage threshold for labeling cells (default: 32)
* `countries`: List of country codes to process (e.g., ["SN"], ["GH"], or ["SN", "GH"])
* `force_reprocess`: Whether to reprocess existing output files (default: false)

Example configuration:

```yaml
threshold: 32
countries: ["SN"]
force_reprocess: false
```

## Data Structure

The toolkit expects data in the following locations:

```javascript
data/Results/
├── SatelliteImageData/
│   └── grids/
│       ├── Senegal_grid_5km.gpkg
│       └── Ghana_grid_5km.gpkg
├── GPSResults/
│   └── WealthGPS/
│       ├── SN_2023_output.gpkg
│       ├── SN_2019_output.gpkg
│       ├── GH_2022_output.gpkg
│       └── ...
└── LabelledGrids/
    └── (Output directory for labeled grids)
```

## Usage

Run the labeling process:

`python main.py`

This will:

1. Load grid cells for each specified country
2. Process wealth index data for each country and year
3. Calculate coverage percentages and weighted wealth indices
4. Apply the configured coverage threshold
5. Save labeled grid cells to the output directory

## Output

The toolkit produces labeled grid files in GeoPackage format:

```javascript
data/Results/LabelledGrids/
├── Senegal_2023_wealthindex_labelled.gpkg
├── Senegal_2019_wealthindex_labelled.gpkg
├── Ghana_2022_wealthindex_labelled.gpkg
└── ...
```

Each output file contains:

* Original grid cell geometry
* Coverage percentage for each cell
* Weighted wealth index for cells meeting the threshold
* NaN values for cells below the threshold

## Processing Workflow

1. **Data Loading**

* Load grid cells for each country
* Load wealth index data for each country-year pair

2. **Coverage Calculation**

* For each grid cell, find intersecting wealth polygons
* Calculate intersection area and percentage of coverage
* Compute area-weighted wealth index based on intersections

3. **Threshold Application**

* Apply configured coverage threshold to filter cells
* Set wealth index to NaN for cells below threshold
* Preserve all original geometry and metadata

4. **Output Generation**

* Save labeled grid cells to GeoPackage format
* Include coverage statistics and weighted indices
* Maintain geospatial integrity for downstream applications

## Performance Considerations

* Skips processing for existing output files unless `force_reprocess` is enabled
* Uses tqdm progress bars for monitoring long-running operations
* Implements proper error handling and logging throughout the pipeline

## Country-Specific Handling

* Senegal (SN): Uses EPSG:32628 (UTM Zone 28N)
* Ghana (GH): Uses EPSG:32630 (UTM Zone 30N)
* Automatically converts wealth data to match grid CRS

## Contact

For questions or support, please open an issue on the GitLab repository.

# Dashboard

An interactive web-based dashboard for visualizing wealth index data across different years and data types. This application provides geospatial mapping, statistical analysis, and time series visualization of wealth distribution in Senegal.

## Overview

The Wealth Index Dashboard combines pre-processing capabilities with an interactive web interface to explore and visualize wealth distribution patterns. It supports both original survey-based wealth indices and satellite image predicted data, allowing for comparison across multiple years.

## Features

* **Interactive Geospatial Visualization**
* Choropleth maps showing wealth distribution
* Consistent color scale across years for comparison
* Major city markers for geographic reference
* Country boundary overlay
* **Time Series Analysis**
* Year-over-year comparison of median wealth indices
* Visual tracking of trends for both data types
* Interactive highlighting of selected year
* **Statistical Insights**
* Key statistics for each dataset (median, mean, min, max, std)
* Real-time updates based on user selections
* Contextual comparison between original and predicted data
* **User Controls**
* Year selection slider for temporal analysis
* Data type toggle between original and predicted data
* Responsive layout for different screen sizes

## Components

### Pre-processing Module

The pre-processing component transforms raw geospatial data into optimized formats for visualization:

* Loads and processes wealth index data from GeoPackage files
* Calculates global statistics and per-year metrics
* Converts data to GeoJSON format for efficient web rendering
* Generates metadata for dashboard configuration
* Clips data to country boundaries for consistent visualization

### Dashboard Application

The dashboard provides an intuitive interface for data exploration:

* Built with Dash and Plotly for interactive visualizations
* Responsive Bootstrap-based layout
* Dark theme for optimal visualization of color gradients
* Automatic browser launch for immediate access

## Installation

### Prerequisites

* Python 3.6 or higher
* Required Python packages:
* dash
* dash-bootstrap-components
* plotly
* geopandas
* pandas
* numpy

### Setup

1. Clone the repository:

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/Dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the data:

```bash
python preprocess.py
```

4. Launch the dashboard:

```bash
python main.py
```

## Data Structure

The dashboard expects data in the following structure:

```javascript
data/
├── Results/
│   ├── LabelledGrids/
│   │   ├── Senegal_2023_wealthindex_labelled.gpkg
│   │   ├── Senegal_2023_wealthindex_labelled_inferred.gpkg
│   │   └── ...
│   └── DashboardData/
│       ├── metadata.json
│       ├── 2023_original.geojson
│       ├── 2023_inferred.geojson
│       └── ...
└── ShapeFiles/
    └── Senegal/
        └── *.shp
```

## Usage

After launching the application, a browser window will automatically open to the dashboard interface. From there, you can:

1. Select different years using the slider control
2. Toggle between original survey data and satellite-predicted data
3. View the wealth distribution map with consistent color scaling
4. Examine statistical summaries for the selected dataset
5. Track changes in median wealth over time through the time series chart

## Performance Considerations

* GeoJSON files are pre-processed to optimize loading times
* Visualization elements use efficient rendering techniques
* Map interactions maintain state between updates for smooth transitions
* Pre-calculated statistics reduce computation during user interaction

## Acknowledgements

* Dash and Plotly for interactive visualization components
* GeoPandas for geospatial data processing
* Carto for providing the dark matter basemap

## Contact

For questions or support, please open an issue on the GitLab repository.

# Regressions

This Python script performs linear regression analysis on geospatial data files to predict wealth indices. The script processes files, performs per-year and combined regressions, and generates various plots and predictions.

## Features

- **Spatial Data Processing**: Handles GeoPackage (.gpkg) files containing wealth index data
- **Dynamic Year Detection**: Automatically detects available years from filenames
- **Dual Data Sources**: Supports both original survey data and satellite-inferred predictions
- **Statistical Summary**: Calculates median, mean, min/max, and standard deviation for each dataset
- **Composite Visualization**: Creates combined overview images showing temporal comparisons

## Installation

### Prerequisites

* Python 3.8+
* Required packages (listed in requirements.txt)

### Setup

1. Clone the repository

```bash
git clone https://gitlab.lrz.de/hailan.wang/satellite.git && cd satellite/Regressions
```

2. Install dependencies

```bash
pip install -r requirements.txt
# Or using conda:
conda install --file requirements.txt
```

## Directory Structure

The tool expects a specific data structure:

```bash
data/
├── Results/
│   ├── LabelledGrids/         # Input GeoPackage files
│   │   ├── Senegal_YYYY_wealthindex_labelled.gpkg      # Original survey data
│   │   └── Senegal_YYYY_wealthindex_labelled_inferred.gpkg  # Satellite predictions
│   └── WealthMaps/            # Output PNG images
└── ShapeFiles/
    └── Senegal/               # Boundary shapefile
```

## Configuration

1. Input Data Preparation:

   - Place GeoPackage files in data/Results/LabelledGrids/
   - Filename format: Senegal_YYYY_wealthindex_labelled[_inferred].gpkg

     - YYYY: 4-digit year
     - _inferred: Marks satellite-predicted data
2. Boundary Data:

   - Ensure Senegal boundary shapefile exists in data/ShapeFiles/Senegal/
   - Script uses first found .shp file in this directory

   ### Technical Settings


   * ****Coordinate System**** : UTM Zone 28N (EPSG:32628) for all spatial operations
   * ****Color Mapping**** : Diverging RdBu_r colormap (blue = low wealth, red = high wealth)
   * ****Required Columns**** : Input files must contain a `weighted_wealth` column

## Usage

Run the wealth mapping script:

```bash
python path/to/wealth_mapping_script.py
```

This will:

1. Load and process all available GeoPackage files
2. Perform regression analysis on the data
3. Generate individual wealth maps for each year and data type
4. Create composite images for temporal comparison
5. Save all outputs to the specified directories

## Outputs

1. Individual Maps:

   - Saved as PNG in data/Results/WealthMaps/
   - Filename format: senegal_wealth_YYYY_[original|inferred].png
2. Composite Images:

   - composite_original_wealth_maps.png
   - composite_inferred_wealth_maps.png
   - Grid layout showing multi-year comparison## Troubleshooting

# Training The Model

The jupyter notebook `cnn_training_tuning.ipynb` is used for training the deep learning model for wealth index prediction. It contains the training and hyperparameter tuning logic for the deep learning model. This notebook walks through:
- Dataset loading and preprocessing
- CNN architecture definition
- Model training with validation
- Hyperparameter tuning (e.g., learning rate, batch size)
- Saving the best model
- Visualizing training history (loss, accuracy)

Further information on the specific usage and steps is given in the notebook itself.

# Inference

The jupyter notebook `cnn_inference.ipynb` provides a streamlined way to load a trained CNN model and perform inference on new data. This includes:
- Loading the saved model
- Preprocessing test or input data
- Performing predictions
- (Optional) Visualizing results

Further information on the specific usage and steps is given in the notebook itself.
