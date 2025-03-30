# Wealth Index Mapping Tool

This Python script performs linear regression analysis on geospatial data files to predict wealth indices. The script processes files, performs per-year and combined regressions, and generates various plots and predictions.

## Features

- **Spatial Data Processing**: Handles GeoPackage (.gpkg) files containing wealth index data
- **Dynamic Year Detection**: Automatically detects available years from filenames
- **Dual Data Sources**: Supports both original survey data and satellite-inferred predictions
- **Statistical Summary**: Calculates median, mean, min/max, and standard deviation for each dataset
- **Composite Visualization**: Creates combined overview images showing temporal comparisons

## Requirements

- Python 3.8+
- Required packages:
* [![Pandas][pandas]][pandas-url]
* [![Numpy][numpy]][numpy-url]
* [![Geoandas][geopandas]][geopandas-url]
* [![Matplotlib][matplotlib]][matplotlib-url]
* [![Pillow][pillow]][pillow-url]

## Installation
1. Clone the repository

```bash
git clone [your-repository-url]
cd [repository-directory]
```
2. Install dependencies

```bash
pip install -r requirements.txt
# Or using conda:
conda install --file requirements.txt
```

## Directory Structure

```bash
data/
├── Results/
│   ├── LabelledGrids/         # Input GeoPackage files
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

## Usage
```bash
python path/to/wealth_mapping_script.py
```

## Outputs
1. Individual Maps:

    - Saved as PNG in data/Results/WealthMaps/

    - Filename format: senegal_wealth_YYYY_[original|inferred].png

2. Composite Images:

    - composite_original_wealth_maps.png

    - composite_inferred_wealth_maps.png

    - Grid layout showing multi-year comparison

## Technical Notes
- Coordinate System: Uses UTM Zone 28N (EPSG:32628) for all spatial operations

- Color Mapping: Diverging RdBu_r colormap (blue = low, red = high wealth)

- Data Validation:

    - Automatically skips invalid/empty geometries

    - Checks for required weighted_wealth column

## Troubleshooting
***Common Issues***:

1. Missing Files:

    - Confirm files are in correct directories

    - Verify filename formatting

2. CRS Mismatches:

    - Ensure input files are in WGS84 (EPSG:4326)

    - Script will reproject to EPSG:32628

3. Geometry Errors:

    - Check for invalid polygons in input GeoPackages

-  Script attempts automatic geometry cleaning

***Error Messages***:

- No GPKG files found: Check LabelledGrids directory

    - No weighted_wealth column: Validate input file structure

    - Invalid geometries: Pre-process data with GIS software

<!-- MARKDOWN LINKS & IMAGES -->
[pandas]:https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white 
[pandas-url]:https://pandas.pydata.org/ 
[numpy]:https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white 
[numpy-url]:https://numpy.org/ 
[geopandas]:https://img.shields.io/badge/geopandas-09a3d5?style=for-the-badge&logo=geopandas&logoColor=white 
[geopandas-url]:https://geopandas.org/ 
[matplotlib]:https://img.shields.io/badge/matplotlib-3776ab?style=for-the-badge&logo=matplotlib&logoColor=white
[matplotlib-url]: https://matplotlib.org/
[pillow]:https://img.shields.io/badge/pillow-cc0000?style=for-the-badge&logo=pillow&logoColor=white 
[pillow-url]:https://python-pillow.org/