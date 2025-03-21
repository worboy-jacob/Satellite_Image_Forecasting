from setuptools import setup, find_packages

setup(
    name="SatelliteImages",
    version="0.1.0",
    description="Package for SatelliteImages",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "earthengine_api",
        "geopandas",
        "numpy",
        "opencv_python",
        "psutil",
        "PyYAML",
        "rasterio",
        "Requests",
        "setuptools",
        "Shapely",
        "tqdm",
        "urllib3",
    ],
    package_data={"": ["d:\satellite\Code\SatelliteImages\config\config.yaml"]},
    entry_points={
        "console_scripts": [
            "SatelliteImages=main:main",
        ],
    },
    python_requires=">=3.12",
)
