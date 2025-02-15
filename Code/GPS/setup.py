# setup.py
from setuptools import setup, find_packages

setup(
    name="gps_wealth_mapping",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "geopandas>=0.12.0",
        "numpy>=1.23.0",
        "matplotlib>=3.5.0",
        "contextily>=1.3.0",
        "shapely>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pyarrow>=12.0.0",
        "pycountry>=22.3.5",
        "pyproj>=3.5.0",
    ],
    python_requires=">=3.8",
)
