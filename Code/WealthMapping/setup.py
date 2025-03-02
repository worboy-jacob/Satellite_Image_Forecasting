from setuptools import setup, find_packages

setup(
    name="wealth_mapping",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "geopandas",
        "numpy",
        "shapely",
        "folium",
        "branca",
        "rtree",
        "tqdm",
        "matplotlib",
        "contextily",
    ],
    description="A package for mapping wealth indices using GPS data",
)
