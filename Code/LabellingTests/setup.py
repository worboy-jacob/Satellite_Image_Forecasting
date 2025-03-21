from setuptools import setup, find_packages

setup(
    name="LabellingTests",
    version="0.1.0",
    description="Package for LabellingTests",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "esda",
        "geopandas",
        "libpysal",
        "matplotlib",
        "numpy",
        "pandas",
        "PyYAML",
        "scipy",
        "setuptools",
        "Shapely",
        "tqdm",
    ],
    package_data={"": ["d:\satellite\Code\LabellingTests\config\config.yaml"]},
    entry_points={
        "console_scripts": [
            "LabellingTests=main:main",
        ],
    },
    python_requires=">=3.12",
)
