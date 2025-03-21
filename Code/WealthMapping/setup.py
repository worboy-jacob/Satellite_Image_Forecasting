from setuptools import setup, find_packages

setup(
    name="WealthMapping",
    version="0.1.0",
    description="Package for WealthMapping",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "geopandas",
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "psutil",
        "PyYAML",
        "setuptools",
        "Shapely",
        "tqdm",
    ],
    package_data={"": ["d:\satellite\Code\WealthMapping\config\config.yaml"]},
    entry_points={
        "console_scripts": [
            "WealthMapping=main:main",
        ],
    },
    python_requires=">=3.12",
)
