from setuptools import setup, find_packages

setup(
    name="LabelImagery",
    version="0.1.0",
    description="Package for LabelImagery",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["geopandas", "numpy", "PyYAML", "setuptools", "tqdm"],
    package_data={"": ["d:\satellite\Code\LabelImagery\config\config.yaml"]},
    entry_points={
        "console_scripts": [
            "LabelImagery=main:main",
        ],
    },
    python_requires=">=3.12",
)
