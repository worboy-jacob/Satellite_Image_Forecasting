from setuptools import setup, find_packages

setup(
    name="Regressions",
    version="0.1.0",
    description="Package for Regressions",
    packages=find_packages(),
    # package_dir not needed for flat structure,
    install_requires=["geopandas", "matplotlib", "numpy", "pandas", "scikit_learn", "seaborn"],
    package_data={"": []},
    entry_points={
        "console_scripts": [
            "Regressions=src.main:main",
        ],
    },
    python_requires=">=3.12",
)
