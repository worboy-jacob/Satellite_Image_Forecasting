from setuptools import setup, find_packages

setup(
    name="Visualization",
    version="0.1.0",
    description="Package for Visualization",
    packages=find_packages(),
    # package_dir not needed for flat structure,
    install_requires=["geopandas", "matplotlib", "numpy", "pandas", "Pillow"],
    package_data={"": []},
    entry_points={
        "console_scripts": [
            "Visualization=src.main:main",
        ],
    },
    python_requires=">=3.12",
)
