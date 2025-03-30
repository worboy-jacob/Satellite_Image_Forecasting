from setuptools import setup, find_packages

setup(
    name="Dashboard",
    version="0.1.0",
    description="Package for Dashboard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["dash", "dash_bootstrap_components", "geopandas", "numpy", "pandas", "plotly"],
    package_data={"": []},
    entry_points={
        "console_scripts": [
            "Dashboard=main:main",
        ],
    },
    python_requires=">=3.12",
)
