# setup.py
from setuptools import setup, find_packages

setup(
    name="wealth_index",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "prince",
        "pyyaml",
        "matplotlib",
        "tqdm",
        "pyarrow",
    ],
)
