from setuptools import setup, find_packages

setup(
    name="WealthIndex",
    version="0.1.0",
    description="Package for WealthIndex",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "prince",
        "psutil",
        "PyYAML",
        "scikit_learn",
        "scipy",
        "seaborn",
        "setuptools",
        "tqdm",
    ],
    package_data={"": ["d:\satellite\Code\WealthIndex\config\config.yaml"]},
    entry_points={
        "console_scripts": [
            "WealthIndex=main:main",
        ],
    },
    python_requires=">=3.12",
)
