from setuptools import setup, find_packages

setup(
    name="cnn",
    version="0.1.0",
    description="Package for cnn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["geopandas", "ipywidgets", "matplotlib", "numpy", "pandas", "scikit_learn", "seaborn", "torch", "torchvision", "tqdm", "jupyter", "ipykernel", "nbformat"],
    package_data={"": ["src\cnn_inference.ipynb", "src\cnn_training_tuning.ipynb"]},
    entry_points={
        "console_scripts": [
            "cnn=src.main:main",
        ],
    },
    python_requires=">=3.12",
    include_package_data=True,
)
