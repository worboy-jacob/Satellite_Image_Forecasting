from setuptools import setup, find_packages

setup(
    name="cnn",
    version="0.1.0",
    description="Package for cnn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    package_data={"": []},
    entry_points={
        "console_scripts": [
            "cnn=src.main:main",
        ],
    },
    python_requires=">=3.12",
)
