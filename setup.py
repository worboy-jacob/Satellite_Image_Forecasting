from setuptools import setup, find_packages

setup(
    name="satellite",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    package_data={"": []},
    entry_points={
        "console_scripts": [
            "satellite=main:main",
        ],
    },
    python_requires=">=3.12",
)
