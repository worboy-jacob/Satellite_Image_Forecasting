#!/usr/bin/env python3
"""
Dependencies and Setup.py Generator

This script analyzes Python files and Jupyter notebooks in a project,
identifies dependencies, and generates setup.py and requirements.txt files.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import subprocess
import shutil


def exit_with_error(message):
    """Print error message and exit with error code."""
    print(f"ERROR: {message}")
    sys.exit(1)


def scan_directory_structure():
    """
    Analyze the current directory structure to understand the project layout.

    Returns:
        dict: Information about the project structure
    """
    current_dir = Path(__file__).parent
    project_name = current_dir.name

    print(f"Analyzing project structure in: {current_dir}")

    # Check if we're in the right directory level
    src_dir = current_dir / "src"
    if not src_dir.exists():
        print(f"Warning: 'src' directory not found directly in {project_name}")

        # Look for src in subdirectories
        possible_src_dirs = list(current_dir.glob("**/src"))
        if possible_src_dirs:
            src_dir = possible_src_dirs[0]
            print(f"Found 'src' directory at: {src_dir}")
        else:
            # If no src directory, use the current directory for scanning
            src_dir = current_dir
            print("Using current directory for code scanning")

    # Find main.py
    main_file = current_dir / "main.py"
    if not main_file.exists():
        main_files = list(current_dir.glob("**/main.py"))
        if main_files:
            main_file = main_files[0]
            print(f"Found main.py at: {main_file}")
        else:
            print("Warning: main.py not found, entry point will need manual adjustment")
            main_file = None

    # Find config directory
    config_dir = current_dir / "config"
    if not config_dir.exists():
        config_dirs = list(current_dir.glob("**/config"))
        if config_dirs:
            config_dir = config_dirs[0]
            print(f"Found config directory at: {config_dir}")
        else:
            print("Warning: No config directory found")
            config_dir = None

    # Find notebooks
    notebook_files = list(current_dir.glob("**/*.ipynb"))
    # Filter out checkpoint files
    notebook_files = [
        nb for nb in notebook_files if ".ipynb_checkpoints" not in str(nb)
    ]

    if notebook_files:
        print(f"Found {len(notebook_files)} Jupyter notebook files")
    else:
        print("No Jupyter notebook files found")

    return {
        "project_name": project_name,
        "current_dir": current_dir,
        "src_dir": src_dir,
        "main_file": main_file,
        "config_dir": config_dir,
        "notebook_files": notebook_files,
    }


def ensure_pipreqs_installed():
    """Ensure pipreqs is installed in the current environment."""
    try:
        import pipreqs

        print("✓ pipreqs is already installed")
        return True
    except ImportError:
        print("Installing pipreqs...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pipreqs"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ pipreqs installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install pipreqs: {e}")
            print(f"Output: {e.stdout}\n{e.stderr}")
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error installing pipreqs: {e}")
            return False


def convert_notebook_to_python(notebook_path, output_dir):
    """
    Convert a Jupyter notebook to a Python file.

    Args:
        notebook_path: Path to the notebook file
        output_dir: Directory to save the Python file

    Returns:
        Path to the generated Python file or None if conversion failed
    """
    try:
        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)

        # Create Python file path
        nb_name = os.path.basename(notebook_path).replace(".ipynb", ".py")
        py_file_path = os.path.join(output_dir, nb_name)

        # Extract code cells and write to Python file
        with open(py_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Converted from notebook: {notebook_path}\n\n")

            for cell in notebook_content.get("cells", []):
                if cell.get("cell_type") == "code":
                    # Join cell source lines
                    source = "".join(cell.get("source", []))
                    if source.strip():
                        f.write(source)
                        # Ensure there's a newline at the end
                        if not source.endswith("\n"):
                            f.write("\n")
                        f.write("\n")

        return py_file_path

    except Exception as e:
        print(f"Warning: Failed to convert notebook {notebook_path}: {e}")
        return None


def find_dependencies(project_dir, notebook_files=None):
    """
    Find all dependencies from Python files and Jupyter notebooks.

    Args:
        project_dir: Path to the project directory
        notebook_files: List of notebook files

    Returns:
        List of unique dependency names
    """
    # Create a temporary directory for converted notebooks
    temp_dir = None
    converted_files = []

    try:
        # Process notebooks if any exist
        if notebook_files:
            print(f"Converting {len(notebook_files)} notebooks to Python files...")
            temp_dir = tempfile.mkdtemp()

            for notebook in notebook_files:
                py_file = convert_notebook_to_python(notebook, temp_dir)
                if py_file:
                    converted_files.append(py_file)

            print(f"✓ Converted {len(converted_files)} notebooks to Python files")

        # Create a temporary file to store the requirements
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            req_path = temp_file.name

        # Run pipreqs on the project directory
        print(f"Analyzing Python dependencies in {project_dir}...")
        cmd = [
            sys.executable,
            "-m",
            "pipreqs.pipreqs",
            str(project_dir),
            "--savepath",
            req_path,
            "--force",
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Read requirements from Python files
        with open(req_path, "r") as f:
            requirements = [
                line.strip().split("==")[0]  # Remove version specifiers
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]

        print(f"✓ Found {len(requirements)} dependencies from Python files")

        # If we have converted notebooks, run pipreqs on the temp directory too
        if temp_dir and converted_files:
            notebook_req_path = os.path.join(temp_dir, "notebook_requirements.txt")

            cmd = [
                sys.executable,
                "-m",
                "pipreqs.pipreqs",
                temp_dir,
                "--savepath",
                notebook_req_path,
                "--force",
            ]

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)

                # Read requirements from notebooks
                with open(notebook_req_path, "r") as f:
                    notebook_requirements = [
                        line.strip().split("==")[0]  # Remove version specifiers
                        for line in f.readlines()
                        if line.strip() and not line.startswith("#")
                    ]

                print(
                    f"✓ Found {len(notebook_requirements)} dependencies from notebooks"
                )
                requirements.extend(notebook_requirements)

            except subprocess.CalledProcessError as e:
                print(f"Warning: Error analyzing notebook dependencies: {e}")
                print(f"Command output: {e.stdout}\n{e.stderr}")
            except Exception as e:
                print(f"Warning: Unexpected error running pipreqs on notebooks: {e}")

        # Normalize and deduplicate dependencies
        excluded_packages = ["pipreqs", "docopt"]
        normalized_deps = []

        for dep in requirements:
            # Normalize case
            dep_lower = dep.lower()

            # Skip excluded packages
            if dep_lower in [p.lower() for p in excluded_packages]:
                continue

            # Skip if already in the normalized list (case-insensitive)
            if any(r.lower() == dep_lower for r in normalized_deps):
                continue

            normalized_deps.append(dep)

        # Add notebook-specific dependencies if notebooks are present
        if notebook_files:
            notebook_specific_deps = ["jupyter", "ipykernel", "nbformat"]
            for dep in notebook_specific_deps:
                if dep.lower() not in [d.lower() for d in normalized_deps]:
                    normalized_deps.append(dep)

        print(f"✓ Found {len(normalized_deps)} total unique dependencies")
        if normalized_deps:
            print(", ".join(normalized_deps))

        return normalized_deps

    except subprocess.CalledProcessError as e:
        print(f"ERROR: pipreqs failed: {e}")
        print(f"Command output: {e.stdout}\n{e.stderr}")
        return []
    except Exception as e:
        print(f"ERROR: Unexpected error finding dependencies: {e}")
        return []
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(req_path):
                os.unlink(req_path)
        except:
            pass

        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                print(f"Warning: Failed to clean up temporary directory: {temp_dir}")


def find_yaml_files(config_dir):
    """
    Find all YAML files in the config directory.

    Args:
        config_dir: Path to the config directory

    Returns:
        List of YAML file paths relative to the config directory
    """
    yaml_files = []

    if not config_dir or not config_dir.exists():
        print("Warning: No config directory to scan for YAML files")
        return yaml_files

    try:
        # Find all .yaml and .yml files
        yaml_files = list(config_dir.glob("**/*.yaml"))
        yaml_files.extend(config_dir.glob("**/*.yml"))

        # Convert to string paths
        yaml_files = [str(f) for f in yaml_files]

        print(f"Found {len(yaml_files)} YAML configuration files")
    except Exception as e:
        print(f"Warning: Error finding YAML files: {e}")

    return yaml_files


def determine_entry_point(project_info):
    """
    Determine the appropriate entry point for the package.

    Args:
        project_info: Dictionary with project structure information

    Returns:
        String representing the entry point path
    """
    main_file = project_info["main_file"]
    src_dir = project_info["src_dir"]
    current_dir = project_info["current_dir"]
    project_name = project_info["project_name"]

    if not main_file:
        print("No main.py found, using default entry point")
        return "src.main:main"

    # Determine the relative import path
    try:
        # Get the path relative to the current directory
        rel_path = main_file.relative_to(current_dir)
        # Convert to import notation (replace / with .)
        import_path = str(rel_path).replace(os.sep, ".")[:-3]  # Remove .py extension

        # If the main file is directly in src, adjust the path
        if str(rel_path) == "src/main.py":
            entry_point = "src.main:main"
        elif str(rel_path) == "main.py":
            entry_point = "main:main"
        else:
            entry_point = f"{import_path}:main"

        print(f"Using entry point: {entry_point}")
        return entry_point
    except Exception as e:
        print(f"Warning: Error determining entry point: {e}")
        return "src.main:main"  # Default fallback


def generate_setup_py(project_info, dependencies, yaml_files):
    """
    Generate setup.py based on project structure and dependencies.

    Args:
        project_info: Dictionary with project structure information
        dependencies: List of dependencies
        yaml_files: List of YAML configuration files
    """
    project_name = project_info["project_name"]
    src_dir = project_info["src_dir"]
    current_dir = project_info["current_dir"]
    notebook_files = project_info.get("notebook_files", [])

    # Python version
    python_version = f">={sys.version_info.major}.{sys.version_info.minor}"

    # Format dependencies for setup.py
    deps_str = "[" + ", ".join(f'"{dep}"' for dep in dependencies) + "]"

    # Format YAML files and notebook files for package_data
    package_data = {}

    # Add YAML files if any
    if yaml_files:
        package_data[""] = yaml_files

    # Add notebook files to package_data if any
    if notebook_files:
        notebook_paths = [str(f.relative_to(current_dir)) for f in notebook_files]
        if "" in package_data:
            package_data[""].extend(notebook_paths)
        else:
            package_data[""] = notebook_paths

    # Format package_data for setup.py
    if package_data:
        package_data_items = []
        for k, v in package_data.items():
            v_str = "[" + ", ".join(f'"{item}"' for item in v) + "]"
            package_data_items.append(f'"{k}": {v_str}')

        package_data_str = "{" + ", ".join(package_data_items) + "}"
    else:
        package_data_str = '{"": []}'

    # Determine entry point
    entry_point = determine_entry_point(project_info)

    # Determine package setup logic based on src directory structure
    is_src_standard = src_dir.name == "src" and src_dir.parent == current_dir

    if is_src_standard:
        # Standard src directory structure
        packages_line = 'packages=find_packages(where="src")'
        package_dir_line = 'package_dir={"": "src"}'
    else:
        # Either a flat structure or a non-standard src location
        rel_src = src_dir.relative_to(current_dir)
        if str(rel_path := rel_src) == ".":
            # Flat structure
            packages_line = "packages=find_packages()"
            package_dir_line = "# package_dir not needed for flat structure"
        else:
            # Non-standard src location
            rel_src_str = str(rel_src)
            packages_line = f'packages=find_packages(where="{rel_src_str}")'
            package_dir_line = f'package_dir={{"": "{rel_src_str}"}}'

    # Generate setup.py file
    setup_contents = f"""from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="0.1.0",
    description="Package for {project_name}",
    {packages_line},
    {package_dir_line},
    install_requires={deps_str},
    package_data={package_data_str},
    entry_points={{
        "console_scripts": [
            "{project_name}={entry_point}",
        ],
    }},
    python_requires="{python_version}",
    include_package_data=True,
)
"""

    # Write setup.py
    setup_path = current_dir / "setup.py"
    try:
        with open(setup_path, "w") as setup_file:
            setup_file.write(setup_contents)
        print(f"✅ setup.py generated at: {setup_path}")
    except Exception as e:
        exit_with_error(f"Failed to write setup.py: {e}")

    # Write requirements.txt
    req_path = current_dir / "requirements.txt"
    try:
        with open(req_path, "w") as req_file:
            req_file.write("\n".join(dependencies))
        print(f"✅ requirements.txt generated at: {req_path}")
    except Exception as e:
        exit_with_error(f"Failed to write requirements.txt: {e}")


def create_init_files(directory):
    """
    Recursively create __init__.py files in directories containing Python files.

    Args:
        directory: Path to the root directory

    Returns:
        tuple: (created_count, existing_count)
    """
    print(f"Creating __init__.py files in Python package directories...")

    # Directories to exclude
    exclude_dirs = {
        ".git",
        ".github",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "build",
        "dist",
        ".tox",
        ".eggs",
        "node_modules",
        "config",
        "logs",
        ".ipynb_checkpoints",  # Exclude Jupyter checkpoints
    }

    created_count = 0
    existing_count = 0
    created_files = []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

        # Check if directory contains Python files
        has_py_files = any(f.endswith(".py") for f in files)

        if has_py_files:  # Only create __init__.py for Python files
            init_file_path = os.path.join(root, "__init__.py")

            if os.path.exists(init_file_path):
                existing_count += 1
            else:
                try:
                    # Create empty __init__.py file
                    with open(init_file_path, "w") as f:
                        pass

                    created_count += 1
                    rel_path = os.path.relpath(init_file_path, directory)
                    created_files.append(rel_path)
                except Exception as e:
                    print(f"Warning: Failed to create {init_file_path}: {e}")

    if created_count > 0:
        print(f"✓ Created {created_count} new __init__.py files:")
        for file in created_files:
            print(f"  - {file}")
    else:
        print("No new __init__.py files needed to be created.")

    print(f"✓ Found {existing_count} existing __init__.py files")

    return created_count, existing_count


def main():
    """Main function to generate setup.py and requirements.txt"""
    print("=" * 60)
    print("SETUP.PY AND REQUIREMENTS.TXT GENERATOR")
    print("=" * 60)

    try:
        # Analyze project structure
        project_info = scan_directory_structure()

        # Create __init__.py files
        print("\n" + "=" * 60)
        print("CREATING __INIT__.PY FILES")
        print("=" * 60)
        create_init_files(project_info["current_dir"])

        # Ensure pipreqs is installed
        if not ensure_pipreqs_installed():
            exit_with_error(
                "Failed to install pipreqs. Cannot continue with dependency detection."
            )

        # Find all dependencies
        print("\n" + "=" * 60)
        print("DETECTING DEPENDENCIES")
        print("=" * 60)
        dependencies = find_dependencies(
            project_info["current_dir"], project_info.get("notebook_files", [])
        )

        if not dependencies:
            print("Warning: No dependencies found. The requirements will be empty.")
            dependencies = []

        # Find YAML files
        yaml_files = find_yaml_files(project_info["config_dir"])

        # Generate setup.py and requirements.txt
        print("\n" + "=" * 60)
        print("GENERATING SETUP FILES")
        print("=" * 60)
        generate_setup_py(project_info, dependencies, yaml_files)

        print("\n✅ Generation complete!")
        print("\n📋 NEXT STEPS:")
        print("1. Review the generated setup.py and requirements.txt files")
        print("2. Run 'pip install -e .' to install the package in development mode")
        if project_info.get("notebook_files"):
            print(
                "3. For Jupyter notebooks, run 'python -m ipykernel install --user --name=your-env-name'"
            )
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        exit_with_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
