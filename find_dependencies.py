#!/usr/bin/env python3
"""
Dependencies and Setup.py Generator using pipreqs

This script analyzes the current codebase structure, identifies dependencies using pipreqs,
and generates appropriate setup.py and requirements.txt files.
"""

import os
import sys
from pathlib import Path
import subprocess


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

    return {
        "project_name": project_name,
        "current_dir": current_dir,
        "src_dir": src_dir,
        "main_file": main_file,
        "config_dir": config_dir,
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
            import subprocess

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


def find_dependencies_with_pipreqs(project_dir):
    """
    Use pipreqs to analyze imports and generate requirements.

    Args:
        project_dir: Path to the project directory

    Returns:
        List of dependency names found
    """
    import subprocess
    import tempfile

    print(f"Analyzing dependencies in {project_dir} using pipreqs...")

    # Create a temporary file to store the requirements
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Run pipreqs to generate requirements file in the temp location
        cmd = [
            sys.executable,
            "-m",
            "pipreqs.pipreqs",
            str(project_dir),
            "--savepath",
            temp_path,
            "--force",
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Read the generated requirements file
        with open(temp_path, "r") as f:
            requirements = [
                line.strip().split("==")[0]  # Remove version specifiers
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]

        # Exclude pipreqs itself and normalize package names
        excluded_packages = ["pipreqs", "docopt"]  # docopt is a dependency of pipreqs
        normalized_requirements = []

        for req in requirements:
            # Normalize case (e.g., pyyaml vs PyYAML)
            req_lower = req.lower()

            # Skip excluded packages
            if req_lower in [p.lower() for p in excluded_packages]:
                continue

            # Handle PyYAML duplication (pyyaml vs. PyYAML)
            if req_lower == "pyyaml" and any(
                r.lower() == "pyyaml" for r in normalized_requirements
            ):
                continue

            normalized_requirements.append(req)

        print(f"✓ pipreqs found {len(normalized_requirements)} dependencies:")
        if normalized_requirements:
            print(", ".join(normalized_requirements))

        return normalized_requirements

    except subprocess.CalledProcessError as e:
        print(f"ERROR: pipreqs failed: {e}")
        print(f"Command output: {e.stdout}\n{e.stderr}")
        return []
    except Exception as e:
        print(f"ERROR: Unexpected error running pipreqs: {e}")
        return []
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


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

    # Python version
    python_version = f">={sys.version_info.major}.{sys.version_info.minor}"

    # Format dependencies for setup.py
    deps_str = "[" + ", ".join(f'"{dep}"' for dep in dependencies) + "]"

    # Format YAML files for package_data if any
    if yaml_files:
        yaml_files_str = [f'"{file}"' for file in yaml_files]
        package_data_str = f'{{"": [{", ".join(yaml_files_str)}]}}'
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
        if str(rel_src) == ".":
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
    }

    created_count = 0
    existing_count = 0
    created_files = []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

        # Check if directory contains Python files
        has_py_files = any(f.endswith(".py") for f in files)

        if has_py_files:
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

        # Find dependencies
        print("\n" + "=" * 60)
        print("DETECTING DEPENDENCIES")
        print("=" * 60)
        dependencies = find_dependencies_with_pipreqs(project_info["current_dir"])

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
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        exit_with_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
