# src/utils/validation.py
from src.utils.paths import get_project_root, get_configs_dir, get_dhs_dir


def validate_data_structure():
    """Validate that required data directories and files exist."""
    project_root = get_project_root()

    required_paths = [
        get_configs_dir() / "config.yaml",
        get_dhs_dir() / "Ghana_Data",
        get_dhs_dir() / "Senegal_Data",
    ]

    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))

    if missing_paths:
        raise FileNotFoundError(f"Required paths not found: {', '.join(missing_paths)}")
