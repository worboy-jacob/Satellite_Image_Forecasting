from pathlib import Path
import yaml
from typing import Dict, Any


class Config:
    """Configuration manager for wealth index calculation."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        return self.config[key]
