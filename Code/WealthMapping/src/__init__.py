# wealth_mapping/__init__.py
from .core.grid_processor import GridProcessor
from .data_processing.wealth_processor import WealthProcessor
from .visualization.plots import WealthMapVisualizer

__version__ = "1.0.0"

__all__ = [
    "GridProcessor",
    "WealthProcessor",
    "BoundaryLoader",
    "WealthMapVisualizer",
]
