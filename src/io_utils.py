"""
Centralized JSON I/O utilities.

Provides consistent interfaces for loading and writing JSON files across
analysis pipelines, reducing boilerplate and improving maintainability.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Parsed JSON data as dictionary
    
    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Dictionary to write
        path: Path to output JSON file
        indent: JSON indentation level (default: 2)
    
    Creates parent directories if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)
