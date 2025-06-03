"""
Configuration Module

This module manages application configuration settings for the HPLC Method Optimizer.
"""

import json
import os
from typing import Any, Dict, Optional

# Default configuration
DEFAULT_CONFIG = {
    "client_lab": "NuraxsDemo",
    "experiment": "HPLC-Optimization",
    "output_dir": "hplc_optimization",
    "default_pdf_dir": "data/pdfs",
    "ui": {
        "page_title": "HPLC Method Optimizer",
        "page_icon": "🧪",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "theme": {
            "primary_color": "#0066cc",
            "background_color": "#f0f2f6",
            "text_color": "#262730",
            "font": "sans-serif",
        },
    },
    "advanced": {
        "debug_mode": False,
        "show_developer_tools": False,
        "cache_timeout_seconds": 3600,
        "max_upload_size_mb": 50,
    },
}

# Config singleton
_config: Dict[str, Any] = {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Optional path to config file

    Returns:
        Configuration dictionary
    """
    global _config

    # Start with default config
    _config = DEFAULT_CONFIG.copy()

    # Try to load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                # Update config with file values (nested update)
                _update_nested_dict(_config, file_config)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")

    # Override with environment variables if present
    _override_from_env(_config)

    return _config


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.

    Returns:
        Configuration dictionary
    """
    global _config

    # If config is empty, load defaults
    if not _config:
        _config = load_config()

    return _config


def save_config(config_path: str) -> bool:
    """
    Save current configuration to file.

    Args:
        config_path: Path to save config file

    Returns:
        True if successful, False otherwise
    """
    global _config

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(_config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        return False


def _update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a nested dictionary with values from another dictionary.

    Args:
        d: Dictionary to update
        u: Dictionary with new values

    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d


def _override_from_env(config: Dict[str, Any]) -> None:
    """
    Override configuration with environment variables.

    Environment variables should be prefixed with HPLC_BO_
    For nested keys, use double underscore, e.g., HPLC_BO_UI__LAYOUT

    Args:
        config: Configuration dictionary to update
    """
    prefix = "HPLC_BO_"

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and split by double underscore
            config_key = key[len(prefix) :].lower()
            parts = config_key.split("__")

            # Navigate to the correct level in the config
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value, converting to appropriate type
            try:
                # Try to convert to appropriate type
                if value.lower() == "true":
                    current[parts[-1]] = True
                elif value.lower() == "false":
                    current[parts[-1]] = False
                elif value.isdigit():
                    current[parts[-1]] = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                    current[parts[-1]] = float(value)
                else:
                    current[parts[-1]] = value
            except Exception:
                # If conversion fails, use the string value
                current[parts[-1]] = value
