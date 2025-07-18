
"""
Configuration loader for the multimodal mental health detection system.
"""

import yaml
import os
from typing import Dict, Any
import logging

class Config:
    """Configuration class to load and manage project settings."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'models.text.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config_dict = self.config

        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]

        config_dict[keys[-1]] = value

    def save(self, path: str = None) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration (defaults to original path)
        """
        if path is None:
            path = self.config_path

        with open(path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

# Global configuration instance
config = Config()
