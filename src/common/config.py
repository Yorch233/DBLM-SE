"""
Author: Yorch233
GitHub: https://github.com/Yorch233/DBLM-SE
Email: qyao@stmail.ujs.edu.cn
Date: 2025-08-05
"""

import inspect
import os
import os.path
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate.logging import get_logger
from tabulate import tabulate

# Configure YAML to preserve dictionary order when dumping
represent_dict_order = lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)

logger = get_logger(__name__)


class BaseConfiguer:
    """
    Base configuration handler for loading and managing YAML configuration files.
    
    Provides functionality for loading YAML files with inheritance support,
    allowing configurations to extend and override parent configurations.
    """

    def __init__(self):
        """Initialize the base configuration handler."""
        pass

    @classmethod
    def load(cls, file_path: str):
        """
        Load and parse a YAML configuration file with inheritance support.
        
        This method recursively loads parent configurations specified in the 'inherit' field
        and merges them with the current configuration, allowing for hierarchical config management.
        
        Args:
            file_path (str): Path to the YAML configuration file to load.
            
        Returns:
            OrderedDict: Parsed configuration dictionary with inheritance applied.
            
        Raises:
            ValueError: If file_path is not a YAML file or inheritance configuration is invalid.
        """
        if '.yaml' not in file_path and '.yml' not in file_path:
            raise ValueError(f'The value of `file_path` should be a path to a yaml file.')
        root_path = os.path.dirname(os.path.abspath(file_path))

        def get_yaml_data(path: str):
            """Read and parse YAML file content."""
            with open(path, encoding='utf-8') as file:
                return yaml.safe_load(file.read())

        def overwrite(ori: dict, con: dict) -> dict:
            """
            Recursively merge two dictionaries, with original values taking precedence.
            
            Args:
                ori (dict): Original dictionary with higher priority values.
                con (dict): Configuration dictionary to be merged.
                
            Returns:
                dict: Merged dictionary with proper precedence handling.
            """
            _con = OrderedDict()
            _con.update(con)
            for k, v in ori.items():
                if isinstance(v, dict) and k in _con:
                    _con[k] = overwrite(ori[k], _con[k])
                else:
                    _con[k] = v
            return _con

        # Load the main configuration file
        _origin = get_yaml_data(file_path)
        _config = OrderedDict()
        
        # Handle inheritance if specified
        if 'inherit' in _origin:
            inherit_ = _origin['inherit']
            if isinstance(inherit_, str) or isinstance(inherit_, list):
                # Normalize to list format
                if isinstance(inherit_, str):
                    inherit_paths = [inherit_]
                else:
                    inherit_paths = inherit_
                
                # Load and merge parent configurations
                for inherit_path in inherit_paths:
                    if not os.path.isabs(inherit_path):
                        inherit_path = os.path.join(root_path, inherit_path)
                    _config.update(cls.load(inherit_path))
            else:
                raise TypeError(
                    f'The field of `inherit` in "{os.path.abspath(file_path)}` should be a string or dictionary.')
        
        # Merge current configuration with inherited configuration
        for key, value in _origin.items():
            if key not in 'inherit':
                if isinstance(value, dict) and key in _config:
                    _config[key] = overwrite(_origin[key], _config[key])
                else:
                    _config[key] = value
        return _config

    @classmethod
    def dump(cls, data: any, output_path: str):
        """
        Dump configuration data to a YAML file.
        
        Args:
            data (any): Configuration data to be written to file.
            output_path (str): Path where the YAML file should be saved.
        """
        with open(output_path, "w", encoding='utf-8') as fo:
            yaml.dump(data, fo, default_flow_style=False)


def read_yml(yml_path):
    """
    Read and parse a YAML configuration file.
    
    Args:
        yml_path (str): Path to the YAML file to read.
        
    Returns:
        OrderedDict: Parsed configuration data.
    """
    yml = BaseConfiguer.load(yml_path)
    return yml


def read_config_from_yaml(config_path: str):
    """
    Read configuration from YAML file and convert to Config object.
    
    This function handles various input formats and validates the configuration file.
    
    Args:
        config_path (str): Path to configuration file or directory containing config.yml.
        
    Returns:
        Config: Configuration object with loaded parameters.
        
    Raises:
        ValueError: If config_path is invalid or file doesn't exist.
    """
    # Handle directory input by looking for config.yml
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.yml')
    
    # Validate file extension
    if not config_path.endswith('yml') and not config_path.endswith('yaml'):
        raise ValueError(f'The value of `config_path` should be a path to a yaml file, not \'{config_path}\'.')
    
    # Check file existence
    if not os.path.exists(config_path):
        raise ValueError(f'The config file `{config_path}` does not exist.')

    # Load and convert to Config object
    config = read_yml(config_path)
    return Config(config)


class Config:
    """
    Configuration wrapper class that provides convenient access to configuration parameters.
    
    Wraps dictionary-based configuration data with additional utility methods for
    printing, saving, and accessing configuration values.
    """
    
    _MAX_LENGTH = 50  # Maximum length for value display in print method

    def __init__(self, config):
        """
        Initialize configuration object.
        
        Args:
            config (dict): Dictionary containing configuration parameters.
        """
        assert isinstance(config, dict), "Config must be a dictionary."
        self.__dict__.update(config)

    def dict(self):
        """
        Get configuration as dictionary, excluding private attributes.
        
        Returns:
            dict: Configuration parameters without private attributes.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def get(self, field, default=None):
        """
        Get configuration field value with default fallback.
        
        Args:
            field (str): Name of the configuration field.
            default (any): Default value if field doesn't exist.
            
        Returns:
            any: Field value or default value.
        """
        return getattr(self, field, default)

    def update(self, config):
        """
        Update configuration with new values.
        
        Args:
            config (dict): Dictionary of new configuration values to update.
        """
        self.__dict__.update(config)

    def save(self, save_path=None, file_name='config.yml'):
        """
        Save current configuration to YAML file.
        
        Args:
            save_path (str, optional): Directory to save configuration. Defaults to self.run_path.
            file_name (str): Name of the configuration file to save.
        """
        if save_path is None:
            save_path = self.run_path
        BaseConfiguer.dump(data=self.dict(), output_path=os.path.join(save_path, file_name))

    def handleOvergLength(self, sentence: str, max_length: int) -> dict:
        """
        Truncate long strings for better display formatting.
        
        Args:
            sentence (str): String to potentially truncate.
            max_length (int): Maximum allowed string length.
            
        Returns:
            str: Truncated string with ellipsis if needed.
        """
        sentence = sentence if len(sentence) < max_length else sentence[:max_length - 1 - 3] + '...'
        return sentence

    def print(self):
        """
        Print configuration in a formatted table for easy viewing.
        """
        con = self.dict()
        table_data = []
        for key, value in con.items():
            table_data.append([str(key), self.handleOvergLength(str(value), self._MAX_LENGTH)])
        print('Configuration:')
        print(tabulate(table_data, headers=["Param", "Value"], tablefmt="pretty"))


def config_from_yaml(config_path: str, key_value: str = None):
    """
    Decorator factory for injecting YAML configuration parameters into class constructors.
    
    This decorator automatically injects configuration values from YAML files into
    class __init__ methods, handling type conversion and parameter validation.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        key_value (str, optional): Specific key in YAML to use as configuration source.
        
    Returns:
        function: Decorator function that modifies class constructor.
    """

    def decorator(cls):
        """
        Decorator that modifies class constructor to use YAML configuration.
        
        Args:
            cls: Class to be decorated.
            
        Returns:
            cls: Modified class with updated constructor.
        """
        # Read configuration file
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"YAML file {config_path} not found")

        configs = read_yml(config_path)

        # Extract specific configuration section if requested
        if key_value is not None:
            configs = configs[key_value]

        # Get class constructor signature for parameter validation
        init_signature = inspect.signature(cls.__init__)
        parameters = init_signature.parameters

        # Generate final default parameters from YAML configuration
        final_defaults = {}
        for name, param in parameters.items():
            if name == 'self':
                continue
            if name in configs:
                # Convert configuration value to expected type
                final_defaults[name] = _parse_config_value(param.annotation, configs[name])

        # Modify class constructor to use YAML configuration
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            """
            Modified constructor that merges YAML config with explicit parameters.
            
            Parameter precedence: Explicit arguments > YAML config > Original defaults
            """
            # Merge parameters with proper precedence
            merged_kwargs = {}

            # Get original default values
            for name, param in parameters.items():
                if param.default != inspect.Parameter.empty and name not in merged_kwargs:
                    merged_kwargs[name] = param.default

            # Apply YAML configuration values
            merged_kwargs.update(final_defaults)
            # Apply explicitly passed arguments (highest priority)
            merged_kwargs.update(kwargs)

            # Validate required parameters are provided
            required_params = [
                name for name, param in parameters.items()
                if param.default == inspect.Parameter.empty and name not in ('self')
            ]
            missing = [p for p in required_params if p not in merged_kwargs]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")

            # Execute original constructor with merged parameters
            original_init(self, *args, **merged_kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


def _parse_config_value(expected_type: Any, value: Any) -> Any:
    """
    Safely convert configuration value to expected type.
    
    Handles special type conversions for common configuration types.
    
    Args:
        expected_type (Any): Expected type annotation from constructor.
        value (Any): Raw configuration value from YAML.
        
    Returns:
        Any: Type-converted configuration value.
    """
    if expected_type is torch.device:
        return torch.device(value)
    if inspect.isclass(expected_type) and issubclass(expected_type, dict):
        return dict(value)  # Ensure dictionary is serializable
    return expected_type(value) if value is not None else None