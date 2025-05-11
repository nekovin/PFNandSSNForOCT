import yaml
import torch
import torch.nn as nn

def parse_config(config):
    """Parse configuration values, converting strings to appropriate objects."""
    parsed_config = {}
    
    for section, values in config.items():
        parsed_config[section] = {}
        for key, value in values.items():
            if key == 'learning_rate' and isinstance(value, str):
                if 'e' in value.lower():
                    parsed_config[section][key] = float(value)
                else:
                    parsed_config[section][key] = float(value)
            
            elif key == 'criterion' and isinstance(value, str):
                if value == 'nn.MSELoss()':
                    parsed_config[section][key] = nn.MSELoss()
                elif value == 'nn.L1Loss()':
                    parsed_config[section][key] = nn.L1Loss()
                else:
                    parsed_config[section][key] = value
            
            elif key == 'optimizer' and isinstance(value, str):
                parsed_config[section][key] = value

            
            else:
                parsed_config[section][key] = value
    
    return parsed_config

def get_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return parse_config(config)

##########


import yaml
import torch
import torch.nn as nn

def parse_config(config, override_args=None):

    parsed_config = {}
    
    for section, values in config.items():
        parsed_config[section] = {}
        for key, value in values.items():
            parsed_config[section][key] = _convert_value(key, value)

    if override_args:
        for section, values in override_args.items():
            if section not in parsed_config:
                parsed_config[section] = {}
            
            for key, value in values.items():
                parsed_config[section][key] = _convert_value(key, value)
    
    return parsed_config

def _convert_value(key, value):
    """Helper function to convert configuration values to appropriate types."""
    if key == 'learning_rate' and isinstance(value, str):
        return float(value)
    
    elif key == 'criterion' and isinstance(value, str):
        if value == 'nn.MSELoss()':
            return nn.MSELoss()
        elif value == 'nn.L1Loss()':
            return nn.L1Loss()
        else:
            return value
    
    elif key == 'optimizer' and isinstance(value, str):
        return value
    
    else:
        return value

def get_config(config_path, override_args=None):
    """
    Load configuration from a YAML file with optional overrides.
    
    Args:
        config_path (str): Path to YAML configuration file
        override_args (dict, optional): Dictionary of override arguments
    
    Returns:
        dict: Parsed configuration with any overridden values
    """
    with open(config_path, 'r') as file:
        #print(f"Loading configuration from {config_path}")
        config = yaml.safe_load(file)
    
    return parse_config(config, override_args)