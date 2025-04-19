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