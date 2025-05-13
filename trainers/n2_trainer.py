from schemas.components.train import train
from utils.config import get_config

def train_n2n(config_path=None, ssm=False, override_args=None):
    
    if config_path is None:
        # Default path to the configuration file
        config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"
        
    config = get_config(config_path, override_args)

    train(config, "n2n", ssm)

def train_n2s(config_path=None, ssm=False, override_config=None):
    
    if config_path is None:
        config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"

    config = get_config(config_path, override_config)

    train(config, "n2s", ssm)

def train_n2v(config_path=None, ssm=False, override_config=None):
    
    if config_path is None:
        # Default path to the configuration file
        config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"

    config = get_config(config_path, override_config)

    train(config, "n2v", ssm)