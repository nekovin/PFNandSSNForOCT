from schemas.components.train import train
from utils.config import get_config

def train_n2n(config_path=None, ssm=False):
    
    if config_path is None:
        # Default path to the configuration file
        config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"
        
    config = get_config(config_path)

    train(config, "n2n", ssm)

if __name__ == "__main__":
    train_n2n()