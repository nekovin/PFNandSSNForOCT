from schemas.components.train import train
from utils.config import get_config

def train_n2s(config_path=None, ssm=False, override_config=None):
    
    if config_path is None:
        config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"

    config = get_config(config_path, override_config)

    train(config, "n2s", ssm)

if __name__ == "__main__":
    train_n2s()