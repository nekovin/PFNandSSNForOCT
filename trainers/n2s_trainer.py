from schemas.components.train import train
from utils.config import get_config

def main():
    
    config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"

    config = get_config(config_path)

    train(config, "n2s", False)

if __name__ == "__main__":
    main()