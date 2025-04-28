from baselines.n2n.n2n import load_model
from scripts.utils import get_config
from scripts.evaluate import evaluate

import torch

def evaluate_n2n(image, config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"):
    
    config = get_config(config_path)

    config['eval']['method'] = 'n2n'

    model = load_model(config)

    return evaluate(image, model)
