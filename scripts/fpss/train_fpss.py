from fpss.trainers import train_speckle_separation_module
from fpss.losses.ssm_loss import custom_loss
from fpss.utils.config import get_config
import os
#from ssm.losses.mse import mse_loss
import torch
def mse_loss(y_true, y_pred):

    return torch.mean((y_true - y_pred) ** 2)

def main():
    
    config_path = os.environ.get("SSM_CONFIG_PATH")

    config = get_config(config_path)

    loss_name = config['training']['criterion']

    if loss_name == 'mse':
        loss_fn = mse_loss
    else:
        loss_fn = custom_loss

    print(config['training'])

    train_speckle_separation_module(config['training'], loss_fn, loss_name)

if __name__ == "__main__":
    main()