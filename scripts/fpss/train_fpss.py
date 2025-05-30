from fpss.trainers import train_fpss
from fpss.losses.ssm_loss import custom_loss
from fpss.utils.config import get_config
import os
import torch
def mse_loss(y_true, y_pred):

    return torch.mean((y_true - y_pred) ** 2)

def main():
    
    config_path = os.environ.get("FPSS_CONFIG_PATH")

    config = get_config(config_path)

    loss_name = config['training']['criterion']

    if loss_name == 'mse':
        loss_fn = mse_loss
    else:
        loss_fn = custom_loss

    print(config['training'])

    train_fpss(config['training'], loss_fn, loss_name)

if __name__ == "__main__":
    main()