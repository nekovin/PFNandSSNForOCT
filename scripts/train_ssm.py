from trainers.ssm_trainer import train_speckle_separation_module_n2n
from losses.ssm_loss import custom_loss
from utils.config import get_config

config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\ssm_config.yaml"

config = get_config(config_path)

loss_names = ['custom_loss', 'mse']
loss_name = 'custom_loss'  # 'mse' or 'custom_loss'

if loss_name == 'mse':
    loss_fn = None
else:
    loss_fn = custom_loss

train_speckle_separation_module_n2n(config['training'], loss_fn, loss_name)