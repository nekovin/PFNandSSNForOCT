from .data import load_stage_1_data
from .train import train_stage1
from .utils import get_stage1_loaders, normalize_image

'''
model = get_e_n2n_unet_model()

stage1_config = {
    "mode" : "train",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": model,
    "model_config": {
        "input_channels": 1,
        "output_channels": 1,
        "num_filters": 32,
        "num_layers": 4,
        "kernel_size": 3,
        "activation": "relu",
        "dropout_rate": 0.5,
    },
    "train_config": {
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_epochs": 1,
        "loss_function": "mse",
        "optimizer": torch.optim.Adam(model.parameters(), lr=1e-3),
        "scheduler": "step_lr",
        "scheduler_step_size": 10,
        "criterion": torch.nn.MSELoss(),
        "early_stopping": True,
        "early_stopping_patience": 5,
        "early_stopping_delta": 0.01,
        "checkpoint_path": rf'checkpoints/stage1_stage1_model.pth',
        "scratch": False,
        "visualise": True,
        "mask_ratio": 0.5,
    },
    "data_config": {
        "regular" : True, # type of preprocessing
        "img_size": (256), #square img
        "num_patients" : 2, # minimum of 2 
        "img_per_patient" : 50,
        "train_data_path": "./data/train",
        "val_data_path": "./data/val",
        "test_data_path": "./data/test"
    },
}
'''

def run_stage1(stage1_config): 

    train, test, evaluate = stage1_config["train"], stage1_config["test"], stage1_config["evaluate"]

    img_size = stage1_config["data_config"]['img_size']
    raw_data, octa_data, dataset, name = load_stage_1_data(
       num_patients = stage1_config["data_config"]['num_patients'],
       img_per_patient = stage1_config["data_config"]['img_per_patient'])

    device = stage1_config['device']
    model = stage1_config['model'].to(device)
    criterion = stage1_config['train_config']['criterion']
    optimizer = stage1_config['train_config']['optimizer']
    #scheduler =  stage1_config['train_config']['scheduler']
    checkpoint_path = stage1_config["train_config"]['stage1_checkpoint_path']
    num_epochs = stage1_config['train_config']['num_epochs']
    visualise = stage1_config['train_config']['visualise']
    scratch = stage1_config['train_config']['scratch']
    mask_ratio = stage1_config['train_config']['mask_ratio']

    normalised_flow_masks = [normalize_image(flow_mask) for flow_mask in octa_data]
    train_loader, val_loader, test_loader = get_stage1_loaders(normalised_flow_masks, img_size)

    history = {}
    
    if train:
      model, history = train_stage1(
          img_size, model, train_loader, val_loader, 
          criterion, optimizer, 
          num_epochs, device, scratch, 
          save_path=checkpoint_path,
            mask_ratio=mask_ratio, 
            visualise=visualise)

    return model, history, train_loader, val_loader, test_loader