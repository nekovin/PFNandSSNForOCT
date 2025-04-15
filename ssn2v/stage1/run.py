from .data import load_stage_1_data
from .train import train_stage1
from .utils import get_stage1_loaders, normalize_image

def run_stage1(stage1_config): 

    img_size = stage1_config["data_config"]['img_size']
    raw_data, octa_data, dataset, name = load_stage_1_data(
       num_patients = stage1_config["data_config"]['num_patients'],
       img_per_patient = stage1_config["data_config"]['img_per_patient'])

    device = stage1_config['device']
    model = stage1_config['model'].to(device)
    criterion = stage1_config['train_config']['criterion']
    optimizer = stage1_config['train_config']['optimizer']
    #scheduler =  stage1_config['train_config']['scheduler']
    checkpoint_path = stage1_config["train_config"]['checkpoint_path']
    num_epochs = stage1_config['train_config']['num_epochs']
    visualise = stage1_config['train_config']['visualise']
    scratch = stage1_config['train_config']['scratch']
    mask_ratio = stage1_config['train_config']['mask_ratio']

    normalised_flow_masks = [normalize_image(flow_mask) for flow_mask in octa_data]
    train_loader, val_loader, test_loader = get_stage1_loaders(normalised_flow_masks, img_size)
    
    if stage1_config['mode'] == 'train':
      model, history = train_stage1(
          img_size, model, train_loader, val_loader, 
          criterion, optimizer, 
          num_epochs, device, scratch, 
          save_path=checkpoint_path,
            mask_ratio=mask_ratio, 
            visualise=visualise)

    return model, history, train_loader, val_loader, test_loader