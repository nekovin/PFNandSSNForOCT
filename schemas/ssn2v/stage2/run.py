import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")

import torch


from stage2.stage_2_dataset import get_stage2_loaders
from stage2.train import train_stage2
from stage2.eval import evaluate_model

def run_stage2(dataset, config):

    # modes
    train, test, evaluate = config["train"], config["test"], config["evaluate"]

    # load data
    img_size = config["data_config"]['img_size']

    train_loader, val_loader, test_loader = get_stage2_loaders(dataset, img_size, test_split=0.2, val_split=0.15) 


    # training config
    device = config['device']
    model = config['model'].to(device)
    criterion = config['train_config']['criterion']
    optimizer = config['train_config']['optimizer']

    # model config
    stage1_checkpoint = torch.load(config["train_config"]['stage1_checkpoint_path'])
    model.load_state_dict(stage1_checkpoint['model_state_dict'])
    history = None

    if train:
        model, history = train_stage2(
            img_size=(img_size, img_size), 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            epochs=config['train_config']['num_epochs'], 
            device=device, 
            scratch=config['train_config']['scratch'], 
            save_path=config['train_config']['stage2_checkpoint_path'], 
            mask_ratio=config['train_config']['mask_ratio'], 
            alpha=config['train_config']['alpha'],
            threshold=config['train_config']['threshold'],
            visualise=config['train_config']['visualise'],
            debug=config['train_config']['debug'],
    )
        
    if test:
        pass

    if evaluate:
        evaluate_model(model, train_loader, device)
        
    
    return model, history, train_loader, val_loader, test_loader