import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")

import torch

from ssn2v.stage2.stage_2_dataset import load_data
from ssn2v.models.enhanced_n2v_unet import get_e_n2n_unet_model
from ssn2v.stage2.train import train_stage2
from ssn2v.stage2.eval import evaluate_model

def stage2(train = True, test = False, evaluate = False):

    train_loader, val_loader, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_e_n2n_unet_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    checkpoint = torch.load(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v\stage2\checkpoints\stage1.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    if train:

        model, history = train_stage2(
            img_size=(256, 256), 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            epochs=5, 
            device=device, 
            scratch=False, 
            save_path=r'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v\stage2\checkpoints\stage2_256_final_N2NUNet_model.pth', 
            mask_ratio=0.5, 
            alpha=1.0,
            threshold=90,
            visualise=False,
            debug=True
    )
        
    if test:
        pass

    if evaluate:
        evaluate_model(model, train_loader, device)
        
    
    return model, history, test_loader