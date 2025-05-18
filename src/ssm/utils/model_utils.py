import torch
from ssm.models.ssm.ssm_attention import get_ssm_model

def load_ssm_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_ssm_model(checkpoint=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model