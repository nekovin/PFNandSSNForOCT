
import torch

def octa_criterion(computed_octa, target_octa):
    """
    Compute the loss between computed OCTA and target OCTA images
    
    Args:
        computed_octa: OCTA computed from denoised OCT
        target_octa: Target OCTA (from stage1 output)
    """
    # Use L2 norm as in the SSN2V paper
    loss = torch.mean((computed_octa - target_octa) ** 2)
    return loss