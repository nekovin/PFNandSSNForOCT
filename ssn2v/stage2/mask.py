import torch

def create_blind_spot_input_fast(image, mask): # This creates an artificial situation: your network learns to reconstruct pixels from surrounding context, but the masking pattern (black dots) doesn't match the actual noise distribution in OCT images.
    blind_input = image.clone()
    #noise = torch.randn_like(image) * image.std() + image.mean()
    #blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    #blind_input = torch.where(mask.bool(), torch.zeros_like(image), image)
    blind_input[mask.bool()] = 0
    return blind_input

def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    #noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    return blind_input


def create_blind_spot_input_with_realistic_noise(image, mask):
    blind_input = image.clone()
    
    # Parameters for OCT-like noise (speckle)
    mean_level = image.mean()
    std_level = image.std() * 0.9  # Adjust based on your OCT noise level
    
    # Generate noise with speckle-like characteristics
    noise = torch.randn_like(image) * std_level + mean_level
    
    # Apply mask
    blind_input[mask.bool()] = noise[mask.bool()]
    return blind_input