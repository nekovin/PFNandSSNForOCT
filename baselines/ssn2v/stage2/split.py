
import torch

def differentiable_threshold_octa_torch(octa, oct, threshold_percentile, smoothness=5.0):
    # Get threshold value via percentile
    sorted_values, _ = torch.sort(oct.reshape(-1))
    threshold_idx = int(sorted_values.shape[0] * threshold_percentile / 100)
    threshold_value = sorted_values[threshold_idx]
    
    # Get background statistics
    background_mask = oct > threshold_value
    if torch.sum(background_mask) > 0:
        background_pixels = oct[background_mask]
        background_mean = torch.mean(background_pixels)
        background_std = torch.std(background_pixels)
        signal_threshold = background_mean + 2 * background_std
    else:
        fallback_idx = int(sorted_values.shape[0] * 0.01)
        signal_threshold = sorted_values[fallback_idx]
    
    # Create a soft mask using sigmoid instead of hard threshold
    # The smoothness parameter controls the sharpness of the transition
    soft_mask = torch.sigmoid(smoothness * (oct - signal_threshold))
    
    # Apply soft mask to preserve gradients
    thresholded_octa = octa * soft_mask
    
    return thresholded_octa