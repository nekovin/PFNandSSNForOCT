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

def compute_octa(oct1, oct2):

    numerator = (oct1 - oct2)**2
    denominator = oct1**2 + oct2**2
    
    epsilon = 1e-10
    octa = numerator / (denominator + epsilon)
    
    return octa

import torch
import numpy as np

def threshold_octa(octa, oct, threshold):
    oct = oct.cpu().numpy()
    octa = octa.detach().cpu().numpy()
    background_mask = oct > np.percentile(oct, threshold)  # Bottom 20% of OCT values
    
    if np.sum(background_mask) > 0:  # Ensure we have background pixels
        background_mean = np.mean(oct[background_mask])
        background_std = np.std(oct[background_mask])
        
        threshold = background_mean + 2 * background_std
    else:
        # If no background pixels, use a fixed threshold
        print("No background pixels found, using fixed threshold")
        threshold = np.percentile(oct, 1)
    
    signal_mask = oct > threshold
    
    thresholded_octa = octa * signal_mask
    
    return thresholded_octa

def threshold_octa_torch(octa: torch.Tensor, oct: torch.Tensor, threshold_percent: float):
    """
    Apply a threshold to the OCTA image based on the statistics of the OCT image.
    
    Args:
        octa (torch.Tensor): The OCTA image tensor (should require gradients).
        oct (torch.Tensor): The corresponding OCT image tensor.
        threshold_percent (float): Percentile value (0-100) for initial thresholding.
        
    Returns:
        torch.Tensor: The thresholded OCTA image.
    """
    # Compute the percentile value using torch.quantile (percentile_percent/100.0)
    t_percentile = torch.quantile(oct, threshold_percent / 100.0)

    # Create the background mask for OCT pixels
    background_mask = oct > t_percentile

    # If we have background pixels, compute the mean and std of those pixels.
    if background_mask.sum() > 0:
        background_mean = torch.mean(oct[background_mask])
        background_std = torch.std(oct[background_mask])
        threshold_val = background_mean + 2 * background_std
    else:
        # If no background pixels, default to a fixed low percentile (e.g., 1st percentile)
        threshold_val = torch.quantile(oct, 0.01)
    
    # Create a signal mask based on the computed threshold_val.
    signal_mask = oct > threshold_val

    # Multiply the OCTA image with the signal mask; cast the boolean mask to float.
    thresholded_octa = octa * signal_mask.float()

    return thresholded_octa

def enhanced_differentiable_threshold_octa_torch(octa, oct, threshold_percentile=80, smoothness=3.0, enhancement_factor=1.2):
    """
    Enhanced differentiable thresholding for OCTA images with vessel structure preservation.
    
    Args:
        octa: The OCTA image tensor (computed from OCT differences)
        oct: The OCT tensor used for thresholding reference
        threshold_percentile: Percentile (0-100) to determine foreground/background
        smoothness: Controls the transition sharpness in the sigmoid (lower = smoother)
        enhancement_factor: Factor to enhance vessel structures (higher enhances vessels)
    
    Returns:
        Thresholded OCTA image with preserved vessel structures
    """
    # Get foreground/background threshold using percentile
    sorted_values, _ = torch.sort(oct.reshape(-1))
    threshold_idx = int(sorted_values.shape[0] * threshold_percentile / 100)
    threshold_value = sorted_values[threshold_idx]
    
    # Use pixels ABOVE threshold as foreground (vessels + tissue)
    # This is more appropriate for OCT where vessels appear bright
    foreground_mask = oct > threshold_value
    
    # Compute statistics for better thresholding
    if torch.sum(foreground_mask) > 0:
        foreground_pixels = oct[foreground_mask]
        foreground_mean = torch.mean(foreground_pixels)
        foreground_std = torch.std(foreground_pixels)
        
        # Set threshold at mean - std to include most vessel structures
        signal_threshold = foreground_mean - foreground_std
    else:
        # Fallback: use high percentile threshold
        fallback_idx = int(sorted_values.shape[0] * 0.95)  # 95th percentile
        signal_threshold = sorted_values[fallback_idx]
    
    # Create gradient-aware mask using softplus instead of sigmoid
    # Softplus provides a smoother transition and better preserves fine structures
    soft_mask = torch.log(1 + torch.exp(smoothness * (oct - signal_threshold)))
    
    # Normalize soft mask to [0,1] range
    if soft_mask.max() > soft_mask.min():
        soft_mask = (soft_mask - soft_mask.min()) / (soft_mask.max() - soft_mask.min())
    
    # Apply vessel enhancement by emphasizing high OCTA values
    enhanced_octa = octa * (1.0 + (octa * enhancement_factor))
    
    # Apply soft mask to the enhanced OCTA
    thresholded_octa = enhanced_octa * soft_mask
    
    # Final normalization to ensure output is well-scaled
    if thresholded_octa.max() > 0:
        thresholded_octa = thresholded_octa / thresholded_octa.max()
    
    return thresholded_octa