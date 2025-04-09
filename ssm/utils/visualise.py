import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm")
from postprocessing.postprocessing import normalize_image
from models.ssm_attention import SpatialAttention

def visualize_progress(model, input_tensor, target_tensor, masked_tensor, epoch):
    """
    Visualize the current model's output during training
    
    Args:
        model: Current model
        input_tensor: Input tensor [1, 1, H, W]
        target_tensor: Target tensor [1, 1, H, W]
        epoch: Current epoch number
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    def adaptive_threshold(denoised_img, sensitivity=0.02):
        # Get an estimate of noise level from background
        bg_mask = denoised_img < np.percentile(denoised_img, 50)
        noise_std = np.std(denoised_img[bg_mask])
        
        # Set threshold as a multiple of background noise
        threshold = sensitivity * noise_std
        return np.maximum(denoised_img - threshold, 0)
    
    # Get components
    input_np = input_tensor[0, 0].cpu().numpy()
    target_np = target_tensor[0, 0].cpu().numpy()
    flow_np = output['flow_component'][0, 0].cpu().numpy()
    noise_np = output['noise_component'][0, 0].cpu().numpy()
    denoised_np = input_np - noise_np
    denoised_np = adaptive_threshold(denoised_np, sensitivity=0.02)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original data
    # 
    axes[0, 0].imshow(input_np, cmap='gray')
    axes[0, 0].set_title("Input")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_np, cmap='gray')
    axes[0, 1].set_title("Target")
    axes[0, 1].axis('off')
    
    # Row 2: Model outputs
    #normalise
    #axes[1, 0].imshow(flow_np, cmap='gray')
    #
    axes[1, 0].imshow(flow_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Flow Component")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noise_np, cmap='gray')
    axes[1, 1].set_title("Noise Component")
    axes[1, 1].axis('off')
    
    #axes[1, 2].imshow(denoised_np, cmap='gray')
    #axes[1, 2].set_title("Denoised (Input - Noise)")
    #axes[1, 2].axis('off')

    axes[1, 2].imshow(masked_tensor, cmap='gray')
    axes[1, 2].set_title("Masked Tensor")
    axes[1, 2].axis('off')

    flow_np = normalize_image(flow_np)
    axes[0, 2].imshow(flow_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("Flow Component (Normalized)")
    axes[0, 2].axis('off')

    
    plt.suptitle(f"Training Progress - Epoch {epoch}")
    plt.tight_layout()
    plt.show()

def visualize_attention_maps(model, input_tensor):
    """
    Visualize attention maps from a SpeckleSeparationUNetAttention model
    
    Args:
        model: Trained SpeckleSeparationUNetAttention model
        input_tensor: Input OCT image tensor (1, C, H, W)
    """
    import matplotlib.pyplot as plt
    import torch
    
    # Set model to eval mode
    model.eval()
    
    # Register hooks to capture attention maps
    attention_maps = []
    
    def hook_fn(module, input, output):
        # For spatial attention, the output is the product of input and attention map
        # We can get the attention map by dividing output by input
        if isinstance(module, SpatialAttention):
            # Extract just the attention map 
            # The map is applied as x * attention_map, so we can derive it
            if output.sum() > 0:  # Avoid division by zero
                # Get the average over channels to visualize
                attention_map = output / (input[0] + 1e-10)
                attention_map = attention_map.mean(dim=1, keepdim=True)
                attention_maps.append(attention_map.detach().cpu())
    
    # Register hooks on spatial attention modules
    hooks = []
    # Encoder attention
    for i, attn_block in enumerate(model.encoder_attentions):
        # Access the spatial attention (second item in sequential)
        spatial_attn = attn_block[1]
        hooks.append(spatial_attn.register_forward_hook(hook_fn))
    
    # Bottleneck attention
    spatial_attn = model.bottleneck_attention[1]
    hooks.append(spatial_attn.register_forward_hook(hook_fn))
    
    # Decoder attention 
    for i, attn_block in enumerate(model.decoder_attentions):
        spatial_attn = attn_block[1]
        hooks.append(spatial_attn.register_forward_hook(hook_fn))
    
    # Final attention
    spatial_attn = model.final_attention[1]
    hooks.append(spatial_attn.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize attention maps
    fig, axes = plt.subplots(2, len(attention_maps)//2 + len(attention_maps)%2, figsize=(15, 6))
    axes = axes.flatten()
    
    # Original input image
    axes[0].imshow(input_tensor[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot attention maps
    for i, attn_map in enumerate(attention_maps):
        im = axes[i+1].imshow(attn_map[0, 0].numpy(), cmap='hot')
        axes[i+1].set_title(f'Attention {i+1}')
        axes[i+1].axis('off')
        fig.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Plot problematic area zoom (top right corner)
    fig2, axes2 = plt.subplots(1, min(4, len(attention_maps)), figsize=(15, 3))
    if len(attention_maps) <= 4:
        axes2 = [axes2]
    
    # Get the upper right corner of each attention map
    for i, attn_map in enumerate(attention_maps[-4:]):  # Just show the last few maps
        if i >= len(axes2):
            break
        # Extract top right corner (25% of width, 25% of height)
        h, w = attn_map.shape[2], attn_map.shape[3]
        corner = attn_map[0, 0, :h//4, 3*w//4:].numpy()
        
        im = axes2[i].imshow(corner, cmap='hot')
        axes2[i].set_title(f'Corner Attention {len(attention_maps)-4+i+1}')
        fig2.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    return fig, fig2, attention_maps