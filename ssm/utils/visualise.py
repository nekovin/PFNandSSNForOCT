
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