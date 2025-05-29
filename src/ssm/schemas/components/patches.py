def extract_patches(image, patch_size=64, stride=32):
    """Extract patches from an image with given patch size and stride."""
    stride = patch_size // 4
    # Handle different image formats
    if len(image.shape) == 4:  # (B, C, H, W)
        b, c, h, w = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension
        b, c, h, w = image.shape
    elif len(image.shape) == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        b, c, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    all_patches = []
    all_locations = []
    
    for i in range(b):
        img_patches = []
        img_locations = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[i, :, y:y+patch_size, x:x+patch_size]
                img_patches.append(patch)
                img_locations.append((y, x))
        
        all_patches.extend(img_patches)
        all_locations.extend([(i, y, x) for y, x in img_locations])
    
    return torch.stack(all_patches), all_locations

def reconstruct_from_patches(patches, locations, image_shape, patch_size=64):
    """Reconstruct an image from patches based on their locations."""
    # Handle different shape formats
    if len(image_shape) == 4:  # (B, C, H, W)
        b, c, h, w = image_shape
    elif len(image_shape) == 3:  # (C, H, W)
        c, h, w = image_shape
        b = 1
    else:
        raise ValueError(f"Unexpected image shape: {image_shape}")
    
    # Check location format
    sample_location = locations[0]
    if len(sample_location) == 3:  # (batch_idx, y, x)
        # Need to group by batch
        batch_reconstructed = []
        
        # Group patches by batch index
        batch_patches = [[] for _ in range(b)]
        batch_locations = [[] for _ in range(b)]
        
        for i, (patch, location) in enumerate(zip(patches, locations)):
            batch_idx, y, x = location
            batch_patches[batch_idx].append(patch)
            batch_locations[batch_idx].append((y, x))
        
        # Reconstruct each batch item
        for i in range(b):
            if len(batch_patches[i]) > 0:
                # Call recursively with simpler locations
                reconstructed = reconstruct_from_patches(
                    torch.stack(batch_patches[i]), 
                    batch_locations[i],
                    (c, h, w),  # Single image shape
                    patch_size
                )
                batch_reconstructed.append(reconstructed)
            else:
                # Empty tensor if no patches for this batch
                batch_reconstructed.append(torch.zeros((c, h, w), device=patches[0].device))
        
        return torch.stack(batch_reconstructed)
    
    elif len(sample_location) == 2:  # (y, x)
        # Simple reconstruction for a single image
        reconstructed = torch.zeros((c, h, w), device=patches[0].device)
        weights = torch.zeros((h, w), device=patches[0].device)
        
        for patch, (y, x) in zip(patches, locations):
            reconstructed[:, y:y+patch_size, x:x+patch_size] += patch
            weights[y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping regions
        weights = weights.unsqueeze(0).repeat(c, 1, 1)
        weights[weights == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / weights
        
        return reconstructed
    
    else:
        raise ValueError(f"Unexpected location format: {sample_location}")
