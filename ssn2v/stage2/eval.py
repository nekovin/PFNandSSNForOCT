import torch
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from stage2.utils import normalize_image_torch

def evaluate_model(model, train_loader, device):
    # Set model to evaluation mode
    model.eval()

    sample = next(iter(train_loader))
    raw1, raw2, octa_calc, stage1_output = sample
    raw1 = raw1.to(device)

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

    # First create a consistent mask (make sure it matches training)
    mask = torch.bernoulli(torch.full((raw1.size(0), 1, raw1.size(2), raw1.size(3)), 
                                    0.5, device=device))  # Use same mask_ratio as training

    # Create blind spot input first
    with torch.no_grad():  # No need for gradients during visualization
        blind1 = create_blind_spot_input_with_realistic_noise(raw1, mask)
        
        # Process both images through the model
        output1 = model(raw1)
        output2 = model(blind1)

        output1 = normalize_image_torch(output1.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)
        output2 = normalize_image_torch(output2.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)

    # Now visualization code
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax[0, 0].imshow(raw1.squeeze().cpu().numpy(), cmap='gray') # potentially use  norm=NoNorm()
    ax[0, 0].set_title('Raw Image')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(output1.squeeze().cpu().numpy(), cmap='gray', norm=NoNorm())
    ax[0, 1].set_title('Model Output')
    ax[0, 1].axis('off')

    ax[1, 0].imshow(blind1.squeeze().cpu().numpy(), cmap='gray')
    ax[1, 0].set_title('Blind Image')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(output2.squeeze().cpu().numpy(), cmap='gray', norm=NoNorm())
    ax[1, 1].set_title('Model Output with Blind')
    ax[1, 1].axis('off')
    plt.tight_layout()

    # save the figure
    plt.savefig('evaluation_results.png')
    plt.show()