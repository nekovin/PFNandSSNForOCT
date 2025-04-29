def get_sample_image(dataloader, device):
    sample = next(iter(dataloader))
    image = sample[0].to(device)
    return image

def denoise_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        denoised_image = model(image)
    return denoised_image['flow_component'].squeeze(0).squeeze(0).cpu().numpy()

from ssm.postprocessing.postprocessing import normalize_image

def plot_sample(image, denoised):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    normalised_denoised = normalize_image(denoised)

    ax[1].imshow(denoised, cmap='gray')
    ax[1].set_title('Denoised Image')
    ax[1].axis('off')

    ax[2].imshow(normalised_denoised, cmap='gray')
    ax[2].set_title('Normalized Denoised Image')
    ax[2].axis('off')

    plt.show()


def eval():
    
    model = create_progressive_fusion_dynamic_unet(base_features=32, use_fusion=True).to(device)
    checkpoint = torch.load(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\phf\src\checkpoints\512_checkpoint_epoch_14.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    image = get_sample_image(val_loader, device)
    denoised = denoise_image(model, image, device)
    #denoised = denoised.squeeze(0).squeeze(0).cpu().numpy()
    denoised = normalize_image(denoised)

    image = image.squeeze(0).squeeze(0).cpu().numpy()

    image = image[0][0]

    denoised = denoised[0][0]

    #base_sample_image = get_sample_image(val_loader, device)
    #denoised = denoise_image(model, base_sample_image, device='cuda')[0][0]
    #sample_image = base_sample_image.cpu().numpy()[0][0]
    #denoised = denoised.cpu().numpy()
    #normalised_denoised = normalize_image(denoised)
    plot_sample(image, denoised)
    metrics = evaluate_oct_denoising(image, denoised)
    print(metrics.keys())
    print(metrics.values())

def save_results():
    os.makedirs('output', exist_ok=True)

    import matplotlib.pyplot as plt
    for i, (image, _) in enumerate(train_loader):
        image = image.to(device)
        #denoised_with_speckle = denoise_image(model, image, device='cuda')[0][0]
        #denoised_with_speckle = denoised_with_speckle.cpu().numpy()
        plt.imsave(f'output/denoised_image_{i}.png', denoised, cmap='gray')
        
        if i >= 20:
            break