import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_images(images):
    # clear output
    clear_output(wait=True)
    cols = len(images)
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15,5))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()