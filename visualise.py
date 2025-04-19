import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchviz import make_dot

def plot_images(images, titles, losses):
    # clear output
    clear_output(wait=True)
    cols = len(images)
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15,5))
    loss_titles = []
    for loss_name, loss_value in losses.items():
        loss_titles.append(f"{loss_name}: {loss_value:.4f}")
    combined_title = " | ".join(loss_titles)
    fig.suptitle(combined_title, fontsize=16)
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_computation_graph(model, loss, speckle_module):

    # Create visualization
    dot = make_dot(loss, params=dict(list(model.named_parameters()) + list(speckle_module.named_parameters())))
    dot.render('results/computation_graph', format='png')