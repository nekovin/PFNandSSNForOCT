from matplotlib.colors import NoNorm
import matplotlib.pyplot as plt
from IPython.display import clear_output
def visualise_n2v(raw1, oct1, oct2, output1, output2, normalize_output1, octa_from_outputs, thresholded_octa, stage1_output):
    """
    Visualize the N2V process with mask overlay
    """
    clear_output(wait=True)
    
    # Create figure with specific axis layout
    fig = plt.figure(figsize=(24, 6))
    
    # Create a grid with specific positions
    grid = plt.GridSpec(3, 5, figure=fig)
    
    # Create only the axes you need
    ax1 = fig.add_subplot(grid[0, 0])
    axraw = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0])

    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[2, 1])
    axnorm = fig.add_subplot(grid[1, 1])

    ax5 = fig.add_subplot(grid[1, 2])
    ax6 = fig.add_subplot(grid[1, 3])
    ax7 = fig.add_subplot(grid[1, 4])
    ax8 = fig.add_subplot(grid[0, 3])
    ax9 = fig.add_subplot(grid[0, 4])
    
    # Plot on individual axes
    ax1.imshow(oct1.squeeze(), cmap='gray', norm=NoNorm())
    ax1.axis('off')
    ax1.set_title('OCT1')

    axraw.imshow(raw1.squeeze(), cmap='gray')
    axraw.axis('off')
    axraw.set_title('Raw1 (norm)')
    
    ax2.imshow(oct2.squeeze(), cmap='gray', norm=NoNorm())
    ax2.axis('off')
    ax2.set_title('OCT2')
    
    ax3.imshow(output1.squeeze(), cmap='gray')
    ax3.axis('off')
    ax3.set_title('Output1')

    ax4.imshow(output2.squeeze(), cmap='gray', norm=NoNorm())
    ax4.axis('off')
    ax4.set_title('Output2')

    axnorm.imshow(normalize_output1.squeeze(), cmap='gray')
    axnorm.axis('off')
    axnorm.set_title('Normalized Output1')

    ax5.imshow(octa_from_outputs.squeeze(), cmap='gray', norm=NoNorm())
    ax5.axis('off')
    ax5.set_title('Octa from outputs')

    ax6.imshow(thresholded_octa.squeeze(), cmap='gray')
    ax6.axis('off')
    ax6.set_title('Thresholded OCTA (normalised)')

    ax7.imshow(stage1_output.squeeze(), cmap='gray')
    ax7.axis('off')
    ax7.set_title('Stage 1 Output (normalised)')

    ax8.imshow(stage1_output.squeeze(), cmap='gray', norm=NoNorm())
    ax8.axis('off')
    ax8.set_title('Stage 1 Output')

    ax9.imshow(thresholded_octa.squeeze(), cmap='gray', norm=NoNorm())
    ax9.axis('off')
    ax9.set_title('Thresholded OCTA')

    plt.tight_layout()
    plt.show()

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()