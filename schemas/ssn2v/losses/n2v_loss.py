import torch
import torch.nn as nn

    
class Noise2VoidLoss(nn.Module):
    def __init__(self):
        super(Noise2VoidLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        """
        When using proper blind-spot convolutions, every output pixel is predicted 
        without seeing the corresponding input pixel.
        
        Args:
            pred: Model predictions
            target: Original unmodified images
        """
        # Calculate MSE across all pixels
        loss = self.mse(pred, target)
        
        # Return mean - all pixels were processed through blind-spot convolutions
        return loss.mean()