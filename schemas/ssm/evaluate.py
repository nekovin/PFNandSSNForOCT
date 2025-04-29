import torch
from ssm.losses.ssm_loss import custom_loss
from visualise import plot_images

from data_loading import get_loaders



def evaluate(model):
    train_loader, val_loader = get_loaders()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_inputs, batch_targets = batch
            outputs = model(batch_inputs)
            flow_component = outputs["flow_component"]
            noise_component = outputs["noise_component"]

            loss_parameters = None
            debug = False
            
            loss = custom_loss(flow_component, noise_component, batch_inputs, batch_targets, loss_parameters, debug)
            total_loss += loss.item()

            images = [batch_inputs[0][0], batch_targets[0][0], flow_component[0][0], noise_component[0][0]]
            titles = ["Input", "Target", "Flow Component", "Noise Component"]
            losses = {"Loss": loss.item()}
            plot_images(images, titles, losses)
    return total_loss / len(val_loader)

#evaluate(ssm, val_loader)