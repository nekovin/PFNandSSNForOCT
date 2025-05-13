import torch
from ssm.losses import custom_loss
from ssm.utils import plot_images
from ssm.data import get_paired_loaders
from ssm.losses.ssm_loss import custom_loss
from ssm.models import get_ssm_model
from ssm.utils import normalize_image
import torch


def evaluate2(train_config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_loss_trained_path = train_config['load_checkpoint'].format(loss_fn=loss_name)
    model = get_ssm_model(checkpoint=custom_loss_trained_path)

    checkpoint_path = train_config['load_checkpoint'].format(loss_fn=loss_name)  
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    train_loader, val_loader = get_loaders(start=25)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_inputs, batch_targets = batch
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            flow_component = outputs["flow_component"]
            noise_component = outputs["noise_component"]

            loss_parameters = None
            debug = False
            
            loss = custom_loss(flow_component, noise_component, batch_inputs, batch_targets, loss_parameters, debug)
            total_loss += loss.item()

            # convert to numpy for visualization
            batch_inputs = batch_inputs.cpu().numpy()
            batch_targets = batch_targets.cpu().numpy()
            flow_component = flow_component.cpu().numpy()
            noise_component = noise_component.cpu().numpy()
            #flow_normalised = (flow_component - flow_component.min()) / (flow_component.max() - flow_component.min())
            flow_normalised = normalize_image(flow_component[0][0])

            images = [batch_inputs[0][0], batch_targets[0][0], flow_component[0][0], noise_component[0][0], flow_normalised]
            titles = ["Input", "Target", "Flow Component", "Noise Component", "Flow Normalised"]
            losses = {"Loss": loss.item()}
            plot_images(images, titles, losses)

            break
    return total_loss / len(val_loader)

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

