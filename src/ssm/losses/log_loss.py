import torch

def lognormal_consistency_loss(denoised, noisy, epsilon=1e-6): # based on noise distribution

    denoised_safe = torch.clamp(denoised, min=epsilon)
    noisy_safe = torch.clamp(noisy, min=epsilon)
    
    ratio = noisy_safe / denoised_safe
    
    ratio_safe = torch.clamp(ratio, min=epsilon, max=10.0)

    log_ratio = torch.log(ratio_safe)
    
    mu = torch.mean(log_ratio)
    sigma = torch.std(log_ratio)
    
    if torch.isnan(mu) or torch.isnan(sigma):
        return torch.tensor(0.0, device=denoised.device, requires_grad=True)

    expected_mu = 0.0
    expected_sigma = 0.5 

    loss = torch.abs(mu - expected_mu) + torch.abs(sigma - expected_sigma)
    return loss