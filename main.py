import os
import torch
from ddpm.models.unet import UNet
from ddpm.ddpm.ddpm_uncon import Diffusion
from ddpm.data_loader.landscape_loader import load_Landscape

def main():
    # Hyperparameters
    epochs = 500
    lr = 3e-4
    noise_steps = 1000
    beta_min = 1e-4
    beta_max = 2e-2
    img_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    data_path = './data'
    
    # Model
    model = UNet(
        in_channels=3,
        out_channels=3,
        time_dims=256,
        device=device,
    )
    model = model.to(device)
    
    # Data
    data_loader = load_Landscape(data_path, batch_size, img_size)
    print(f'Number of Training Images: {len(data_loader.dataset)}')
    
    # Diffusion
    diffusion = Diffusion(model, epochs, lr, noise_steps, beta_min, beta_max, img_size, device)
    diffusion.train(data_loader)
    
if __name__ == '__main__':
    main()