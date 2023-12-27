import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

class Diffusion(object):
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 3e-4,
        noise_steps: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        img_size: int = 256,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        results_dir: str = 'results',
        name: str = 'diffusion',
    ):
        self.model = model
        self.epochs = epochs
        self.noise_steps = noise_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.img_size = img_size
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.noise_steps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.results_dir = os.path.join(results_dir, name)
        if os.path.exists(self.results_dir):
            os.system(f'rm -rf {self.results_dir}/*')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'models'), exist_ok=True)
        
    def sample_timestep(self, batch_size):
        return torch.randint(1, self.noise_steps, (batch_size,)).to(self.device)

    def forward_diffusion(self, x0, t):
        sqrt_alpha_t = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]
        epsilon = torch.randn_like(x0)
        xt = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * epsilon
        return xt, epsilon
    
    def sample(self, batch_size):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
                epsilon_pred = self.model(x, t)
                alpha = self.alpha[i]
                alpha_cumprod = self.alpha_cumprod[i]
                beta = self.beta[i]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x =  1 / torch.sqrt(alpha) * (x - (beta / (torch.sqrt(1 - alpha_cumprod))) * epsilon_pred) + torch.sqrt(beta) * noise
        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
        
    def train(self, data_loader):
        self.model.train()
        for epoch in range(self.epochs):
            pbar = tqdm(data_loader)
            losses = []
            for i, (imgs, _) in enumerate(pbar):
                x0 = imgs.to(self.device)
                batch_size = x0.shape[0]
                t = self.sample_timestep(batch_size)
                xt, epsilon = self.forward_diffusion(x0, t)
                epsilon_pred = self.model(xt, t)
                loss = F.mse_loss(epsilon_pred, epsilon)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({'loss': loss.item()})
                losses.append(loss.item())
            
            print(f'Epoch {epoch} | Loss: {sum(losses)/len(losses)}')
            sampled_imgs = self.sample(batch_size=16)
            print(sampled_imgs.shape)
            grid = torchvision.utils.make_grid(sampled_imgs, nrow=4)
            ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
            im = Image.fromarray(ndarr)
            im.save(f'{self.results_dir}/images/{epoch}.png')
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f'{self.results_dir}/models/{epoch}.pth')