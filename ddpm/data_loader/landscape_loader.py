import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

def load_Landscape(data_path, batch_size, image_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(1.25*image_size)),  # image_size + 1/4 *image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader