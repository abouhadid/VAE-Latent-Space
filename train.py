import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F 
from model import UnFlatten,VAE

#initialize parameters
batch_size = 32     
epochs = 10               
image_size = 32
hidden_size = 1024        
latent_size = 32         
lr = 1e-3                
train_loss = []

device = "cpu" 
print(f"Using {device} device")

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()]),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()]),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)
vae.train()

for epoch in range(epochs):
  for i, (images, _) in enumerate(train_dataloader):
    images = images.to(device)
    optimizer.zero_grad()
    reconstructed_image, mean, log_var = vae(images)
    CE = F.binary_cross_entropy(reconstructed_image, images, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = CE + KLD
    loss.backward()
    train_loss.append(loss.item())
    optimizer.step()

    if(i % 100 == 0):
      print("Loss:")
      print(loss.item() / len(images))
torch.save(vae, "/home/jassem/fact/DS_project/checkpoints/CVAE_weights")
