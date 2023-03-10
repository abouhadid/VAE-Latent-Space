#import dependencies
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

#initialize parameters
batch_size = 32     
epochs = 10               
image_size = 32
hidden_size = 1024        
latent_size = 32         
lr = 1e-3                
train_loss = []



class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 1024, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=1, image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size):
        super(VAE,self,).__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(image_channels, 32, 4, 2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, 4, 2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 128, 4, 2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 256, 4, 2),
        nn.LeakyReLU(0.2),
        nn.Flatten(),
    )
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.Sequential(
                                UnFlatten(),
                                nn.ConvTranspose2d(hidden_size, 128, 5, 2),
                                nn.ReLU(),
                                nn.ConvTranspose2d(128, 64, 5, 2),
                                nn.ReLU(),
                                nn.ConvTranspose2d(64, 32, 6, 2),
                                nn.ReLU(),
                                nn.ConvTranspose2d(32, image_channels, 6, 2),
                                nn.Sigmoid()   
                              )
    def sample(self, log_var, mean):
      std = torch.exp(0.5*log_var)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mean)
    def forward(self, x):
      x = self.encoder(x)
      log_var = self.encoder_logvar(x)
      mean = self.encoder_mean(x)
      z = self.sample(log_var, mean)
      x = self.fc(z)
      x = self.decoder(x)

      return x, mean, log_var
    def generate(self,mean,log_var):
      z = self.sample(log_var, mean)
      x = self.fc(z)
      x = self.decoder(x)

      return x


