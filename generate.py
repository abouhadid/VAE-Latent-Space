import os
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from model import UnFlatten,VAE
cwd_path = os.path.abspath(os.getcwd())
image_path = cwd_path+"/samples/"
weights_path = cwd_path+"/checkpoints/CVAE_weights"

device = torch.device("cpu")
print("Device:", device)
counter = 0
for j in range(32):
    model = torch.load(weights_path)
    model.to(device)
    mean_gen= torch.rand(32, 32)
    log_var_gen = torch.rand(32, 32)
    new_images = model.generate(mean_gen,log_var_gen)
    new_images = new_images .cpu().detach().numpy()
    for i in range(new_images.shape[0]):
        plt.imshow(new_images[i][0]*255, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        file_path = image_path + str(counter) + ".png"
        counter+=1
        plt.savefig(file_path, bbox_inches='tight')
