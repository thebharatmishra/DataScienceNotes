from torch import nn
from troch.optim import Adam
from torch.utils.data import Dataloader
from torchvision import datasets
from torchvision.transformers import ToTensor

train= datasets.Mnist(root='data', train=True, transform=ToTensor(), download=True)
test = datasets.Mnist(root='data', train=False, transform=ToTensor(), download=True)