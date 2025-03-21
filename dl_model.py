

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# Define the model
class TorchModel(nn.Module):
    def __init__(self): 
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Example layer
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x