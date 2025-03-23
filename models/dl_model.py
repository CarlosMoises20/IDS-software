

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models



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
    
    # TODO: fix
    def train(self, num_epochs, learning_rate, momentum, weight_decay):
        
        
        for epoch in range(num_epochs):

            """
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))"
            """