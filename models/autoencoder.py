
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


# source: https://benjoe.medium.com/anomaly-detection-using-pytorch-autoencoder-and-mnist-31c5c2186329
class Autoencoder(nn.Module):
    
    def __init__(self, df):

        super(Autoencoder, self).__init__()

        self.__enc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.__dec = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU()
        )

        self.__dataloader = self.__prepare_data(df)



    def __prepare_data(self, df):
        pdf = df.toPandas().apply(pd.to_numeric, errors='raise')
        print(pdf[["tmst", "MACPayload", "MessageType", "FCnt"]].applymap(str).head(5))
        data_tensor = torch.tensor(pdf.values, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader


    def forward(self, x):
        encode = self.__enc(x)
        decode = self.__dec(encode)
        return decode
    

    """
    Train the Autoencoder model using normal LoRaWAN traffic data.

    Args:
        num_epochs: Number of epochs for training.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization).
        device: 'cpu' or 'cuda' for GPU acceleration.

    """
    def train(self, num_epochs, learning_rate, momentum=0.0005, device='cpu'):
        
        #self.to(device)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=momentum)
        criterion = nn.MSELoss()  # Autoencoders use MSE loss

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in self.__dataloader:
                #batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                recon = self(batch)

                # Compute loss
                loss = criterion(recon, batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.__dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        print("Training complete!")
