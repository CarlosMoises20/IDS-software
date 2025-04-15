
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from models.cnn import CNN_Decoder, CNN_Encoder
from common.auxiliary_functions import format_time

"""
class Network(nn.Module):

    def __init__(self, output_size):
        super(Network, self).__init__()
        self.encoder = CNN_Encoder(output_size)
        self.decoder = CNN_Decoder(output_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)
"""


class Autoencoder(nn.Module):
    
    def __init__(self, df_train, df_test):

        super(Autoencoder, self).__init__()

        # Calculate the "features" vector size
        self.__input_size = df_train.select("features").first()["features"].size

        self.__enc = nn.Sequential(
            nn.Linear(self.__input_size, 512),  
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
            nn.Linear(512, self.__input_size),  
            nn.Sigmoid() 
        )

        self.__dataloader = self.__prepare_data(df_train)

        #self.__testdataloader = self.__prepare_data(df_test)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.model = Network(output_size=128)

        self.to(self.device)

        self.__criterion = nn.MSELoss()


    """
    Transform pyspark dataframe into a pytorch dataloader

    Converts the Spark DataFrame into PyTorch Tensors and prepares DataLoaders.

    Assumes the DataFrame has a column 'features' with normalized feature vectors.
    
    Args:
        df (pyspark dataframe): dataframe to be transformed into a dataloader

    Returns:
        pytorch dataloader: dataloader with transformed data

    """
    def __prepare_data(self, df):

        # Collecct the dataframe pyspark vectors and convert then into numpy format
        features = np.array(df.select("features").rdd.map(lambda row: row.features.toArray()).collect())

        # Convert data to pytorch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Dataset PyTorch
        dataset = TensorDataset(features_tensor)

        # Create the DataLoader
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        return dataloader


    """
    Forward pass of the Autoencoder model.

    Args:
        x: Input tensor.

    Returns:
        Tensor: Output tensor.

    """
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
    def train(self, num_epochs=30, learning_rate=0.7, weight_decay=0.00001, momentum=0.9):
        
        #self.train()

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        
        print("Starting Autoencoder training...")
        
        start_time = time.time()

        for epoch in range(num_epochs):
            
            ep_start = time.time()

            running_loss = 0.0

            for i, (data,) in enumerate(self.__dataloader):

                optimizer.zero_grad()

                outputs = self(data)

                loss = self.__criterion(outputs, data)
                
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()                 
            
            ep_end = time.time()

            average_loss = (running_loss / len(self.__dataloader)) * 100

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}%, Time: {format_time(ep_end - ep_start)}")


        end_time = time.time()
        
        print("Training complete! Total time of training was", format_time(end_time - start_time), "\n\n")


    """
    TODO: this function probably isn't necessary, since this algorithm isn't used for classification
    Test the Autoencoder model on test data.

    """
    """
    def test(self):

        #self.eval()

        total_loss = 0.0

        with torch.no_grad():
            for (data,) in self.__testdataloader:
                outputs = self(data)
                loss = self.__criterion(outputs, data)
                total_loss += loss.item()

        average_loss = total_loss / len(self.__testdataloader)
        print(f"Test MSE Loss: {average_loss:.6f}")
        return average_loss
    """
