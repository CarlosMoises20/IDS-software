
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from common.auxiliary_functions import format_time


class Autoencoder(nn.Module):
    
    def __init__(self, df_train, df_test):

        super(Autoencoder, self).__init__()

        self.__enc = nn.Sequential(
            nn.Linear(37, 512),  
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
            nn.Linear(512, 37),  
            nn.Sigmoid() 
        )

        self.__traindataloader = self.__prepare_data(df_train)

        self.__testdataloader = self.__prepare_data(df_test)


    """
    Transform pyspark dataframe into a pytorch dataloader
    
    Args:
        df (pyspark dataframe): dataframe to be transformed into a dataloader

    Returns:
        pytorch dataloader: dataloader with transformed data
    """
    def __prepare_data(self, df):
        pdf = df.toPandas().apply(pd.to_numeric, errors='raise')
        data_tensor = torch.tensor(pdf.values, dtype=torch.float32)
        dataset = TensorDataset(data_tensor, )      # TensorDataset expects many tensors, so we use a tuple
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8,
                                pin_memory=True, drop_last=True)
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
    def train(self, num_epochs=15, learning_rate=0.8, weight_decay=0.00001, momentum=0.9, 
              device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.to(device)
        
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        criterion = nn.MSELoss()  # Autoencoders use MSE loss
        
        print("Starting Autoencoder training...")
        
        start_time = time.time()

        for epoch in range(num_epochs):
            
            ep_start = time.time()

            running_loss = 0.0

            for i, (data,) in enumerate(self.__traindataloader):

                optimizer.zero_grad()

                outputs = self(data)

                loss = criterion(outputs, data)
                
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item() 
                
            
            ep_end = time.time()

            average_loss = running_loss / len(self.__traindataloader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}, Time: {format_time(ep_end - ep_start)}")


        end_time = time.time()
        
        print("Training complete! Total time of training was", format_time(end_time - start_time), "\n\n")


        # TODO: return the model, total loss and average accuracy


    """
    Test the Autoencoder model on test data.

    """
    def test(self, model):

        correct = 0
        total = 0

        dataiter = iter(self.__testdataloader)

        with torch.no_grad():
            for data in dataiter:
                results, labels = data
                outputs = model(results)
                _, predicted = torch.max(outputs.data, 1)
                total += data.size(0)
                correct += (predicted == data).sum().item()

        accuracy = (100 * correct) // total

        return accuracy
