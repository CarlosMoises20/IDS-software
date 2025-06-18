import time, torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from common.auxiliary_functions import format_time
from pyspark.sql import Row
from pyspark.sql.functions import when, col
from sklearn.metrics import classification_report


class Autoencoder(nn.Module):
    
    def __init__(self, df_train, df_test):

        super(Autoencoder, self).__init__()

        self.__df_test = df_test

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

        self.__traindataloader = self.__prepare_data(df_train)

        self.__testdataloader = self.__prepare_data(df_test)

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__criterion = nn.MSELoss()


    """
    Transform pyspark dataframe into a pytorch dataloader

    Converts the Spark DataFrame into PyTorch Tensors and prepares DataLoaders.

    Assumes the DataFrame has a column 'features' with normalized feature vectors.
    
    Args:
        df_model (pyspark dataframe): dataframe to be transformed into a dataloader

    Returns:
        pytorch dataloader: dataloader with transformed data

    """
    def __prepare_data(self, df_model):

        # Collect the dataframe pyspark vectors and convert then into numpy format
        features = np.array(df_model.select("features").rdd.map(lambda row: row.features.toArray()).collect())

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
    def train(self, num_epochs=30, learning_rate=0.98, weight_decay=0.00001, momentum=0.8):

        self.to(self.__device)
        
        super().train()

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        for epoch in range(num_epochs):
            
            # NOTE: Uncomment the commented lines to print accuracy and processing time of each epoch

            #ep_start = time.time()
            running_loss = 0.0

            for _, (data,) in enumerate(self.__traindataloader):
                optimizer.zero_grad()
                outputs = self(data)
                loss = self.__criterion(outputs, data)
                loss.backward()  
                optimizer.step()
                running_loss += loss.item()                 
            
            #ep_end = time.time()

            #average_loss = (running_loss / len(self.__traindataloader)) * 100

            #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}%, Time: {format_time(ep_end - ep_start)}")

        return self.cpu()

    
    """
    Applies the trained model to compute reconstruction errors and label each example.

    Returns:
        DataFrame with original data, reconstruction error, and predicted label (0=normal, 1=anomaly)

    """
    def test(self):

        super().eval()
        errors = []
        true_labels = []

        with torch.no_grad():
            for (data,) in self.__testdataloader:
                output = self(data)
                batch_errors = torch.mean((output - data) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())

        # Compute threshold
        mean = np.mean(errors)
        std = np.std(errors)
        threshold = mean + (2 * std)

        # Predicted labels (1 = anomaly)
        predicted_labels = [1 if err > threshold else 0 for err in errors]

        # Get true labels from Spark DataFrame
        true_labels = self.__df_test.select("intrusion").rdd.map(lambda row: row["intrusion"]).collect()

        # Compute confusion matrix
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        confusion_matrix = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }

        report = classification_report(true_labels, predicted_labels, output_dict=True)
        return accuracy, confusion_matrix, report