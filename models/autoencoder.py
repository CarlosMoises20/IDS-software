
import time, torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from common.auxiliary_functions import format_time
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


class Autoencoder(nn.Module):
    
    def __init__(self, pdf, dev_addr):

        super(Autoencoder, self).__init__()

        self.__pdf = pdf

        print(pdf["features"])

        #
        #self.__spark_session = spark_session

        self.__dev_addr = dev_addr

        # Calculate the "features" vector size
        self.__input_size = int(pdf["features"][0]["size"])

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

        self.__dataloader = self.__prepare_data(pdf)

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __prepare_data(self, pdf):

        # TODO: continue bug fixing from this point
        for i, f in enumerate(pdf["features"]):
            values = f["values"] if isinstance(f, dict) else f.toArray()
            if len(values) != self.__input_size:
                print(f"[ERROR] Feature at index {i} has size {len(values)} but expected {self.__input_size}")

        # Collecct the dataframe pyspark vectors and convert then into numpy format
        features = np.array([np.array(f["values"]) if isinstance(f, dict) else f.toArray() for f in pdf["features"]])

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
    def train(self, num_epochs=30, learning_rate=0.75, weight_decay=0.00001, momentum=0.8):

        self.to(self.__device)
        
        super().train()

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        for epoch in range(num_epochs):
            
            ep_start = time.time()
            running_loss = 0.0

            for _, (data,) in enumerate(self.__dataloader):
                optimizer.zero_grad()
                outputs = self(data)
                loss = self.__criterion(outputs, data)
                loss.backward()  
                optimizer.step()
                running_loss += loss.item()                 
            
            ep_end = time.time()

            average_loss = (running_loss / len(self.__dataloader)) * 100

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}%, Time: {format_time(ep_end - ep_start)}")


    """
    Test the Autoencoder model on test data.

    """
    def __compute_reconstruction_errors(self):

        super().eval()
        errors = []

        with torch.no_grad():
            for (data,) in self.__dataloader:
                output = self(data)
                batch_errors = torch.mean((output - data) ** 2, dim=1)  # MSE per sample
                errors.extend(batch_errors.cpu().numpy())  # convert to list

        return errors


    """
    Applies the trained model to compute reconstruction errors and label each example.

    Returns:
        DataFrame with original data, reconstruction error, and predicted label (0=normal, 1=anomaly)

    """
    def label_data_by_reconstruction_error(self):

        errors = self.__compute_reconstruction_errors()

        # Threshold is mean + 2*std, to determine intrusions that correspond to a reconstruction error
        # over the threshold
        mean = np.mean(errors)
        std = np.std(errors)
        threshold = mean + 2 * std

        # Label based on threshold
        labels = [1 if err > threshold else 0 for err in errors]

        schema = StructType([
            StructField("reconstruction_error", FloatType(), False),
            StructField("intrusion", IntegerType(), False)
        ])

        # TODO: fix this... update column "intrusion" based on reconstruction error

        return errors

        """errors_labels = [(float(err), int(lbl)) for err, lbl in zip(errors, labels)]
        errors_df = (self.__spark_session).createDataFrame(errors_labels, schema)

        # Add columns to original df (you can use withColumn if aligning by index is guaranteed)
        df_with_errors = (self.__df).withColumn("row_idx", F.monotonically_increasing_id())
        errors_df = errors_df.withColumn("row_idx", F.monotonically_increasing_id())

        result_df = df_with_errors.join(errors_df, on="row_idx").drop("row_idx")

        csv_filename = f"./generatedDatasets/labeled_autoencoder_output_device_{self.__dev_addr}.csv"

        result_df.toPandas().to_csv(csv_filename, index=False)

        return result_df"""