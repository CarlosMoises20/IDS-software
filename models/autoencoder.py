import time, torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from common.auxiliary_functions import format_time
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


class Autoencoder(nn.Module):
    
    def __init__(self, spark_session, df_model, dev_addr):

        super(Autoencoder, self).__init__()

        self.__spark_session = spark_session

        self.__df_model = df_model

        self.__dev_addr = dev_addr

        # Calculate the "features" vector size
        self.__input_size = df_model.select("features").first()["features"].size

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

        self.__dataloader = self.__prepare_data(df_model)

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

        # Collecct the dataframe pyspark vectors and convert then into numpy format
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
    def train(self, num_epochs=20, learning_rate=0.75, weight_decay=0.00001, momentum=0.8):

        self.to(self.__device)
        
        super().train()

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        for epoch in range(num_epochs):
            
            # NOTE: Uncomment the commented lines to print accuracy and processing time of each epoch

            #ep_start = time.time()
            running_loss = 0.0

            for _, (data,) in enumerate(self.__dataloader):
                optimizer.zero_grad()
                outputs = self(data)
                loss = self.__criterion(outputs, data)
                loss.backward()  
                optimizer.step()
                running_loss += loss.item()                 
            
            #ep_end = time.time()

            #average_loss = (running_loss / len(self.__dataloader)) * 100

            #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}%, Time: {format_time(ep_end - ep_start)}")


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
    def label_data_by_reconstruction_error(self, threshold=None):

        errors = self.__compute_reconstruction_errors()

        # If no threshold provided, use mean + 2*std as default
        if threshold is None:
            mean = np.mean(errors)
            std = np.std(errors)
            threshold = mean + 2 * std

        #print(f"AE threshold for device {self.__dev_addr}: {round(threshold, 5)}")

        # Label based on threshold
        labels = [1 if err > threshold else 0 for err in errors]


        # TODO: try a more efficient approach that does the same: create the dataframe to be used
        # in supervised learning (on this current case, Random Forest)

        schema = StructType([
            StructField("reconstruction_error", FloatType(), False),
            StructField("intrusion", IntegerType(), False)
        ])

        self.__df_model.coalesce(numPartitions=800)

        errors_labels = [(float(err), int(lbl)) for err, lbl in zip(errors, labels)]
        errors_df_model = (self.__spark_session).createDataFrame(errors_labels, schema)

        # Add columns to original df_model (you can use withColumn if aligning by index is guaranteed)
        df_model_with_errors = (self.__df_model).withColumn("row_idx", F.monotonically_increasing_id())
        errors_df_model = errors_df_model.withColumn("row_idx", F.monotonically_increasing_id())

        result_df_model = df_model_with_errors.join(errors_df_model, on="row_idx").drop("row_idx")

        csv_filename = f"./generatedDatasets/labeled_autoencoder_output_device_{self.__dev_addr}.csv"

        result_df_model.drop("features").toPandas().to_csv(csv_filename, index=False)

        return result_df_model