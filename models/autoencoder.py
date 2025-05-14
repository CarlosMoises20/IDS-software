import time, torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from common.auxiliary_functions import format_time
from pyspark.sql import Row, functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


class Autoencoder(nn.Module):
    
    def __init__(self, spark_session, df_model_train, df_model_test, dev_addr, dataset_type):

        super(Autoencoder, self).__init__()

        self.__spark_session = spark_session

        self.__df_model_train = df_model_train

        self.__df_model_test = df_model_test

        self.__dev_addr = dev_addr

        self.__dataset_type = dataset_type.value["name"]

        self.__input_size = df_model_train.select("features").first()["features"].size

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

        self.__traindataloader = self.__prepare_data(df_model_train)

        self.__testdataloader = self.__prepare_data(df_model_test)

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


    """
    Test the Autoencoder model on test data.

    """
    def __compute_reconstruction_errors(self, dataloader):

        super().eval()
        errors = []

        with torch.no_grad():
            for (data,) in dataloader:
                output = self(data)
                batch_errors = torch.mean((output - data) ** 2, dim=1)  # MSE per sample
                errors.extend(batch_errors.cpu().numpy())  # convert to list

        return errors


    def __label_data(self, errors, df_model):

        # Define a threshold to be used to define the binary label
        mean = np.mean(errors)
        std = np.std(errors)
        threshold = mean + 2 * std

        # Label based on threshold
        labels = [1 if err > threshold else 0 for err in errors]

        errors_labels = [(float(err), int(lbl)) for err, lbl in zip(errors, labels)]

        errors_rdd = self.__spark_session.sparkContext.parallelize(errors_labels)
        errors_rdd_indexed = errors_rdd.zipWithIndex().map(
            lambda x: Row(reconstruction_error=x[0][0], intrusion=x[0][1], row_idx=x[1])
        )
        errors_df_model = self.__spark_session.createDataFrame(errors_rdd_indexed)
       
        df_model_indexed = df_model.rdd.zipWithIndex().map(lambda x: Row(**x[0].asDict(), row_idx=x[1]))
        df_model_with_index = self.__spark_session.createDataFrame(df_model_indexed)
        
        result_df_model = df_model_with_index.join(errors_df_model, on="row_idx").drop("row_idx")

        # NOTE: uncomment these two lines if you want to generate a CSV file with the dataframe with the
        # label generated by AE

        csv_filename = f"./generatedDatasets/labeled_autoencoder_output_device_{self.__dev_addr}_{self.__dataset_type}.csv"
        result_df_model.drop("features").toPandas().to_csv(csv_filename, index=False)

        return result_df_model



    """
    Applies the trained model to compute reconstruction errors and label each example.

    Returns:
        DataFrame with original data, reconstruction error, and predicted label (0=normal, 1=anomaly)

    """
    def label_data_by_reconstruction_error(self):

        errors_train = self.__compute_reconstruction_errors(self.__traindataloader)
        errors_test = self.__compute_reconstruction_errors(self.__testdataloader)

        result_df_train = self.__label_data(errors_train, self.__df_model_train)
        result_df_test = self.__label_data(errors_test, self.__df_model_test)

        return result_df_train, result_df_test