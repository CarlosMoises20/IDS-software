import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class Autoencoder(nn.Module):

    def __init__(self, df, featuresCol):
        super(Autoencoder, self).__init__()

        self.__featuresCol = featuresCol

        self.__input_size = df.select(featuresCol).first()[featuresCol].size

        # Encoder
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

        # Decoder (used only for training)
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

        # Save original DataFrame
        self.__dataloader = self.__prepare_data(df)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__criterion = nn.MSELoss()

    def __prepare_data(self, df):
        features = np.array(df.select(self.__featuresCol).rdd.map(lambda row: row.features.toArray()).collect())
        features_tensor = torch.tensor(features, dtype=torch.float32)
        dataset = TensorDataset(features_tensor)
        return DataLoader(dataset, batch_size=64, shuffle=True)

    def forward(self, x):
        return self.__dec(self.__enc(x))

    def train(self, num_epochs=50, learning_rate=0.95, weight_decay=0.00001, momentum=0.8):
        self.to(self.__device)
        super().train()

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        for _ in range(num_epochs):
            running_loss = 0.0
            for _, (data,) in enumerate(self.__dataloader):
                data = data.to(self.__device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = self.__criterion(outputs, data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        return self.cpu()

    def transform(self):
        """
        Retorna a representação codificada (reduzida) do DataFrame original.
        """
        self.eval()
        encoded_vectors = []

        with torch.no_grad():
            for (data,) in self.__dataloader:
                encoded = self.__enc(data)
                encoded_vectors.extend(encoded.numpy())

        # Retorna uma lista de vetores codificados
        return encoded_vectors
