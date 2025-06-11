"""
Exact GPU-based 1-NN Classifiers + One-Class Models (Single, Parallel, CUDA)
Includes: Manhattan, One-Class SVM, Isolation Forest, K-Means, Autoencoder (Improved with Batch Scoring)
"""

import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getStatistics(start_time):
    print("Total Execution Time:", time.time() - start_time)
    process = psutil.Process()
    memory_kb = process.memory_full_info().uss / 1024
    print("Memory Usage (KB):", memory_kb)

# ---------------------- Improved Autoencoder Model ----------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---------------------- Autoencoder Modes ----------------------
def compute_autoencoder(training, testing, mode="cuda", epochs=100, batch_size=64):
    device = torch.device("cuda" if mode == "cuda" and torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    training_np = scaler.fit_transform(training.to_numpy())
    testing_np = scaler.transform(testing.to_numpy())

    x_train = torch.tensor(training_np, dtype=torch.float32)
    x_test = torch.tensor(testing_np, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False)

    model = Autoencoder(x_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
        avg_loss = total_loss / len(x_train)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    print("Scoring test samples in batches...")
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch_data = batch[0].to(device)
            output = model(batch_data)
            loss = torch.mean((batch_data - output) ** 2, dim=1)
            errors.extend(loss.cpu().numpy())

    return np.array(errors)

# ---------------------- Main Entry ----------------------
def rasterOneClassAutoencoder(training, testing, topElements=-1):
    start_time = time.time()
    distances = compute_autoencoder(training, testing)
    testing['AE_ReconError'] = distances
    sorted_df = testing.sort_values('AE_ReconError').head(topElements)
    getStatistics(start_time)
    return testing, sorted_df
