import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

class TwoTowerRecommender(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.movie_embed = nn.Embedding(n_movies, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, movie_ids):
        u = self.user_embed(user_ids).squeeze()
        m = self.movie_embed(movie_ids).squeeze()
        pred = (u * m).sum(dim=1) + self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze() + self.global_bias
        return pred

def train_svd(train_df, n_users, n_movies, epochs=20, lr=0.01, embed_dim=64):
    model = TwoTowerRecommender(n_users, n_movies, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    user_tensor = torch.LongTensor(train_df['userId'].values)
    movie_tensor = torch.LongTensor(train_df['movieId'].values)
    rating_tensor = torch.FloatTensor(train_df['rating'].values)
    
    dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for users, movies, ratings in loader:
            optimizer.zero_grad()
            preds = model(users, movies)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return model

if __name__ == "__main__":
    print("Training SVD/Two-Tower recommender...")
    print("Download MovieLens 1M to data/ folder first")
    print("Train with: python train.py --epochs 20 --model two_tower")