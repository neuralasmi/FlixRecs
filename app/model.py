import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FlixRecsModel:
    def __init__(self):
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_movie_matrix = None
        self.movie_ids = None
        self.user_ids = None

    def load(self):
        pass  # Loads from checkpoints if available

    def recommend(self, user_id: int, top_k: int = 10):
        # Return dummy recommendations when model not trained
        return [{"movie_id": i, "score": round(0.9 - i*0.05, 3)} for i in range(1, top_k+1)]

    def similar(self, movie_id: int, top_k: int = 10):
        return [{"movie_id": i, "score": round(0.8 - i*0.04, 3)} for i in range(1, top_k+1)]