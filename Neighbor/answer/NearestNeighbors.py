import torch
import argparse
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
args = parser.parse_args()

# Load all embeddings
embeddings = torch.load("data/Neighbor/answer/embeddings/all_answers.bin", weights_only=False)

# Fit NearestNeighbors model
neigh = NearestNeighbors(n_neighbors=500, n_jobs=-1, metric="cosine")
neigh.fit(embeddings)

# Compute neighbors
distances, indices = neigh.kneighbors(embeddings)

# Save results
torch.save((distances, indices), "data/Neighbor/answer/embeddings/neighbor500_all.bin")