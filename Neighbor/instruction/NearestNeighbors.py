import torch
import argparse
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
args = parser.parse_args()

embeddings = torch.load("data/Neighbor/instruction/embeddings/better.bin")
neigh = NearestNeighbors(n_neighbors = 500, n_jobs = -1, metric = "cosine")
neigh.fit(embeddings)
torch.save(neigh.kneighbors(embeddings), "data/Neighbor/instruction/embeddings/neighbor500_better.bin")