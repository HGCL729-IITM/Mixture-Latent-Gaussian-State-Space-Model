#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Fuzzy DBSCAN Grid Search on Co-association Matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --------------------------------------------------------------------------
# USER INPUTS
# --------------------------------------------------------------------------
eps_range = np.linspace(0.05, 5, 100)
min_pts_range = range(1, 51)             # integers 1 to 200
# --------------------------------------------------------------------------

def fuzzy_dbscan_precomputed(D, eps, min_pts):
    n_points = D.shape[0]
    labels = np.full(n_points, 0, dtype=int)
    cluster_id = 0

    for i in range(n_points):
        if labels[i] != 0:
            continue
        neighbors_indices = np.where(D[i] <= eps)[0]
        if len(neighbors_indices) < min_pts:
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seed_set = set(neighbors_indices)
        seed_set.remove(i)
        while seed_set:
            current_point_idx = seed_set.pop()
            if labels[current_point_idx] <= 0:
                labels[current_point_idx] = cluster_id
                current_neighbors = np.where(D[current_point_idx] <= eps)[0]
                if len(current_neighbors) >= min_pts:
                    seed_set.update(current_neighbors)

    n_clusters = cluster_id
    memberships = np.zeros((n_points, n_clusters + 1))
    for i in range(n_points):
        neighbor_indices = np.where(D[i] <= eps)[0]
        cluster_neighbor_counts = np.zeros(n_clusters + 1)
        for neighbor_idx in neighbor_indices:
            cluster_label = labels[neighbor_idx]
            if cluster_label > 0:
                cluster_neighbor_counts[cluster_label] += 1
        point_memberships = cluster_neighbor_counts / min_pts
        point_memberships = np.minimum(point_memberships, 1.0)
        total_membership = np.sum(point_memberships)
        if total_membership == 0:
            memberships[i, 0] = 1.0
        else:
            memberships[i, :] = point_memberships
            if total_membership < 1.0:
                memberships[i, 0] = 1.0 - total_membership
    return memberships, labels

# 1. Load the co-association matrix
A = pd.read_csv("/Users/debdut/Desktop/synthetic - 24th July/simulated_datasets/Simulation_1/variance_growth/coassociation_matrix_k=10 to 15.csv").values
print("Successfully loaded co-association matrix.")
D = 1 - A
D = np.clip(D, 0, 1)

# 2. Grid search over eps and min_pts
results = []
print("Running grid search...")
for eps in tqdm(eps_range, desc="eps sweep"):
    for min_pts in min_pts_range:
        memberships, hard_labels = fuzzy_dbscan_precomputed(D, eps=eps, min_pts=min_pts)
        n_clusters_found = len(np.unique(hard_labels)) - (1 if 0 in np.unique(hard_labels) else 0)
        n_noise_points = np.sum(hard_labels == 0)
        results.append([eps, min_pts, n_clusters_found, n_noise_points])

# 3. Save results
df_results = pd.DataFrame(results, columns=["eps", "min_pts", "n_clusters", "n_noise"])
df_results.to_csv("/Users/debdut/Desktop/synthetic - 24th July/simulated_datasets/Simulation_1/variance_growth/fuzzy_dbscan_grid_results.csv", index=False)
print("Grid search results saved to 'fuzzy_dbscan_grid_results.csv'")

# 4. Scatter plot: eps vs min_pts, colored by clusters found
plt.figure(figsize=(10, 6))
sc = plt.scatter(df_results["eps"], df_results["min_pts"], 
                 c=df_results["n_clusters"], cmap="viridis", s=15)
plt.colorbar(sc, label="Number of Clusters Found")
plt.xlabel("eps")
plt.ylabel("min_pts")
plt.title("Fuzzy DBSCAN Grid Search: Clusters Found")
plt.grid(True, alpha=0.3)
plt.savefig("fuzzy_dbscan_grid_clusters.png", dpi=300)
plt.show()
