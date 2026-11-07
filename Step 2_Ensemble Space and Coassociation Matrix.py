#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy Ensemble Clustering (Kiran & Srinivas 2021)
- Uses saved label_m_r.pickle files (cluster assignments per config)
- Builds ensemble space & co-association matrix
- Runs Relational FCM on co-association matrix
- Model selection with XB index
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# USER INPUTS
# ----------------------------
label_dir = "/Users/debdut/Desktop/synthetic - 24th July/simulated_datasets/Simulation_1/variance_growth/labels/"

# ----------------------------
# Step 1: Load all label files
# ----------------------------
def load_all_labels(label_dir):
    label_files = sorted([f for f in os.listdir(label_dir) if f.startswith("label_") and f.endswith(".pickle")])
    all_labels = []
    for f in label_files:
        with open(os.path.join(label_dir, f), "rb") as handle:
            labels = pickle.load(handle)
            labels = np.array(labels).ravel()
            all_labels.append(labels)
    return all_labels, label_files

all_labels, label_files = load_all_labels(label_dir)
print(f"Loaded {len(all_labels)} label sets.")

# ----------------------------
# Step 2: Build co-association matrix (Eq. 3–6)
# ----------------------------
n_series = len(all_labels[0])   # number of time series (should be 50)
A = np.zeros((n_series, n_series))

for labels in all_labels:
    for i in range(n_series):
        for j in range(n_series):
            if labels[i] == labels[j]:
                A[i, j] += 1

A = A / len(all_labels)   # normalize

pd.DataFrame(A).to_csv("coassociation_matrix_k=10 to 15.csv", index=False)

plt.figure(figsize=(8,6))
sns.heatmap(A, cmap="viridis", vmin=0, vmax=1, cbar_kws={'label': 'Co-association'})
plt.title("Co-association Matrix")
plt.tight_layout()
plt.savefig("coassociation_matrix.png", dpi=1200)
plt.show()