# ============================================================
# RIFT - STEP 1: Dimension Clustering
# ============================================================

import os
import csv
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Optional metrics (used in other configurations)
from tslearn.metrics import dtw
from fastdtw import fastdtw


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def write_unique_row_to_csv(file_path, row_data, delimiter=';'):
    """Write a row to CSV only if it is not already present."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(row_data)
    else:
        with open(file_path, 'r', newline='') as f:
            for line in f:
                if line.strip() == delimiter.join(row_data):
                    return
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(row_data)


def determine_column_type(col):
    """Infer a coarse column type used downstream."""
    if col.dtype == 'bool':
        return 'B'
    elif col.dtype == 'object' and col.str.len().max() == 1:
        return 'C'
    else:
        return 'D'


# ------------------------------------------------------------
# Distance matrix computation
# ------------------------------------------------------------

def compute_distance_matrix(df, metric="pearson"):
    """
    Compute a pairwise distance matrix between dimensions.

    For Pearson/Spearman/Kendall:
        d(x, y) = 1 - |corr(x, y)|
    """
    cols = df.columns
    n = len(cols)
    dist_full = np.zeros((n, n))

    if metric in ["pearson", "spearman", "kendall"]:
        corr = df.corr(method=metric).fillna(0)
        dist_full = 1 - np.abs(corr.values)

    elif metric == "sbd":
        def sbd(x, y):
            x = (x - np.mean(x)) / (np.std(x) + 1e-8)
            y = (y - np.mean(y)) / (np.std(y) + 1e-8)
            cc = np.correlate(x, y, mode="full")
            return 1 - np.max(cc) / len(x)

        for i in range(n):
            for j in range(i + 1, n):
                d = sbd(df[cols[i]].values, df[cols[j]].values)
                dist_full[i, j] = dist_full[j, i] = d

    elif metric.startswith("dtw"):
        window = None
        if "-" in metric:
            _, _, w = metric.split("-")
            window = int(w)

        for i in range(n):
            for j in range(i + 1, n):
                if window:
                    d = dtw(
                        df[cols[i]].values,
                        df[cols[j]].values,
                        global_constraint="sakoe_chiba",
                        sakoe_chiba_radius=window
                    )
                else:
                    d = fastdtw(
                        df[cols[i]].values,
                        df[cols[j]].values
                    )[0]
                dist_full[i, j] = dist_full[j, i] = d
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    np.fill_diagonal(dist_full, 0)
    return pd.DataFrame(dist_full, index=cols, columns=cols)


# ------------------------------------------------------------
# Cluster refinement operators
# ------------------------------------------------------------

def split_large_cluster(cluster_columns, dist_matrix, max_size):
    """
    Split an oversized cluster using hierarchical clustering.
    """
    global global_cluster_id

    sub_dist = dist_matrix.loc[cluster_columns, cluster_columns].values
    sub_vector = squareform(sub_dist, force="tovector")

    sub_linkage = linkage(sub_vector, method="ward")
    k = int(np.ceil(len(cluster_columns) / max_size))
    labels = fcluster(sub_linkage, k, criterion="maxclust")

    new_clusters = {}
    for lbl in np.unique(labels):
        new_name = f"cluster_{global_cluster_id}"
        global_cluster_id += 1
        new_clusters[new_name] = [
            cluster_columns[i]
            for i in range(len(cluster_columns))
            if labels[i] == lbl
        ]

    return new_clusters


def merge_small_cluster(cluster_columns, clusters, dist_matrix, max_size):
    """
    Merge a small cluster into the nearest compatible cluster
    (i.e., respecting the max_size constraint).
    """
    best_target = None
    best_distance = np.inf

    for cid, cols in clusters.items():
        if len(cols) + len(cluster_columns) > max_size:
            continue

        d = np.mean(dist_matrix.loc[cluster_columns, cols].values)
        if d < best_distance:
            best_distance = d
            best_target = cid

    if best_target is not None:
        clusters[best_target].extend(cluster_columns)
    else:
        # Fallback: keep the cluster isolated (rare in practice)
        clusters[f"cluster_{len(clusters) + 1}"] = cluster_columns


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------

min_cluster_size = 6
max_cluster_size = 13

dataset_name = "S4-ADL5_20000_130"
dataset_mv = f"{dataset_name}_130000_1"
metric = "pearson"

# ------------------------------------------------------------
# Load headers and dataset
# ------------------------------------------------------------

headers = []
with open("../Preprocessing/Headers/Headers.csv", "r") as f:
    for line in f:
        headers.append(line.strip().split(";"))

columns = None
for h in headers:
    if h[-1] == dataset_mv:
        columns = [c for c in h[:-1] if c]
        break

if columns is None:
    raise ValueError(f"Header for dataset '{dataset_mv}' not found.")

df = pd.read_csv(
    f"../Datasets/DOMINO_Datasets/{dataset_name}/{dataset_mv}.csv",
    sep=";",
    names=columns
)

# ------------------------------------------------------------
# Step 1.1 – Correlation-based distance
# ------------------------------------------------------------

dist_matrix = compute_distance_matrix(df, metric=metric)

# ------------------------------------------------------------
# Step 1.2 – Initial hierarchical clustering
# ------------------------------------------------------------

dist_vector = squareform(dist_matrix.values, force="tovector")
Z = linkage(dist_vector, method="ward")

# Initial cut used only to obtain a starting partition
initial_labels = fcluster(Z, 8, criterion="maxclust")

clusters = {
    f"cluster_{i}": dist_matrix.columns[initial_labels == i].tolist()
    for i in np.unique(initial_labels)
}

global_cluster_id = len(clusters) + 1

# ------------------------------------------------------------
# Step 1.3 – Split oversized clusters
# ------------------------------------------------------------

split_required = True
while split_required:
    split_required = False
    oversized = {
        cid: cols for cid, cols in clusters.items()
        if len(cols) > max_cluster_size
    }

    for cid, cols in oversized.items():
        split_required = True
        del clusters[cid]
        clusters.update(
            split_large_cluster(cols, dist_matrix, max_cluster_size)
        )

# ------------------------------------------------------------
# Step 1.4 – Merge undersized clusters
# ------------------------------------------------------------

small_clusters = {
    cid: cols for cid, cols in clusters.items()
    if len(cols) < min_cluster_size
}

for cid, cols in list(small_clusters.items()):
    if cid not in clusters:
        continue

    del clusters[cid]
    merge_small_cluster(cols, clusters, dist_matrix, max_cluster_size)

# ------------------------------------------------------------
# Step 1.5 – Save clustered datasets
# ------------------------------------------------------------

base_path = f"../Datasets/Missing_Datasets/{dataset_name}/{dataset_mv}"
df_missing = pd.read_csv(
    f"{base_path}.csv",
    sep=";",
    names=columns
)

dataset_mv_balanced = f"{dataset_mv}_Balanced"

for cid, cols in clusters.items():
    os.makedirs(base_path, exist_ok=True)

    cluster_df = df_missing[cols]
    fname_no_ext = f"{dataset_mv_balanced}_{cid}"
    fname = f"{fname_no_ext}.csv"
    fpath = os.path.join(base_path, fname)

    cluster_df.to_csv(fpath, index=False, sep=";")
    print(f"Cluster {cid} saved at {fpath}")

    column_types = [determine_column_type(cluster_df[c]) for c in cluster_df.columns]
    write_unique_row_to_csv(
        "../Preprocessing/ColumnTypes/ColumnTypesClust.csv",
        column_types + [fname_no_ext]
    )

    write_unique_row_to_csv(
        "../Preprocessing/Headers/HeadersClust.csv",
        cluster_df.columns.tolist() + [fname_no_ext]
    )

    original_file = f"../Preprocessing/Initial_Tuples/{dataset_name}/{dataset_mv}.csv"
    os.makedirs("InitialTuples", exist_ok=True)

    with open(original_file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        rows = [r for r in reader if r[1] in cluster_df.columns.tolist()]

    with open(f"InitialTuples/{fname}", "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows)

    print(f"Filtered initial tuples for cluster {cid} saved.")
