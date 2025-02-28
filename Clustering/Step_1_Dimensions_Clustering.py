# Functions for saving data
import os
import pandas as pd
import numpy as np
import csv
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def write_unique_row_to_csv(file_path, row_data, delimiter=';'):
    """Writes a row to a CSV file only if it is not already present."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(row_data)
    else:
        with open(file_path, 'r', newline='') as file:
            for line in file:
                if line.strip() == delimiter.join(row_data):
                    return  # Row already exists, do nothing
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(row_data)

def determine_column_type(col):
    """Determines the type of a given column."""
    if col.dtype == 'bool':
        return 'B'  # Boolean type
    elif col.dtype == 'object' and col.str.len().max() == 1:
        return 'C'  # Categorical type (single character strings)
    else:
        return 'D'  # Numerical or other data type

def split_large_cluster(cluster_columns, correlation_matrix, max_size=15):
    """Splits a cluster that exceeds the maximum size using hierarchical clustering."""
    global global_cluster_id
    # Extract the correlation submatrix for the cluster
    sub_corr_matrix = correlation_matrix.loc[cluster_columns, cluster_columns]
    # Convert correlation to distance (1 - absolute correlation)
    sub_distance_matrix = 1 - np.abs(sub_corr_matrix.values)
    sub_distance_matrix = squareform(sub_distance_matrix, force='tovector')
    sub_distance_matrix = np.clip(sub_distance_matrix, 0, None)

    # Apply hierarchical clustering (Ward's method) to split the cluster
    sub_linkage_matrix = linkage(sub_distance_matrix, method='ward')
    k = int(np.ceil(len(cluster_columns) / max_size))  # Determine number of sub-clusters
    sub_clusters = fcluster(sub_linkage_matrix, k, criterion='maxclust')

    # Create new sub-clusters
    new_sub_clusters = {}
    for i in np.unique(sub_clusters):
        new_cluster_name = f'cluster_{global_cluster_id}'
        global_cluster_id += 1
        new_sub_clusters[new_cluster_name] = [
            cluster_columns[j] for j in range(len(cluster_columns)) if sub_clusters[j] == i
        ]
    return new_sub_clusters

def merge_small_clusters(small_cluster, other_clusters, correlation_matrix, min_size=5):
    """Merges small clusters with the most similar existing cluster based on correlation."""
    small_cluster_set = set(small_cluster)
    best_similarity = -np.inf
    best_merge_target = None

    # Find the cluster with the highest average correlation to the small cluster
    for cluster_id, cluster_columns in other_clusters.items():
        inter_corr = correlation_matrix.loc[small_cluster, cluster_columns].values.flatten()
        avg_similarity = np.mean(inter_corr)
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_merge_target = cluster_id

    # Merge with the most similar cluster if possible, otherwise create a new cluster
    if best_merge_target:
        other_clusters[best_merge_target].extend(small_cluster)
    else:
        other_clusters[f'cluster_{len(other_clusters) + 1}'] = small_cluster

# Parameters and dataset name
min_cluster_size = 6
max_cluster_size = 13
dataset_name = 'S4-ADL5-MNAR_20000_130'
dataset_name_MV = f'{dataset_name}_260000_1'

# Read dataset headers
headers = []
with open('../Preprocessing/Headers/Headers.csv', 'r') as f:
    for line in f:
        values = line.strip().split(';')
        headers.append(values)

# Identify the relevant columns for the dataset
columns = None
for header in headers:
    if header[-1] == dataset_name_MV:
        columns = [col for col in header[:-1] if col]
        break
if columns is None:
    raise ValueError(f"Header for dataset '{dataset_name_MV}' not found in the header file.")

# Load the dataset and compute Pearson correlation matrix
df_from_csv = pd.read_csv(f"../Datasets/DOMINO_Datasets/{dataset_name}/{dataset_name_MV}.csv", sep=";", names=columns)
correlation_matrix_from_csv = df_from_csv.corr(method='pearson')

# Compute distance matrix (1 - absolute correlation)
distance_matrix = 1 - np.abs(correlation_matrix_from_csv.values)
distance_matrix = np.clip(distance_matrix, 0, None)
distance_matrix = squareform(distance_matrix, force='tovector')

# Perform hierarchical clustering using Ward’s method
linkage_matrix = linkage(distance_matrix, method='ward')
initial_clusters = fcluster(linkage_matrix, 8, criterion='maxclust')

# Assign clusters based on initial clustering
clusters = {f'cluster_{i}': correlation_matrix_from_csv.columns[initial_clusters == i].tolist()
            for i in np.unique(initial_clusters)}

# Initialize global cluster ID counter
global_cluster_id = len(clusters) + 1

# Iterative process to balance clusters
max_iterations = 100  # Prevent infinite loops
iteration = 0

while iteration < max_iterations:
    iteration += 1
    print(f"Iteration {iteration}, clusters: {clusters}")

    # Identify clusters that need to be split or merged
    clusters_to_split = {k: v for k, v in clusters.items() if len(v) > max_cluster_size}
    clusters_to_merge = {k: v for k, v in clusters.items() if len(v) < min_cluster_size}

    # If no adjustments are needed, exit the loop
    if not clusters_to_split and not clusters_to_merge:
        break

    # Split large clusters
    for cluster_id, cluster_columns in clusters_to_split.items():
        del clusters[cluster_id]
        new_sub_clusters = split_large_cluster(cluster_columns, correlation_matrix_from_csv, max_size=max_cluster_size)
        clusters.update(new_sub_clusters)

    # Merge small clusters
    for cluster_id, cluster_columns in list(clusters_to_merge.items()):
        if cluster_id in clusters:  # Ensure the cluster still exists
            del clusters[cluster_id]
            merge_small_clusters(cluster_columns, clusters, correlation_matrix_from_csv, min_size=min_cluster_size)

    # Stop if all clusters are within the desired size range
    if all(len(v) <= max_cluster_size for v in clusters.values()):
        break

if iteration >= max_iterations:
    print("Warning: Maximum number of iterations reached while balancing clusters.")

# Save results
base_path = f"../Datasets/Missing_Datasets/{dataset_name}/{dataset_name_MV}"
df_from_csv = pd.read_csv(f"../Datasets/Missing_Datasets/{dataset_name}/{dataset_name_MV}.csv", sep=";", names=columns)
dataset_name_MV_B = f'{dataset_name_MV}_Balanced'

for cluster_id, cluster_columns in clusters.items():
    cluster_folder_path = os.path.join(base_path)
    os.makedirs(cluster_folder_path, exist_ok=True)

    # Create a new dataframe for the cluster
    cluster_df = df_from_csv[cluster_columns]

    # Save the cluster dataset as a CSV file
    cluster_filename_no_ext = f"{dataset_name_MV_B}_{cluster_id}"
    cluster_filename = f"{dataset_name_MV_B}_{cluster_id}.csv"
    cluster_filepath = os.path.join(cluster_folder_path, cluster_filename)
    cluster_df.to_csv(cluster_filepath, index=False, sep=";")
    print(f"Cluster {cluster_id} saved at {cluster_filepath}")

    # Save column types
    column_types = [determine_column_type(cluster_df[col]) for col in cluster_df.columns]
    write_unique_row_to_csv('../Preprocessing/ColumnTypes/ColumnTypesClust.csv', column_types + [cluster_filename_no_ext])

    # Save headers
    write_unique_row_to_csv('../Preprocessing/Headers/HeadersClust.csv', cluster_df.columns.tolist() + [cluster_filename_no_ext])

    # Filter original file for tuples related to the cluster
    original_file_path = f'../Preprocessing/Initial_Tuples/{dataset_name}/{dataset_name_MV}.csv'
    os.makedirs("InitialTuples/", exist_ok=True)

    with open(original_file_path, 'r') as original_file:
        reader = csv.reader(original_file, delimiter=';')
        filtered_rows = [row for row in reader if row[1] in cluster_df.columns.tolist()]

    # Save filtered rows
    with open(f"InitialTuples/{cluster_filename}", 'w', newline='') as filtered_file:
        writer = csv.writer(filtered_file, delimiter=';')
        writer.writerows(filtered_rows)

    print(f"Filtered initial tuples for cluster {cluster_id} saved.")
