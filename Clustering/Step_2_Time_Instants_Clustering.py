import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
import csv
import warnings

# Suppress specific warnings from KMedoids related to empty clusters
warnings.filterwarnings("ignore", message="Cluster .* is empty!")


# Function to check if a row already exists in a file
def is_row_in_file(file_path, row_data, delimiter=';'):
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', newline='') as file:
        for line in file:
            if line.strip() == delimiter.join(row_data):
                return True
    return False


# Function to write a row to a CSV file if it's not already present
def write_unique_row_to_csv(file_path, row_data, delimiter=';'):
    if not is_row_in_file(file_path, row_data, delimiter):
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(row_data)


# Function to find the medoid of a cluster
def find_medoid(cluster_columns, correlation_matrix):
    # Subset of the correlation matrix for the given cluster
    cluster_corr = correlation_matrix.loc[cluster_columns, cluster_columns]
    # Compute the sum of distances for each column in the cluster
    total_distances = 1 - np.abs(cluster_corr).sum(axis=1)
    # Find the medoid: the column with the minimum sum of distances
    medoid = total_distances.idxmin()
    return medoid


# Function to determine the type of a column based on its data
def determine_column_type(col):
    if col.dtype == 'bool':
        return 'B'  # Boolean type
    elif col.dtype == 'object' and col.str.len().max() == 1:
        return 'C'  # Categorical (single character) type
    else:
        return 'D'  # Default type (numerical or mixed)


# Main function to perform clustering and save the medoids
def cluster_and_save(input_file, output_file, k, dataset_type, nome_output, categorical_columns=None):
    """
    Performs clustering on a dataset using K-Medoids and saves the medoids to a file.

    Args:
        input_file (str): Path to the input dataset file.
        output_file (str): Path to save the medoids.
        k (int): Number of medoids (clusters) to find.
        dataset_type (str): Type of dataset ('numerico' for numerical, 'misto' for mixed).
        nome_output (str): Name identifier for output.
        categorical_columns (list, optional): List of indices of categorical columns (only for mixed datasets).
    """

    # Read the headers file line by line
    headers = []
    with open('../Preprocessing/Headers/HeadersClust.csv', 'r') as f:
        for line in f:
            valori = line.strip().split(';')
            headers.append(valori)

    # Find the specific header based on the dataset name
    colonne = None
    for header in headers:
        if header[-1] == nome_dataset_MV:  # The last value is the dataset name
            colonne = [col for col in header[:-1] if col]  # Exclude the last element and remove empty values
            break

    if colonne is None:
        raise ValueError(f"Header for dataset '{nome_dataset_MV}' not found in the header file.")

    # Read the dataset
    data = pd.read_csv(input_file, sep=';')
    data = data[~data.eq('?').any(axis=1)]  # Remove rows containing missing values represented by '?'

    # Data transformation based on dataset type
    if dataset_type == "numeric":
        data_transformed = data.values  # No transformation needed for purely numerical data
        transformer = None
    elif dataset_type == "mixed":
        if categorical_columns is None:
            raise ValueError("You must specify categorical column indices for a mixed dataset.")

        # Identify numerical columns by excluding categorical ones
        categorical_columns = set(categorical_columns)
        numerical_columns = [i for i in range(data.shape[1]) if i not in categorical_columns]

        # Apply preprocessing: standardization for numerical columns, one-hot encoding for categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_columns),
                ("cat", OneHotEncoder(), list(categorical_columns)),
            ]
        )
        data_transformed = preprocessor.fit_transform(data)
        transformer = preprocessor
    else:
        raise ValueError("Dataset type must be 'numeric' or 'mixed'.")

    # Adjust k if it exceeds the number of available data points
    if k > len(data_transformed):
        k = len(data_transformed)
        print("Adjusting k to:", k)

    # Apply K-Medoids clustering
    kmedoids = KMedoids(n_clusters=k, random_state=42, metric="euclidean")
    kmedoids.fit(data_transformed)

    # Retrieve medoid indices
    medoid_indices = kmedoids.medoid_indices_

    # Retrieve original rows corresponding to the medoids
    medoids_original = data.iloc[medoid_indices]

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the medoids to the output file
    try:
        medoids_original.to_csv(output_file, sep=';', index=False)
        print(f"File successfully saved at {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    # Save column types and headers
    column_types_output_path = '../Preprocessing/ColumnTypes/ColumnTypesClust.csv'
    header_output_path = '../Preprocessing/Headers/HeadersClust.csv'

    column_types = [determine_column_type(data[col]) for col in data.columns]
    write_unique_row_to_csv(column_types_output_path, column_types + [nome_output])
    write_unique_row_to_csv(header_output_path, data.columns.tolist() + [nome_output])

    print(f"Medoids saved in {output_file}. Medoid indices: {medoid_indices}")


# List of k values for clustering
k_arr = [100, 300, 500, 1000]

# Main execution loop
if __name__ == "__main__":
    for k in k_arr:
        nome_dataset = 'S4-ADL5-MNAR_20000_130'
        nome_dataset_MV = f'{nome_dataset}_520000_1_Balanced_cluster_10'
        nome_dataset_output = f"{nome_dataset_MV}_{k}_row_clustering"

        input_file = f"../Datasets/Missing_Datasets/{nome_dataset}/{nome_dataset}_520000_1/{nome_dataset_MV}.csv"
        output_file = f"../Datasets/Missing_Datasets/{nome_dataset}/{nome_dataset}_520000_1/{nome_dataset_output}.csv"

        # set dataset_type to "numeric" if all the dimensions are numeric, otherwise use "mixed" if there are textual values
        dataset_type = "numeric"
        # positions of dimensions with textual values. [0,3,5] are for the Telemetry dataset
        categorical_columns = [0, 3, 5]

        # Run clustering and save medoids
        cluster_and_save(input_file, output_file, k, dataset_type, nome_dataset_output, categorical_columns)
