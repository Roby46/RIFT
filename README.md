![image](https://github.com/user-attachments/assets/8c2d8387-cc84-4713-9443-810a4167b8ad)# RIFT

 **Work in progress**

This repository contains supplemental material related to the paper *RIFT: RFD-based Imputation Framework for Multivariate Time Series*.

## DATASETS

The **Datasets** folder contains both the original datasets (**Starting Datasets** folder) and the datasets with missing values (**Missing Datasets** folder), which were used for the experimental evaluation described in the paper.

The name of the original datasets is composed as follows: "{name}\_{length}\_{number of dimensions}.csv". 
The name of the missing datasets is composed as follows: "{name}\_{length}\_{number of dimensions}\_{number of missing values}\_{version}.csv".

## MISSING VALUES INJECTION

The **Preprocessing** folder includes two scripts for injecting missing values into the datasets in the **Starting_Datasets** folder:

- **injector.py**: Injects missing values according to the MCAR (Missing Completely at Random) mechanism. The parameters of the script are the following:

```
dataset = 'Air'              # Name of the datasets
delimiter = ';'              # Delimiter of the dataset
percentages = [5,10,20]      # Missing rates 
lengths=[3000]               # Lenght of the series
columns=[13]                 # Number of columns/dimensions
iterations = [1]             # This is just a number that will be included in the output file's name, so it is possible to generate different version of the same dataset with the same missing rates
null_value = '?'             # The missing value will be represented with "?"
```
- **injector_block.py**: Injects missing values based on the MNAR (Missing Not At Random) mechanism. Specifically, this script simulates sensor malfunctions by introducing consecutive missing values for a randomly determined time period. This script has the same parameters of  **injector.py**, with the addition of the following one:
```
max_malfunction_duration = 50 # Maximum malfunction duration
```

Both scripts save the resulting datasets in the **Missing Datasets** folder. Additionally, removed values and their original positions are stored in the **Initial_Tuples** folder. These files are necessary for running the imputation algorithm.

## CLUSTERING

The **Clustering** folder contains scripts for implementing the first two steps of the RIFT framework:

- **Dimensions_Clustering.py**: Implements the clustering strategy used in the paper. It partitions the multivariate time series into clusters based on correlations between dimensions while balancing the size of the resulting clusters. The script has the following parameters:

```
min_cluster_size = 6                          # Minimum cluster size
max_cluster_size = 14                         # Maximum cluster size
dataset_name = 'S4-ADL5_20000_130'            # Name of the original dataset
dataset_name_MV = f'{dataset_name}_260000_1'  # Name of the dataset + version details (number of missing values and version)
```
  
- **Time_Instants_Clustering.py**: Implements the clustering strategy used in step 2 of the framework, selecting the *k* most representative time instants. The script has the following parameters:
```
dataset_name = 'S4-ADL5_20000_130'                                       # Name of the original dataset
MVs=520000
version=1
dataset_name_MV = f'{dataset_name}_{MVs}_{version}_Balanced_cluster_10'  # Name of the cluster
k_arr = [100, 300, 500, 1000]                                            # Number of medoids to save
dataset_type = "numeric"                                                 # set to "numeric" if all dimensions are numeric, otherwise use "mixed" 


```
Both scripts save the processed data in the **Missing Datasets** folder, inside the subfolder corresponding to the dataset they were applied to.
