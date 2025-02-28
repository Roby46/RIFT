# RIFT

 **Work in progress**

This repository contains supplemental material related to the paper *RIFT: RFD-based Imputation Framework for Multivariate Time Series*.

## DATASETS

The **Datasets** folder contains both the original datasets (**Starting Datasets** folder) and the datasets with missing values (**Missing Datasets** folder), which were used for the experimental evaluation described in the paper.

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

- **Clustering_Balanced.py**: Implements the clustering strategy used in the paper. It partitions the multivariate time series into clusters based on correlations between dimensions while balancing the size of the resulting clusters.
- **Clustering_rows.py**: Implements the clustering strategy used in step 2 of the framework, selecting the *k* most representative time instants.

Both scripts save the processed data in the **Missing Datasets** folder, inside the subfolder corresponding to the dataset they were applied to.
