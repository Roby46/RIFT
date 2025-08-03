# RIFT: RFD-based Imputation Framework for Time Series

 <img src="Images/Overview.png" alt="...">

This repository contains supplemental material related to the paper *RIFT: RFD-based Imputation Framework for  Time Series*.

## EXPERIMENTAL EVALUATION

In the following, we provide the experiments that were omitted in the paper due to space constraints.  

### Datasets details

<img src="Images/Datasets.png" alt="...">

The Figure above shows the details of the datasets employed in our evaluation, also reporting the total number of Missing Values for each configuration. These datasets were selected to cover different domains, such as human activity recognition, environmental sensing, industrial automation, and air quality monitoring (i.e., Opportunity, Telemetry, Genesis,and Air Quality, resp.).

### Evaluation metrics

To evaluate the imputations, we consider Precision, Recall and RMSE. Formally, let _tr_ be the number of correct imputations, _imp_ the numberof imputations performed and _miss_ be the number of MVs originally present inthe dataset, then:

<img src="Images/Evaluation Metrics.png" alt="...">


### Experiment with MNAR assumption

<img src="Images/Legend.png" alt="...">

<img src="Images/MNAR Experiment.png" alt="...">

In addition to our main experiments conducted under the MAR assumption, we performed a complementary evaluation on the Opportunity dataset to investigate Missing Not at Random (MNAR) scenarios. More specifically, we injected missing values by simulating sensor failures for a random period of time, until reaching a missing rate percentage of 5%, 10%, and 20%. The figure above shows the results in terms of Precision, Recall, and RMSE. By comparing the latter with those under the MAR assumption, it is possible to generally notice a similar behavior of all metrics, even with a slight degradation due to the more complex scenario. However, only SPLINE, BFILL, and FFILL showed a significant drop in both Precision and Recall.

This degradation is due to the fact that, unlike the MAR scenario where missing values are more evenly distributed, in the MNAR case, there are prolonged periods of missing data due to the simulated sensor failures. As a result, imputation methods such as BFILL and FFILL, which rely on propagating the nearest observed values, struggled to accurately reconstruct the data. Similarly, SPLINE was affected by the lack of regularly distributed reference points, leading to less precise estimations.

As regards the proposed approach, RIFT showed a similar behavior, providing few but very precise imputations. Even in this scenario, by applying RIFT as a pre-processing step, we were able to improve the baselines’ imputations, with an average improvement of 4.36% for Precision and Recall, respectively, and an average reduction of 6.74% for the RMSE.


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

### Requirements

Make sure to have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- scikit-learn-extra
- scipy

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

## DISCOVERY

After performing the clustering step, it is possible to run an RFD discovery process to extract the RFDs holding for each cluster. For this step, any RFD discovery algorithm can be employed (since every algorithm extracts a set of complete, correct and minimal RFDs the results will be the same). The syntax used to represent the discovered RFDs should be the following:

```
RHS;COL0;COL1;COL2;COL3;COL4;COL5;COL6;COL7;COL8;COL9;COL10;COL11;COL12;COL13;COL14;COL15
COL0;1.0;0.0;?;0.0;0.0;0.0;?;?;?;?;?;?;?;?;?;?
```
Specifically, each RFD is represented by a row. The column **RHS** contains the attribute present in the RHS of the RFD. The attributes whose values are "?" are not involved in the RFD, whereas otherwise its associated similarity threshold is reported. For instance, the example above represents a single RFD: COL1(0.0), COL3(0.0), COL4(0.0), COL5(0.0)-->COL0(1.0).

## IMPUTATION

After performing all the required discovery processes, it is possible to start the imputation step. To this end, the .jar files RIFT and  RIFT+INDEXING can execute the imputation processes on a set of clusters in a sequential way. The imputation process requires (1) the RFD discovered for each cluster, which have to be put inside the RFD folre; (2) the files inside the  **Initial_Tuples** folder, (3) the Header and the ColumnTypes files, which have to be put in the CandidateDataset folder and contains all the informations of all datasets (these files are automatically updated when injecting missing values in the preprocessing step); (4) the datasets, which have to be put in the dataset folder. This repository contains two .jar files: SequentialImputationExecutor performs the imputations process by employing only the decay mechanism, while SequentialImputationExecutorIndexing also adds the use of the indexing strategy. Both jars requires the following arguments:
```
        String basePath = "C:\\Users\\rstan\\eclipse-workspace\\Imputation\\";   //Main Directory that contains the folders described above
        String delim = ";";  //Dataset delimiter
        String nullValue = "?"; //Symbol that represents missing values
        int windowSize = 25000; //Dataset size (temporal instants)
        int clusterRows = 100; //rows used for the discovery step. In this case, the algorithm will use the RFDs discovered with 100 medoids (k)
        int[] clusterIds = {1,2}; //cluster IDs. Here we have two clusters
        int[] attributeCounts = {6,11}; //Number of dimensions in each cluster (e.g. cluster 1 has 6 dimensions)
        String dataset = CATS_16000_17_13600_1 //Dataset name ({name}\_{length}\_{number of dimensions}\_{number of missing values}\_{version})
        thresholds=0.5 //Threshold used for the discovery in the discovery step (required by RENUVER)

```

