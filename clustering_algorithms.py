import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as kmeans_sklearn
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmedoids import kmedoids
import warnings
np.warnings = warnings
from kmodes.kmodes import KModes as kkmodes
from local_packages.SoftModes.KModes_class import KModes as kmodes_new
from local_packages.SoftModes.kmeans_functions import Kmeans
from kmodes.util.dissim import ng_dissim
from stepmix.stepmix import StepMix
from LSHkCenters.LSHkCenters import LSHkCenters
from local_packages.cdclustering.cdclustering import CDClustering  # Custom package for CDClustering
from local_packages.k_PbC_master.kPbC_unlabels_sil import (
    k_PbC,  
    transform_data,  
    fpmax,  
    Initiate_Centers_with_Transactions,  
    remove_overlapping_transactions,  
    initial_assignment  
)
from local_packages.k_SCC_master.Code.sake_kscc_it import KSCC_Kernel_IT
from sklearn.metrics import pairwise_distances


def run_kmeans_rand(data, num_clusters):
    """Run KMeans algorithm on one-hot encoded data."""
    
    # One-hot encode using pandas
    one_hot_encoded_data = pd.get_dummies(data)
    
    # Initialize KMeans with random initialization
    kmeans = kmeans_sklearn(n_clusters=num_clusters, init='random', n_init=1, max_iter=50)
    
    # Fit the model on the one-hot encoded data
    labels = kmeans.fit(one_hot_encoded_data).labels_
    
    return labels


def run_kmeans_pp(data, num_clusters):
    print(f"Running KMeans with random initialization on data:")
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    print(f"First few rows of data:\n{data.head() if isinstance(data, pd.DataFrame) else data[:5]}")
    
    one_hot_encoded_data = pd.get_dummies(data)

    """Run KMeans++ algorithm."""
    kmeans = kmeans_sklearn(n_clusters=num_clusters, init='k-means++', n_init=1, max_iter=50)
    return kmeans.fit(one_hot_encoded_data).labels_


def run_pyclust_kmedoids(data, num_clusters):
    """
    Run KMedoids with distance matrix support, using Hamming distance.
    
    :param data: Input DataFrame (categorical data)
    :param num_clusters: The number of clusters to fit
    :return: Cluster labels
    """
    print('pyclust kmedoids running')
    # Generate the Hamming distance matrix from the input data
    distance_matrix = pairwise_distances(data, metric='hamming')
    print('distance_matrix:', distance_matrix)

    # Initialize cluster centers using k-means++ initializer
    centers = kmeans_plusplus_initializer(distance_matrix, num_clusters, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize(return_index=True)
    print('centers:', centers)
    # Create and run the KMedoids algorithm using the precomputed distance matrix
    kmedoids_instance = kmedoids(distance_matrix, centers, data_type='distance_matrix')
    kmedoids_instance.process()
    
    # Get the cluster assignments for each point
    clusters = kmedoids_instance.get_clusters()

    # Convert clusters to label format
    labels = np.empty(len(distance_matrix), dtype=int)
    for cluster_label, cluster_indices in enumerate(clusters):
        for index in cluster_indices:
            labels[index] = cluster_label
            
    return labels


def run_kmedoids_hamming(data, num_clusters):
    one_hot_encoded_data = pd.get_dummies(data)

    """Run KMedoids with Hamming distance."""
    kmedoids = KMedoids(n_clusters=num_clusters, metric='hamming', init='heuristic', max_iter=300).fit(one_hot_encoded_data)
    return kmedoids.labels_

def run_kmodes(data, num_clusters):
    """Run KModes algorithm."""
    km = kkmodes(n_clusters=num_clusters, init='Cao', cat_dissim=ng_dissim)
    labels = km.fit_predict(data)
    return labels

def run_lshfkcenters(data, num_clusters):
    """Run LSHKCenters algorithm."""
    num_rows = data.shape[0]
    # Create a new numpy array of zeros with the same number of rows (y not used)
    y_lshfkcenters = np.zeros(num_rows)
    # Apply factorize to each column in the DataFrame
    for column in data.columns:
        data[column], _ = pd.factorize(data[column])   
    data = data.to_numpy()
    lshfkcenters = LSHkCenters(data, y_lshfkcenters, n_init=100, k=num_clusters)
    lshfkcenters.SetupLSH()
    return lshfkcenters.DoCluster()

def run_kPbC(dataset, num_clusters):
    """
    Run kPbC clustering algorithm.
    
    :param dataset: Pandas DataFrame of categorical data
    :param num_clusters: Number of clusters to form
    :return: Cluster labels
    """
    k = num_clusters
    min_support = 0.0003

    def preprocess_column(column, col_index):
        categorical_values = set(column)
        value_mapping = {value: f"{value}_{col_index}" for value in categorical_values}
        return column.map(value_mapping)

    for col_index, column in enumerate(dataset.columns, start=1):
        dataset[column] = preprocess_column(dataset[column], col_index)

    # Convert dataset to a list of tuples for kPbC
    X = [tuple(row) for row in dataset.values.tolist()]

    transformed_data, columns = transform_data(X, sparse=False)
    df = pd.DataFrame(transformed_data, columns=columns)

    # Find maximal frequent itemsets using FPMax algorithm
    mfi_df = fpmax(df, min_support=min_support, use_colnames=True, max_len=None, verbose=0)

    if mfi_df.empty or mfi_df.shape[0] < k:
        raise RuntimeError(f"Cannot find enough clusters at minsup={min_support}. Please adjust the minSup threshold.")

    # Initialize initial groups based on maximal frequent itemsets
    initial_groups = Initiate_Centers_with_Transactions(mfi_df, k, X)
    initial_groups = remove_overlapping_transactions(initial_groups)

    output_file = "initial_clusters.txt"
    initial_assignment(initial_groups, X, output_file)

    labels = k_PbC(X, k)

    print('kPbC labels:', labels)
    return labels



def run_kscc(X, num_clusters):
    """
    Function to run KSCC clustering algorithm.
    
    :param X: Input data as a numpy array or pandas DataFrame
    :param num_clusters: The number of clusters to fit
    :return: Cluster labels
    """
    n_init = 100  # Set the number of initializations or iterations
    
    kscc = KSCC_Kernel_IT(n_clusters=num_clusters, n_init=n_init, verbose=0)
    
    labels = kscc.fit_predict(X)
    
    return labels

def run_stepmix(categorical_data, num_clusters):
    """Run StepMix clustering algorithm."""
    print("Running StepMix with data:")
    print(f"Data type: {type(categorical_data)}")
    print(f"Data shape: {categorical_data.shape}")
    print(f"First few rows of data:\n{categorical_data.head() if isinstance(categorical_data, pd.DataFrame) else categorical_data[:5]}")

    model = StepMix(n_components=num_clusters, n_steps=3, measurement="categorical")
    
    try:
        model.fit(categorical_data)
    except Exception as e:
        print(f"Error during StepMix fitting: {e}")
        return None
    
    print("stepmix labels:",model.predict(categorical_data))
    
    return model.predict(categorical_data)

def run_lloyd_kmeans(categorical_data, num_clusters):
    """Run KMeans with random initialization using Lloyd's KMeans implementation."""
    n_init, max_iter = 25, 200
    lloyd_rand = Kmeans(init="random", n_clusters=num_clusters, n_init=n_init, max_iter=max_iter)
    lloyd_rand.fit(X=categorical_data.astype(float), true_labels=None)
    return lloyd_rand.labels_

def run_lloyd_kmeanspp(categorical_data, num_clusters):
    """Run KMeans++ using Lloyd's KMeans implementation."""
    n_init, max_iter = 25, 200
    lloyd = Kmeans(init="k-means++", n_clusters=num_clusters, n_init=n_init, max_iter=max_iter)
    lloyd.fit(X=categorical_data.astype(float), true_labels=None)
    return lloyd.labels_

def run_softmodes(data, num_clusters):
    """Run SoftModes clustering algorithm."""
    kmodes = kmodes_new(init="D1-seeding", n_clusters=num_clusters, algorithm="softmodes", t=3)
    kmodes.fit(data)
    return kmodes.labels_

def run_CDClustering(data, num_clusters):
    """Run CDClustering algorithm."""
    cd_clustering = CDClustering()
    distances = cd_clustering.compute_pairwise_hamming_distances(data)
    cdf, bins = cd_clustering.compute_cdf(distances)
    R = cd_clustering.estimate_distance_threshold(cdf, bins, num_clusters)
    G = cd_clustering.build_graph(data, R)
    partition = cd_clustering.detect_communities(G)
    return cd_clustering.form_clusters_with_labels(data, partition, num_clusters)
