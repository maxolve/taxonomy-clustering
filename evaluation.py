import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import silhouette_score, davies_bouldin_score
import networkx as nx
from preprocessing import preprocess_data
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform
from clustering_algorithms import (
    run_kmeans_rand, 
    run_kmeans_pp, 
    run_pyclust_kmedoids, 
    run_stepmix, 
    run_kmodes, 
    run_softmodes, 
    run_lloyd_kmeans, 
    run_lloyd_kmeanspp, 
    run_kPbC, 
    run_kscc,
    run_lshfkcenters,
    run_CDClustering
)

def calculate_category_utility(data, clusters):
    """
    Calculate the Category Utility for clustering results.

    :param data: DataFrame, where each column is a categorical feature
    :param clusters: Array-like, cluster labels for each instance in the data
    :return: Category Utility score
    """
    n = len(data)  # Total number of samples
    total_cu = 0
    unique_clusters = np.unique(clusters)

    # Calculate overall probability p(x_j = v|T) for each value in each feature
    overall_probs = {col: data[col].value_counts(normalize=True) ** 2 for col in data.columns}

    for cluster in unique_clusters:
        cluster_data = data[clusters == cluster]
        cluster_size = len(cluster_data)
        cluster_weight = cluster_size / n

        # Calculate conditional probabilities p(x_j = v|C_i) for the cluster
        cluster_probs = {col: cluster_data[col].value_counts(normalize=True) ** 2 for col in data.columns}

        # Sum over all features
        feature_sums = 0
        for col in data.columns:
            # Sum over all values that appear in the dataset (not just the cluster)
            value_sums = 0
            for value in data[col].unique():
                p_cluster_v = cluster_probs[col].get(value, 0)
                p_overall_v = overall_probs[col].get(value, 0)
                value_sums += p_cluster_v - p_overall_v
            
            feature_sums += value_sums
        
        # Weight the contribution of this cluster by its size
        total_cu += cluster_weight * feature_sums

    return total_cu






# from scratch implementation for testing purposes
def category_utility_cpp(ds, clustering, m):
    N = len(ds)  # number of data items
    dim = ds.shape[1]  # number of attributes

    # Step 1: Compute number of items in each cluster
    clusterCts = [0] * m
    for i in range(N):
        cid = clustering[i]
        clusterCts[cid] += 1

    # Possible quick exit
    for k in range(m):
        if clusterCts[k] == 0:
            return 0.0  # bad clustering

    # Step 2: Compute number of unique values for each attribute
    uniqueVals = [0] * dim
    for j in range(dim):
        maxID = ds.iloc[:, j].max()
        uniqueVals[j] = maxID + 1

    # Step 3: Compute unconditional counts
    attCts = np.zeros((dim, max(uniqueVals)), dtype=int)
    for j in range(dim):
        for i in range(N):
            id = ds.iloc[i, j]
            attCts[j][id] += 1

    # Step 4: Compute conditional counts
    condCts = np.zeros((m, dim, max(uniqueVals)), dtype=int)
    for k in range(m):
        for j in range(dim):
            for i in range(N):
                if clustering[i] != k:
                    continue
                id = ds.iloc[i, j]
                condCts[k][j][id] += 1

    # Step 5: Compute unconditional sum of squared probabilities
    unCondSum = 0.0
    for j in range(dim):
        for jj in range(attCts[j].shape[0]):
            unCondSum += (1.0 * attCts[j][jj] / N) * (1.0 * attCts[j][jj] / N)

    # Step 6: Compute conditional sum of squared probabilities for each cluster
    condSum = [0.0] * m
    for k in range(m):
        if clusterCts[k] == 0:
            raise Exception("empty cluster computing cond SSP")
        sum = 0.0
        for j in range(dim):
            for jj in range(condCts[k][j].shape[0]):
                sum += (1.0 * condCts[k][j][jj] / clusterCts[k]) * (1.0 * condCts[k][j][jj] / clusterCts[k])
            condSum[k] = sum

    # Step 7: Compute probability of each cluster
    probC = [0.0] * m
    for k in range(m):
        probC[k] = 1.0 * clusterCts[k] / N

    # Step 8: Compute the Category Utility
    left = 1.0 / m
    right = 0.0
    for k in range(m):
        right += probC[k] * (condSum[k] - unCondSum)

    cu = left * right
    return cu



class ClusteringModularity:
    def __init__(self, dataframe):
        self.df = dataframe
        self.encoded_df = self.encode_categorical_dataframe(dataframe)
    
    def encode_categorical_dataframe(self, df):
        encoders = {col: LabelEncoder().fit(df[col]) for col in df}
        encoded_df = df.copy()
        for col, encoder in encoders.items():
            encoded_df[col] = encoder.transform(df[col])
        return encoded_df

    def hamming_distance(self, x, y):
        """Calculate the Hamming distance between two arrays."""
        return np.sum(x != y)

    def compute_pairwise_hamming_distances(self, data):
        """Compute pairwise Hamming distances for all pairs in the dataset."""
        n = len(data)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.hamming_distance(data.iloc[i], data.iloc[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def compute_cdf(self, distances):
        """Compute the cumulative distribution function of the Hamming distances."""
        hist, bins = np.histogram(distances.ravel(), bins=range(int(np.max(distances)) + 2), density=True)
        cdf = np.cumsum(hist * np.diff(bins))
        return cdf, bins

    def estimate_distance_threshold(self, cdf, bins, k):
        """Estimate the Hamming distance threshold R."""
        for i in range(len(cdf)):
            if cdf[i] >= 1/k:
                return bins[i]
        return bins[-1]

    def create_adjacency_matrix(self, distances, threshold):
        """Create an adjacency matrix using the given distance threshold."""
        adjacency_matrix = (distances <= threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
        return adjacency_matrix

    def calculate_modularity(self, adjacency_matrix, cluster_labels):
        """Calculate the modularity of the clustering results."""
        G = nx.from_numpy_array(adjacency_matrix)
        communities = {}
        for node, label in enumerate(cluster_labels):
            if label not in communities:
                communities[label] = []
            communities[label].append(node)
        community_list = list(communities.values())
        modularity = nx.algorithms.community.modularity(G, community_list)
        return modularity
    
def modularity_with_threshold(data, labels, k):

    clustering_modularity = ClusteringModularity(data)

    distances = clustering_modularity.compute_pairwise_hamming_distances(clustering_modularity.encoded_df)

    # Compute the cumulative distribution function (CDF)
    cdf, bins = clustering_modularity.compute_cdf(distances)

    # Estimate the distance threshold R
    threshold = clustering_modularity.estimate_distance_threshold(cdf, bins, k)
    print(f"Estimated Distance Threshold: {threshold}")

    if threshold == 0:
        threshold = 1
        print("Adjusting the threshold to a minimum value of 1 to ensure connectivity.")

    adjacency_matrix = clustering_modularity.create_adjacency_matrix(distances, threshold)
    print("Adjacency Matrix:")
    print(adjacency_matrix)

    if np.sum(adjacency_matrix) == 0:
        print("Warning: The adjacency matrix has no edges. Adjust the threshold or check the data.")

    cluster_labels = labels

    modularity_score = clustering_modularity.calculate_modularity(adjacency_matrix, cluster_labels)
    return modularity_score

# Create an adjacency matrix using Hamming distance
def create_adjacency_matrix_using_hamming(df):

    distances = pdist(df.values, metric='hamming')

    distance_matrix = squareform(distances)

    similarity_matrix = 1 - distance_matrix
    return similarity_matrix


#  Calculate modularity
def calculate_modularity(dataframe, cluster_labels):
    
    adjacency_matrix = create_adjacency_matrix_using_hamming(dataframe)

    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    communities = {}
    for node, label in enumerate(cluster_labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    community_list = list(communities.values())
    modularity = nx.algorithms.community.modularity(G, community_list)
    return modularity




def compute_boxplot_stats(values):
    """
    Calculate statistics needed for a boxplot:
    lower quartile, upper quartile, median, lower whisker, upper whisker
    """
    quartiles = np.percentile(values, [25, 50, 75])
    lower_quartile, median, upper_quartile = quartiles
    iqr = upper_quartile - lower_quartile
    lower_whisker = np.min(values[values >= lower_quartile - 1.5 * iqr])
    upper_whisker = np.max(values[values <= upper_quartile + 1.5 * iqr])
    return lower_quartile, upper_quartile, median, lower_whisker, upper_whisker



# Evaluation function to run algorithms and save results
def evaluate_algorithms(df, dataset_name, metrics, k_values, n_iterations=100):
    # Define the clustering algorithms and the data type they require
    algorithms = {
        'kmeans_rand': ('onehot_encoded_pandas_df', run_kmeans_rand),
        'kmeans_pp': ('onehot_encoded_pandas_df', run_kmeans_pp),
        'kmedoids': ('onehot_encoded_pandas_df', run_pyclust_kmedoids), 
        'kmodes': ('categorical_pandas_df', run_kmodes),  
        'softmodes': ('categorical_numpy', run_softmodes),
        'kmeans_rand': ('onehot_encoded_numpy', run_lloyd_kmeans),
        'kmeans_pp': ('onehot_encoded_numpy', run_lloyd_kmeanspp),
        'stepmix': ('categorical_numpy', run_stepmix),  
        'kpbc': ('categorical_pandas_df', run_kPbC),  
        'kscc': ('categorical_pandas_df', run_kscc),
        'lshfkcenters': ('categorical_pandas_df', run_lshfkcenters),
        'CDCluster': ('categorical_pandas_df', run_CDClustering)
    }

    # Keep a copy of the original categorical dataset for evaluation
    categorical_data_num = preprocess_data(dataset_name, df, return_type='categorical_numpy')
    categorical_data = preprocess_data(dataset_name, df, return_type='categorical_pandas_df')

    for k in k_values:
        for metric in metrics:
            metric_results = []

            for algo_name, (required_data_type, algo_func) in algorithms.items():
                algo_metric_results = []
                for i in range(n_iterations):
                    try:
                        processed_data = preprocess_data(dataset_name, df, return_type=required_data_type)

                        labels = algo_func(processed_data, k)

                        if metric == 'silhouette':
                            print("silhouette")
                            score = silhouette_score(categorical_data_num, labels, metric='hamming')
                        elif metric == 'davies_bouldin':
                            score = davies_bouldin_score(categorical_data_num, labels)
                        elif metric == 'category_utility':
                            score = calculate_category_utility(categorical_data, labels)
                        elif metric == 'category_utility_cpp':
                            score = category_utility_cpp(categorical_data, labels, k)
                        elif metric == 'modularity':
                            score = calculate_modularity(categorical_data, labels)
                        elif metric == 'modularity_with_threshold':
                            score = modularity_with_threshold(categorical_data, labels, k)
                        else:
                            raise ValueError(f"Unknown metric: {metric}")

                        algo_metric_results.append(score)
                    except Exception as e:
                        print(f"Error running {algo_name} for {k} clusters: {e}")
                        continue

                if algo_metric_results:
                    algo_metric_results = np.array(algo_metric_results)
                    lower_quartile, upper_quartile, median, lower_whisker, upper_whisker = compute_boxplot_stats(algo_metric_results)
                    metric_results.append([
                        algo_name, lower_quartile, upper_quartile, median, lower_whisker, upper_whisker
                    ])

            # Save boxplot stats to CSV
            if metric_results:
                output_dir = os.path.join("output", dataset_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{metric}_boxplot_stats_{dataset_name}-{k}.csv")
                metric_df = pd.DataFrame(metric_results, columns=['Algorithm', 'lower_quartile', 'upper_quartile', 'median', 'lower_whisker', 'upper_whisker'])
                metric_df.to_csv(output_path, index=False)

