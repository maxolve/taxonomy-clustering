from datasets import load_dataset
from evaluation import evaluate_algorithms
from config import CLUSTER_NUM_RANGE, N_ITERATIONS

def main():
    # List of datasets
    datasets = ["zoo", "soybean-small", "torno", "gerlach", "fischer", 
                "thiebes", "schmidt-kraeplin", "maas", "muller"]  # Add other datasets as needed
    
    all_metrics = [
        'silhouette', 
        'davies_bouldin', 
        'category_utility', 
        'modularity_with_threshold'
    ]
    
    for dataset_name in datasets:
        print(f"Loading dataset: {dataset_name}")
        df = load_dataset(dataset_name)
        
        print(f"Evaluating algorithms for dataset: {dataset_name}")
        evaluate_algorithms(
            df=df,
            k_values=CLUSTER_NUM_RANGE,
            dataset_name=dataset_name,
            metrics=all_metrics,
            n_iterations=N_ITERATIONS
        )
        
if __name__ == "__main__":
    main()
