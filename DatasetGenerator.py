import numpy as np
import pandas as pd
import timeit
import random
import copy
from sorters import ALGORITHMS
from feature_extractor import FeatureExtractor

class DatasetGenerator:
    def __init__(self, sizes=[100, 500, 1000, 5000, 10000]):
        self.sizes = sizes
        self.distributions = [
            "uniform", "nearly_sorted", "reverse_sorted", 
            "many_duplicates", "gaussian", "skewed"
        ]

    def generate_array(self, size, dist_type):
        if dist_type == "uniform":
            return np.random.randint(0, size * 10, size).tolist()
            
        elif dist_type == "nearly_sorted":
            arr = list(range(size))
            swaps = int(size * 0.05)
            for _ in range(swaps):
                i, j = random.sample(range(size), 2)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
            
        elif dist_type == "reverse_sorted":
            return list(range(size, 0, -1))
            
        elif dist_type == "many_duplicates":
            alphabet = np.random.randint(0, 50, 10)
            return np.random.choice(alphabet, size).tolist()
            
        elif dist_type == "gaussian":
            arr = np.random.normal(loc=0, scale=1000, size=size)
            return np.round(arr).astype(int).tolist()
            
        elif dist_type == "skewed":
            # Power law distribution mimics real-world heavy-tailed data
            arr = np.random.zipf(a=2.0, size=size)
            return arr.tolist()

    def benchmark_algorithms(self, arr, runs=5):
        times = {}
        for name, func in ALGORITHMS.items():
            timer = timeit.Timer(lambda: func(copy.deepcopy(arr)))
            try:
                run_times = timer.repeat(repeat=runs, number=1)
                times[name] = np.median(run_times)
            except RecursionError:
                times[name] = float('inf') 
                
        best_algo = min(times, key=times.get)
        return best_algo, times

    def build_dataset(self, samples_per_config=10):
        """Generates the full dataset and returns it as a Pandas DataFrame."""
        dataset = []
        
        for size in self.sizes:
            for dist in self.distributions:
                for _ in range(samples_per_config):
                    arr = self.generate_array(size, dist)
                    
                    features = FeatureExtractor.extract_all_features(arr)
                    
                    best_algo, all_times = self.benchmark_algorithms(arr)
                    
                    record = {**features, "best_algorithm": best_algo}
                    dataset.append(record)
                    
        return pd.DataFrame(dataset)