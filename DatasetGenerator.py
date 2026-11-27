import numpy as np
import pandas as pd
import timeit
from sorters import ALGORITHMS
from feature_extractor import FeatureExtractor


class DatasetGenerator:
    """Builds (array features -> fastest algorithm) rows. The array families below
    are chosen so every algorithm has a realistic situation where it wins."""

    def __init__(self, sizes=(10, 30, 60, 150, 400, 1000, 2000), seed=42):
        self.sizes = list(sizes)
        self.rng = np.random.default_rng(seed)
        self.distributions = [
            "uniform", "uniform_wide", "float_uniform", "nearly_sorted", "sorted",
            "reverse_sorted", "sawtooth", "sorted_with_tail", "many_duplicates",
            "few_distinct_wide", "gaussian", "skewed",
        ]

    def generate_array(self, size, dist):
        rng = self.rng
        wide = 10**9
        if dist == "uniform":
            return rng.integers(0, size * 10, size).tolist()
        if dist == "uniform_wide":
            return rng.integers(0, wide, size).tolist()
        if dist == "float_uniform":
            return rng.random(size).tolist()
        if dist == "sorted":
            return list(range(size))
        if dist == "reverse_sorted":
            return list(range(size, 0, -1))
        if dist == "nearly_sorted":
            arr = list(range(size))
            for _ in range(max(1, int(size * rng.uniform(0.01, 0.08)))):
                i, j = rng.choice(size, 2, replace=False)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        if dist == "sawtooth": 
            k = int(min(rng.choice([4, 8, 16, 32]), max(size // 2, 2)))
            parts = np.array_split(rng.integers(0, wide, size), k)
            return np.concatenate([np.sort(p) for p in parts]).tolist()
        if dist == "sorted_with_tail":  
            cut = int(size * 0.9)
            return np.concatenate([np.sort(rng.integers(0, wide, cut)), rng.integers(0, wide, size - cut)]).tolist()
        if dist == "many_duplicates": 
            return rng.choice(rng.integers(0, 50, 10), size).tolist()
        if dist == "few_distinct_wide": 
            return rng.choice(rng.integers(0, wide, int(rng.integers(5, 21))), size).tolist()
        if dist == "gaussian":
            return np.round(rng.normal(0, 1000, size)).astype(int).tolist()
        if dist == "skewed": 
            a = float(rng.choice([1.3, 1.5, 2.0]))
            return np.minimum(rng.zipf(a, size), wide).tolist()
        raise ValueError(dist)

    def benchmark_algorithms(self, arr, runs=3):
        times = {}
        for name, func in ALGORITHMS.items():
            timer = timeit.Timer(lambda: func(list(arr)))
            try:
                times[name] = float(np.median(timer.repeat(repeat=runs, number=1)))
            except Exception: 
                times[name] = float("inf")
        return min(times, key=times.get), times

    def build_dataset(self, samples_per_config=10):
        rows = []
        for size in self.sizes:
            for dist in self.distributions:
                for _ in range(samples_per_config):
                    arr = self.generate_array(size, dist)
                    best, _ = self.benchmark_algorithms(arr)
                    rows.append({**FeatureExtractor.extract(arr), "best_algorithm": best})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating dataset (a few minutes)...")
    df = DatasetGenerator().build_dataset(samples_per_config=10)
    df.to_csv("sorting_meta_features.csv", index=False)
    print(df["best_algorithm"].value_counts().to_string())
    print("✅ 'sorting_meta_features.csv' generated!")
