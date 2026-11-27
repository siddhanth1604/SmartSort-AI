import numpy as np
import scipy.stats as stats


class FeatureExtractor:
    @staticmethod
    def calculate_entropy(arr):
        """Measures the randomness of the data distribution."""
        if len(arr) == 0:
            return 0
        _, counts = np.unique(arr, return_counts=True)
        return stats.entropy(counts)

    @staticmethod
    def calculate_runs(arr):
        """Counts monotonic sequences. High runs mean TimSort is likely best."""
        if len(arr) <= 1:
            return 1
        runs = 1
        is_ascending = arr[1] >= arr[0]

        for i in range(1, len(arr) - 1):
            if is_ascending and arr[i + 1] < arr[i]:
                runs += 1
                is_ascending = False
            elif not is_ascending and arr[i + 1] >= arr[i]:
                runs += 1
                is_ascending = True
        return runs / len(arr)  # Normalized run count

    @staticmethod
    def count_inversions(arr):
        """Direct measure of 'sortedness' using a correct merge-sort count."""

        def sort_count(a):
            n = len(a)
            if n <= 1:
                return a, 0
            mid = n // 2
            left, left_inv = sort_count(a[:mid])
            right, right_inv = sort_count(a[mid:])
            merged, i, j = [], 0, 0
            inv = left_inv + right_inv
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    inv += len(left) - i
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, inv

        n = len(arr)
        max_inv = n * (n - 1) / 2
        if max_inv == 0:
            return 0
        return sort_count(list(arr))[1] / max_inv

    @classmethod
    def extract(cls, arr):
        """Compiles all meta-features into a dictionary for the ML model."""
        arr_np = np.array(arr)

        return {
            "size": len(arr),
            "entropy": cls.calculate_entropy(arr),
            "normalized_runs": cls.calculate_runs(arr),
            "inversion_ratio": cls.count_inversions(arr),
            "variance": np.var(arr_np) if len(arr) > 0 else 0,
            "duplicate_ratio": 1.0 - (len(np.unique(arr_np)) / len(arr)) if len(arr) > 0 else 0,
            "key_range_ratio": (np.max(arr_np) - np.min(arr_np)) / len(arr) if len(arr) > 0 else 0,
        }
