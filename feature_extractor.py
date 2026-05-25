import numpy as np
import scipy.stats as stats

class FeatureExtractor:
    @staticmethod
    def calculate_entropy(arr):
        """Measures the randomness of the data distribution."""
        if len(arr) == 0: return 0
        _, counts = np.unique(arr, return_counts=True)
        return stats.entropy(counts)

    @staticmethod
    def calculate_runs(arr):
        """Counts monotonic sequences. High runs mean TimSort is likely best."""
        if len(arr) <= 1: return 1
        runs = 1
        is_ascending = arr[1] >= arr[0]
        
        for i in range(1, len(arr) - 1):
            if is_ascending and arr[i+1] < arr[i]:
                runs += 1
                is_ascending = False
            elif not is_ascending and arr[i+1] >= arr[i]:
                runs += 1
                is_ascending = True
        return runs / len(arr) # Normalized run count

    @staticmethod
    def count_inversions(arr):
        """Direct measure of 'sortedness'. Uses modified merge sort."""
        def merge_count(temp_arr, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                inv_count += merge_count(temp_arr, left, mid)
                inv_count += merge_count(temp_arr, mid + 1, right)
                inv_count += merge(temp_arr, left, mid, right)
            return inv_count

        def merge(temp_arr, left, mid, right):
            i = left
            j = mid + 1
            k = left
            inv_count = 0
            while i <= mid and j <= right:
                if temp_arr[i] <= temp_arr[j]:
                    k += 1; i += 1
                else:
                    inv_count += (mid - i + 1)
                    k += 1; j += 1
            return inv_count

        temp = list(arr)
        # Normalize by max possible inversions: n*(n-1)/2
        max_inv = (len(arr) * (len(arr) - 1)) / 2
        if max_inv == 0: return 0
        return merge_count(temp, 0, len(arr) - 1) / max_inv

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
            "key_range_ratio": (np.max(arr_np) - np.min(arr_np)) / len(arr) if len(arr) > 0 else 0
        }