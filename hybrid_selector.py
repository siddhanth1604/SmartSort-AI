import copy
import pandas as pd
import joblib
from feature_extractor import FeatureExtractor
from sorters import ALGORITHMS


class HybridSelector:
    def __init__(self, model_path="rf_sorting_model.pkl", encoder_path="label_encoder.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
        except FileNotFoundError:
            print("Model files not found. Please train the ML pipeline first.")
            self.model = None

    def sort(self, arr):
        """Always returns (sorted_array, description)."""
        arr = list(arr)
        n = len(arr)

        if n <= 1:
            return arr, "Trivial (size <= 1)"
        if n < 32:
            ALGORITHMS["InsertionSort"](arr)
            return arr, "InsertionSort (FSM Rule: Size < 32)"

        if self.model is None:
            raise RuntimeError("Model not initialized.")

        feats = FeatureExtractor.extract(arr)
        cols = list(getattr(self.model, "feature_names_in_", feats.keys()))
        X = pd.DataFrame([feats])[cols]

        algo_name = self.encoder.inverse_transform(self.model.predict(X))[0]

        working = copy.deepcopy(arr)
        result = ALGORITHMS[algo_name](working)
        sorted_arr = result if result is not None else working
        return sorted_arr, f"{algo_name} (ML Selected)"


if __name__ == "__main__":
    from DatasetGenerator import DatasetGenerator
    from ml_pipeline import MLPipeline

    print("1. Generating dataset...")
    df = DatasetGenerator(sizes=[100, 500, 1000, 2000]).build_dataset(samples_per_config=15)
    df.to_csv("sorting_meta_features.csv", index=False)

    print("\n2. Training ML model...")
    MLPipeline().train_and_save_model()

    print("\n3. Testing Hybrid Selector...")
    selector = HybridSelector()
    tests = [
        ("Tiny Array", [5, 2, 9, 1, 5, 6]),
        ("Reverse Sorted", list(range(1000, 0, -1))),
        ("Nearly Sorted", list(range(990)) + [0] * 10),
    ]
    for name, data in tests:
        _, chosen = selector.sort(data)
        print(f"Input Type: {name} -> Handled by: {chosen}")
