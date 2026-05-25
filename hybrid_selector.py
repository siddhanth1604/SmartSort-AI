import numpy as np
import pandas as pd
import joblib
from feature_extractor import FeatureExtractor
from sorters import ALGORITHMS
from DatasetGenerator import DatasetGenerator
from ml_pipeline import MLPipeline

class HybridSelector:
    def __init__(self, model_path="rf_sorting_model.pkl", encoder_path="label_encoder.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
        except FileNotFoundError:
            print("Model files not found. Please train the ML pipeline first.")
            self.model = None

    def sort(self, arr):
        n = len(arr)
        
        if n <= 1:
            return arr
        if n < 32:
            # Insertion sort is universally fastest for tiny arrays
            ALGORITHMS["InsertionSort"](arr)
            return arr, "InsertionSort (FSM Rule: Size < 32)"
            
        if self.model is None:
            raise RuntimeError("Model not initialized.")
            
        features = FeatureExtractor.extract_all_features(arr)
        feature_df = pd.DataFrame([features])
        
        predicted_class = self.model.predict(feature_df)[0]
        algo_name = self.encoder.inverse_transform([predicted_class])[0]
        
        sorted_arr = ALGORITHMS[algo_name](arr)
        
        if sorted_arr is not None:
            arr = sorted_arr
            
        return arr, f"{algo_name} (ML Selected)"

if __name__ == "__main__":
    print("1. Generating Dataset (This should take about 1 to 2 minutes max)...")
    
    generator = DatasetGenerator(sizes=[100, 500, 1000, 2000]) 
    df = generator.build_dataset(samples_per_config=15) 
    
    df.to_csv("sorting_meta_features.csv", index=False)
    print("Dataset saved to sorting_meta_features.csv")
    
    print("\n2. Training ML Model...")
    pipeline = MLPipeline()
    pipeline.train_and_save_model()
    
    print("\n3. Testing Hybrid Selector...")
    selector = HybridSelector()
    
    test_arrays = [
        ("Tiny Array", [5, 2, 9, 1, 5, 6]),
        ("Reverse Sorted", list(range(1000, 0, -1))),
        ("Nearly Sorted", list(range(1000))[:990] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ]
    
    for name, data in test_arrays:
        _, chosen_algo = selector.sort(data)
        print(f"Input Type: {name} -> Handled by: {chosen_algo}")