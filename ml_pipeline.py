import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


class MLPipeline:
    def __init__(self, data_path="sorting_meta_features.csv"):
        self.df = pd.read_csv(data_path)
        self.label_encoder = LabelEncoder()

        self.X = self.df.drop(columns=["best_algorithm"])
        self.y = self.label_encoder.fit_transform(self.df["best_algorithm"])

        self.feature_names = self.X.columns.tolist()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.baseline_model = DecisionTreeClassifier(random_state=42)

    def evaluate_cross_validation(self, splits=5):
        """Proves model stability using Stratified K-Fold CV."""
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        rf_scores, dt_scores = [], []

        for train_idx, test_idx in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            self.model.fit(X_train, y_train)
            rf_scores.append(accuracy_score(y_test, self.model.predict(X_test)))

            self.baseline_model.fit(X_train, y_train)
            dt_scores.append(accuracy_score(y_test, self.baseline_model.predict(X_test)))

        print(f"\n--- {splits}-Fold Cross Validation Results ---")
        print(f"Decision Tree (Baseline) Mean Accuracy: {np.mean(dt_scores):.4f}")
        print(f"Random Forest (Proposed) Mean Accuracy: {np.mean(rf_scores):.4f}")

    def train_final_model(self):
        """Trains on a train split and prints a detailed classification report."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print("\n--- Final Model Classification Report ---")
        labels = np.unique(np.concatenate([y_test, y_pred]))
        target_names = self.label_encoder.inverse_transform(labels)
        print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))
        return self.model, self.label_encoder

    def get_feature_importance(self):
        """Extracts feature importance for the research paper analysis."""
        importances = self.model.feature_importances_
        d = dict(zip(self.feature_names, importances))
        return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

    def train_and_save_model(self):
        print("\nTraining final model on full dataset...")
        final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        final_rf.fit(self.X, self.y)

        folder = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(folder, "rf_sorting_model.pkl")
        encoder_path = os.path.join(folder, "label_encoder.pkl")

        joblib.dump(final_rf, model_path)
        joblib.dump(self.label_encoder, encoder_path)

        print(f"✅ Final model saved to: {model_path}")
        print(f"✅ Encoder saved to: {encoder_path}")


if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.evaluate_cross_validation()
    pipeline.train_final_model()
    print("\nFeature importance:")
    for name, score in pipeline.get_feature_importance().items():
        print(f"  {name:<18}{score:.4f}")
    pipeline.train_and_save_model()
