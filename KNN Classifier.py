import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ---------------- Distance Metrics ----------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# ---------------- KNN Classifier ----------------
class KNN:
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [self.distance_metric(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# ---------------- Experiment Function ----------------
def run_knn_experiment(X, y, dataset_name, test_size, distance_metric, metric_name):
    print(f"\n--- {dataset_name} Dataset | Split: {int((1 - test_size) * 100)}-{int(test_size * 100)} | Distance: {metric_name} ---")
    for k in [3, 5, 7]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = KNN(k=k, distance_metric=distance_metric)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        print(f"k={k} | Accuracy: {accuracy:.4f}")

# ---------------------- a) Glass Dataset ----------------------
glass_df = pd.read_csv('glass.csv')  # Make sure glass.csv is in your directory
X_glass = glass_df.drop('Type', axis=1).values
y_glass = glass_df['Type'].values

for test_size in [0.1, 0.3]:
    run_knn_experiment(X_glass, y_glass, "Glass", test_size, euclidean_distance, "Euclidean")
    run_knn_experiment(X_glass, y_glass, "Glass", test_size, manhattan_distance, "Manhattan")

# ---------------------- b) Fruit Dataset ----------------------
fruit_df = pd.read_csv('fruits.csv')  # Make sure fruits.csv is in your directory
X_fruit = fruit_df[['mass', 'width', 'height', 'color_score']].values
y_fruit = fruit_df['fruit_label'].values

for test_size in [0.1, 0.3]:
    run_knn_experiment(X_fruit, y_fruit, "Fruit", test_size, euclidean_distance, "Euclidean")
    run_knn_experiment(X_fruit, y_fruit, "Fruit", test_size, manhattan_distance, "Manhattan")
