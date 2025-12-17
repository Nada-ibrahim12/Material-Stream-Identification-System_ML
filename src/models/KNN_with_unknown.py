from joblib import dump
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
from pathlib import Path

processed_dir = "data/processed"

splitting = os.path.exists(
    os.path.join(processed_dir, 'x_features_train.npy')) and os.path.exists(os.path.join(processed_dir, 'x_features_val.npy'))

if splitting:
    print("Loading features from train/val split:")
    X_train = np.load(os.path.join(processed_dir, 'x_features_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_labels_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'x_features_val.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_labels_val.npy'))
else:
    print("Loading combined features:")
    X = np.load(os.path.join(processed_dir, 'x_features.npy'))
    y = np.load(os.path.join(processed_dir, 'y_labels.npy'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [3, 5, 7, 9, 11]

print(f"KNN Classification Results")

best_k = None
best_accuracy = 0

for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k, weights='uniform', metric='cosine')
    knn.fit(X_train_scaled, y_train)

    train_accuracy = knn.score(X_train_scaled, y_train)
    test_accuracy = knn.score(X_test_scaled, y_test)

    print(f"K={k:2d}: Train Accuracy: {train_accuracy*100:6.2f}%  |  Test Accuracy: {test_accuracy*100:6.2f}%  |  Gap: {(train_accuracy - test_accuracy)*100:6.2f}%")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_k = k

print(f"Best K: {best_k} with Test Accuracy: {best_accuracy*100:.2f}%")

final_knn = KNeighborsClassifier(
    n_neighbors=best_k, weights='uniform', metric='cosine')
final_knn.fit(X_train_scaled, y_train)

print("\nModel trained and ready for OOD detection.\n")
class_names = np.unique(y_train)
class_map = {int(label): f"Class_{int(label)}" for label in class_names}
print(f"Known classes: {len(class_names)} | Classes: {list(class_names)}\n")


def unknown_predictions(X_scaled, distance_threshold=0.75):

    distances, indices = final_knn.kneighbors(X_scaled)
    avg_distances = distances.mean(axis=1)

    predictions = final_knn.predict(X_scaled).copy()

    predictions[avg_distances > distance_threshold] = -1

    return predictions, avg_distances

print(f"-------------------Testing with OOD samples---------------------")

np.random.seed(42)
samples = 100

noise = np.random.randn(samples // 2, X_train_scaled.shape[1]) * 3

extreme = np.random.uniform(-5, 5,
                            (samples // 2, X_train_scaled.shape[1]))

ood_samples = np.vstack([noise, extreme])

X_eval = np.vstack([X_test_scaled, ood_samples])
y_true = np.hstack([y_test, np.full(len(ood_samples), -1)]
                   )  

y_pred_dist, avg_dist = unknown_predictions(X_eval, distance_threshold=0.75)

n_unknown = (y_pred_dist == -1).sum()
n_known = (y_pred_dist != -1).sum()

print(f"Predictions: {n_known} KNOWN | {n_unknown} UNKNOWN")

if n_unknown > 0:
    true_unknown = (y_true == -1).sum()
    detected_unknown = ((y_pred_dist == -1) & (y_true == -1)).sum()
    detection_rate = detected_unknown / true_unknown if true_unknown > 0 else 0
    print(
        f"UNKNOWN detection rate: {detection_rate*100:.2f}% ({detected_unknown}/{true_unknown})")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_dict = {
    "knn": final_knn,
    "scaler": scaler,
    "threshold": 0.75,
    "k": best_k,
    "metric": "cosine",
    "weights": "uniform",
    "best_accuracy": best_accuracy,
    "unknown_class": -1
}

model_path = models_dir / f"knn_with_unknown_k{best_k}_model.pkl"
dump(model_dict, model_path)
print(f"\nModel saved to: {model_path}")
