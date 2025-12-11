import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import os

processed_dir = "data/processed"

use_split = os.path.exists(
    os.path.join(processed_dir, 'x_features_train.npy')) and os.path.exists(os.path.join(processed_dir, 'x_features_val.npy'))

if use_split:
    print("Loading features from train/val split...")
    X_train = np.load(os.path.join(processed_dir, 'x_features_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_labels_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'x_features_val.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_labels_val.npy'))
else:
    print("Loading combined features (backward compatibility)...")
    X = np.load(os.path.join(processed_dir, 'x_features.npy'))
    y = np.load(os.path.join(processed_dir, 'y_labels.npy'))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDataset Information:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features per sample: {X_train.shape[1]}")

k_values = [3, 5, 7, 9, 11]

print(f"\n{'='*70}")
print(f"KNN Classification Results")
print(f"{'='*70}\n")

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

print(f"\n{'='*70}")
print(f"Best K: {best_k} with Test Accuracy: {best_accuracy*100:.2f}%")
print(f"{'='*70}\n")

# Train final model with best K
final_knn = KNeighborsClassifier(
    n_neighbors=best_k, weights='uniform', metric='cosine')
final_knn.fit(X_train_scaled, y_train)

# ====================================
# UNKNOWN CLASS DETECTION 
# ====================================

print(f"\n{'='*70}")
print(f"UNKNOWN CLASS DETECTION METHODS")
print(f"{'='*70}\n")

# Get class information
class_names = np.unique(y_train)
class_map = {int(label): f"Class_{int(label)}" for label in class_names}
print(f"Known classes: {len(class_names)} | Classes: {list(class_names)}\n")


# Based on average distance to K neighbors
def predict_with_distance(X_scaled, distance_threshold=0.75):
    """
    Reject prediction if average distance to K neighbors is too large
    Larger distance = less confident
    """
    distances, indices = final_knn.kneighbors(X_scaled)
    avg_distances = distances.mean(axis=1)

    predictions = final_knn.predict(X_scaled).copy()

    # Mark as UNKNOWN if distance too large
    predictions[avg_distances > distance_threshold] = -1

    return predictions, avg_distances


print(f"{'='*70}")
print(f"TESTING WITH SYNTHETIC OUT-OF-DISTRIBUTION DATA")
print(f"{'='*70}\n")

# Create OOD samples: random noise + extreme values
np.random.seed(42)
n_ood_samples = 100

# Random noise
ood_noise = np.random.randn(n_ood_samples // 2, X_train_scaled.shape[1]) * 3

# Extreme values (far from training distribution)
ood_extreme = np.random.uniform(-5, 5,
                                (n_ood_samples // 2, X_train_scaled.shape[1]))

ood_samples = np.vstack([ood_noise, ood_extreme])

# Combine test set with OOD samples
X_eval = np.vstack([X_test_scaled, ood_samples])
y_true = np.hstack([y_test, np.full(len(ood_samples), -1)]
                   )  # -1 = true unknown

print(f"Known samples (test set): {len(X_test_scaled)}")
print(f"Unknown samples (OOD): {len(ood_samples)}")
print(f"Total evaluation set: {len(X_eval)}\n")


print(f"\n{'='*70}")
print(f"METHOD 2: DISTANCE THRESHOLD (distance_threshold=0.75)")
print(f"{'='*70}\n")

y_pred_dist, avg_dist = predict_with_distance(X_eval, distance_threshold=0.75)

n_unknown = (y_pred_dist == -1).sum()
n_known = (y_pred_dist != -1).sum()

print(f"Predictions: {n_known} KNOWN | {n_unknown} UNKNOWN")
print(f"Average distance range: {avg_dist.min():.3f} - {avg_dist.max():.3f}")

if n_known > 0:
    known_correct = ((y_pred_dist != -1) & (y_true != -1)).sum()
    known_acc = known_correct / (y_true != -1).sum()
    print(f"Accuracy on KNOWN samples: {known_acc*100:.2f}%")

if n_unknown > 0:
    true_unknown = (y_true == -1).sum()
    detected_unknown = ((y_pred_dist == -1) & (y_true == -1)).sum()
    detection_rate = detected_unknown / true_unknown if true_unknown > 0 else 0
    print(
        f"UNKNOWN detection rate: {detection_rate*100:.2f}% ({detected_unknown}/{true_unknown})")


print(f"\n{'='*70}")

known_acc_2 = (((y_pred_dist != -1) & (y_true != -1)).sum() /
               ((y_true != -1).sum())) if (y_true != -1).sum() > 0 else 0
detect_2 = (((y_pred_dist == -1) & (y_true == -1)).sum() /
            ((y_true == -1).sum())) if (y_true == -1).sum() > 0 else 0
print(
    f"| Distance Threshold      |     {known_acc_2*100:6.2f}%   |      {detect_2*100:6.2f}%       |")

print(f"\n{'='*70}\n")
