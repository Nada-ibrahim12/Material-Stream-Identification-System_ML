import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

y_pred = final_knn.predict(X_test_scaled)
