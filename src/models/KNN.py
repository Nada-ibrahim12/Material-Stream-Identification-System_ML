import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib
from pathlib import Path

processed_dir = Path.cwd() / "data" / "processed"

X_train = np.load(processed_dir / 'x_features_train.npy')
y_train = np.load(processed_dir / 'y_labels_train.npy')
X_test = np.load(processed_dir / 'x_features_val.npy')
y_test = np.load(processed_dir / 'y_labels_val.npy')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [3, 5, 7, 9, 11]
thr = 0.58

# Create models directory if it doesn't exist
models_dir = Path.cwd() / "models"
models_dir.mkdir(exist_ok=True)

print("========== KNN RESULTS ==========")

for k in k_values:
    print(f"\n🔹 K = {k}")

    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',
        metric='cosine'
    )
    knn.fit(X_train_scaled, y_train)

    train_acc = knn.score(X_train_scaled, y_train)
    test_acc = knn.score(X_test_scaled, y_test)

    print(f"Before Unknown → Train {train_acc*100:.2f}% | "
          f"Test {test_acc*100:.2f}% | "
          f"Gap {(train_acc-test_acc)*100:.2f}%")

    distances, _ = knn.kneighbors(X_test_scaled)
    avg_dist = distances.mean(axis=1)

    y_pred = []
    for i in range(len(X_test_scaled)):
        if avg_dist[i] > thr:
            y_pred.append(6)
        else:
            y_pred.append(
                knn.predict(X_test_scaled[i].reshape(1, -1))[0]
            )
    y_pred = np.array(y_pred)

    known_mask = (y_test != 6)
    unknown_mask = (y_test == 6)

    known_acc = (y_pred[known_mask] == y_test[known_mask]).mean()
    unknown_reject = (
        (y_pred[unknown_mask] == 6).mean()
        if np.sum(unknown_mask) > 0 else 0.0
    )
    overall_acc = (y_pred == y_test).mean()

    print(f"After Unknown → Known {known_acc*100:.2f}% | "
          f"Unknown Reject {unknown_reject*100:.2f}% | "
          f"Overall {overall_acc*100:.2f}%")

    model_dict = {
        "knn": knn,
        "scaler": scaler,
        "threshold": thr,
        "k": k,
        "metric": "cosine",
        "weights": "uniform"
    }

    model_path = models_dir / f"knn_k{k}_model.pkl"
    joblib.dump(model_dict, model_path)
    print(f"Model saved to: {model_path}")
