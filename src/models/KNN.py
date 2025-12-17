import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
thr = 0.7

models_dir = Path.cwd() / "models"
models_dir.mkdir(exist_ok=True)

print("KNN RESULTS: ")

for k in k_values:
    print(f"\n K = {k}")

    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform',metric='cosine')
    knn.fit(X_train_scaled, y_train)

    train_accuracy = knn.score(X_train_scaled, y_train)
    test_accuracy = knn.score(X_test_scaled, y_test)
    gap = train_accuracy - test_accuracy

    print(f"Before Unknown : Train {train_accuracy*100:.2f}% | "f"Test {test_accuracy*100:.2f}% | "f"Gap {(gap)*100:.2f}%")

    distances, _ = knn.kneighbors(X_test_scaled)
    avg_dist = distances.mean(axis=1)

    y_pred = []
    test_val_count = len(X_test_scaled)
    for i in range(test_val_count):
        if avg_dist[i] > thr:
            y_pred.append(6)
        else:
            y_pred.append(knn.predict(X_test_scaled[i].reshape(1, -1))[0])

    y_pred = np.array(y_pred)

    known_mask = y_test != 6
    unknown_mask = y_test == 6
    known_accuracy = (y_pred[known_mask] == y_test[known_mask]).mean()

    unknown_detect = []
    if np.sum(unknown_mask) > 0:
        unknown_detect = (y_pred[unknown_mask] == 6).mean()
    else:
        unknown_detect = 0.0

    overall_acc = (y_pred == y_test).mean()

    print(f"After Unknown : Known {known_accuracy*100:.2f}% | "f"Unknown Reject {unknown_detect*100:.2f}% | "f"Overall {overall_acc*100:.2f}%")

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
