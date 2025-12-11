import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os

from sklearn.svm import SVC

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
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=0.3, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)

train_acc = svm.score(X_train_scaled, y_train)
test_acc = svm.score(X_val_scaled, y_test)

print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Val Accuracy: {test_acc*100:.2f}%")

y_pred = svm.predict(X_val_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))