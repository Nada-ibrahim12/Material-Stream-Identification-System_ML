import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

processed_dir = "data/processed"

splitting = os.path.exists(os.path.join(processed_dir, 'x_features_train.npy')) and os.path.exists(os.path.join(processed_dir, 'x_features_val.npy'))

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

c_values = [0.3, 0.5, 0.7,0.9, 1,3, 5,10]

best_c = None
best_accuracy = 0

for c in c_values:
    svm = SVC(kernel='rbf', C=c, gamma='scale',class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)

    train_accuracy = svm.score(X_train_scaled, y_train)
    test_accuracy = svm.score(X_val_scaled, y_test)

    gap = train_accuracy - test_accuracy

    print(f"C={c:4}: Train Accuracy: {train_accuracy*100:6.2f}%  |  Val Accuracy: {test_accuracy*100:6.2f}%  |  Gap: {(gap)*100:6.2f}%")

    if test_accuracy > best_accuracy :
        best_accuracy = test_accuracy
        best_c = c

print(f"Best C: {best_c} with Val Accuracy: {best_accuracy*100:.2f}%")