import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

processed_dir = "data/processed"

unknown_class = 6
dist_thr = 2.2  

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
X_test_scaled   = scaler.transform(X_test)

known_classes = np.unique(y_train[y_train != unknown_class])

svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm.fit(  
    X_train_scaled[np.isin(y_train, known_classes)],
    y_train[np.isin(y_train, known_classes)]
)

centroids = {}
for c in known_classes:
    centroids[c] = X_train_scaled[y_train == c].mean(axis=0)

distances = []
for c in known_classes:
    c_features = X_train_scaled[y_train == c]
    centroid = centroids[c]
    dist = np.linalg.norm(c_features - centroid, axis=1)
    distances.extend(dist)

final_thr = np.mean(distances) + dist_thr * np.std(distances)
print("Auto distance threshold:", final_thr)

y_pred = []
covered = []

for x in X_test_scaled:
    final_distances = []
    for c in known_classes:
        dist = np.linalg.norm(x - centroids[c])
        final_distances.append(dist)
    final_distances = np.array(final_distances)
    min_dist = final_distances.min()

    if min_dist > final_thr:
        y_pred.append(unknown_class)
        covered.append(False)
    else:
        y_pred.append(svm.predict([x])[0])
        covered.append(True)
    
y_pred = np.array(y_pred)
covered = np.array(covered)

known_selector   = (y_test != unknown_class)
unknown_selector = (y_test == unknown_class)

known_accuracy = (y_pred[known_selector] == y_test[known_selector]).mean()

unknown_detect = []
if np.sum(unknown_selector) > 0:
    unknown_detect = (y_pred[unknown_selector] == unknown_class).mean()
else:
    unknown_detect = 0.0
unknown_count = np.sum(y_pred == unknown_class)

overall_accuracy = (y_pred == y_test).mean()
coverage = covered.mean()

if covered.sum() > 0:
    covered_accuracy = (y_pred[covered] == y_test[covered]).mean()
else:
    covered_accuracy = 0.0

print("\nSVM Unknown Detection (Auto Threshold):")
print(f"Known Accuracy:        {known_accuracy*100:.2f}%")
print(f"Unknown Detection:     {unknown_detect*100:.2f}%")
print(f"Coverage:              {coverage*100:.2f}%")
print(f"Accuracy on Covered:   {covered_accuracy*100:.2f}%")
print(f"Overall Accuracy:      {overall_accuracy*100:.2f}%")
print(f"Unknown samples detected: {unknown_count}")