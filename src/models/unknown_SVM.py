import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.decomposition import PCA
import os


processed_dir = "data/processed"
unknown_class = 6  
apply_pca = True  
pca_variance = 0.80 

splitting = os.path.exists(os.path.join(processed_dir, 'x_features_train.npy')) and \
            os.path.exists(os.path.join(processed_dir, 'x_features_val.npy'))

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
X_test_scaled = scaler.transform(X_test)

if apply_pca:
    pca = PCA(n_components=pca_variance)
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

one_class_svms = {}
for cls in np.unique(y_train):
    svm_oc = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    svm_oc.fit(X_train_scaled[y_train == cls])
    one_class_svms[cls] = svm_oc

y_pred = []
for i, x in enumerate(X_test_scaled):
    is_known = False
    for cls, oc in one_class_svms.items():
        if oc.predict([x])[0] == 1:  # inside known-class boundary
            # Assign normal SVM prediction for that sample
            y_pred.append(svm.predict([x])[0])
            is_known = True
            break
    if not is_known:

        y_pred.append(unknown_class)

y_pred = np.array(y_pred)

unknown_selector = (y_test == unknown_class)
known_selector = (y_test != unknown_class)

known_accuracy = (y_pred[known_selector] == y_test[known_selector]).mean()
if np.sum(unknown_selector) > 0:
    unknown_detect = (y_pred[unknown_selector] == unknown_class).mean()
else:
    unknown_detect = 0.0

overall_accuracy = (y_pred == y_test).mean()
unknown_count = np.sum(y_pred == unknown_class)


print("\nSVM Results:")
print(f"Known Accuracy: {known_accuracy*100:.2f}%")
print(f"Unknown Detection: {unknown_detect*100:.2f}%")
print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
print(f"Unknown samples detected: {unknown_count}")
