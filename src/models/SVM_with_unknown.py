import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

processed_dir = "data/processed"

unknown_class = 6
dist_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0] # Distance threshold multiplier (mean + m * std)

# Probability and margin thresholds for secondary rejection
prob_thr = 0.35
margin_thr = 0.05

splitting = os.path.exists(os.path.join(processed_dir, 'x_features_train.npy')) and os.path.exists(
    os.path.join(processed_dir, 'x_features_val.npy'))

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

known_classes = np.unique(y_train[y_train != unknown_class])

svm = SVC(kernel='rbf', C=3, gamma='scale', probability=True, random_state=42)
known_filter = np.isin(y_train, known_classes)
svm.fit(X_train_scaled[known_filter], y_train[known_filter])

centroids = {}
for c in known_classes:
    centroids[c] = X_train_scaled[y_train == c].mean(axis=0)

distances = []
for c in known_classes:
    c_features = X_train_scaled[y_train == c]
    centroid = centroids[c]
    dist = np.linalg.norm(c_features - centroid, axis=1)
    distances.extend(dist)

distances = np.array(distances)
mean_dist = distances.mean()
std_dist = distances.std()

def evaluate_threshold(multiplier):
    thr = mean_dist + multiplier * std_dist
    preds = []
    covered_flags = []
    for x in X_test_scaled:
        final_distances = []
        for c in known_classes:
            dist = np.linalg.norm(x - centroids[c])
            final_distances.append(dist)

        min_dist = min(final_distances)
        
        if min_dist > thr:
            preds.append(unknown_class)
            covered_flags.append(False)
            continue

        # Secondary rejection using probability and margin
        probs = svm.predict_proba([x])[0] # get predicted probabilities for sample x
        sorted_probs = sorted(probs, reverse=True) # descending order
        max_prob = sorted_probs[0]

        # Second highest probability or 0 if only one class exists
        if len(sorted_probs) > 1:
            second_prob = sorted_probs[1]
        else:
            second_prob = 0.0
        margin = max_prob - second_prob # margin between 2 probs

        if max_prob < prob_thr or margin < margin_thr:
            preds.append(unknown_class)
            covered_flags.append(False)
        else:
            preds.append(svm.predict([x])[0])
            covered_flags.append(True)

    preds = np.array(preds)
    covered_flags = np.array(covered_flags)

    known_mask = (y_test != unknown_class)
    unknown_mask = (y_test == unknown_class)

    known_acc = (preds[known_mask] == y_test[known_mask]).mean()
    if unknown_mask.sum() > 0:
        unk_detect = (preds[unknown_mask] == unknown_class).mean()
    else:
        unk_detect = 0.0

    overall_acc = (preds == y_test).mean()
    coverage = covered_flags.mean()

    covered_acc = []
    if covered_flags.sum() > 0:
        covered_acc = (preds[covered_flags] == y_test[covered_flags]).mean()
    else: 
        covered_acc = 0.0
        
    unk_count = np.sum(preds == unknown_class)

    return {
        "multiplier": multiplier,
        "thr": thr,
        "known_acc": known_acc,
        "unk_detect": unk_detect,
        "overall_acc": overall_acc,
        "coverage": coverage,
        "covered_acc": covered_acc,
        "unk_count": unk_count,
    }

results = []
for m in dist_multipliers:
    result = evaluate_threshold(m)
    results.append(result)

print("\nSVM Unknown Detection:")
print("mult | thr | known% | unk_detect% | overall% | coverage% | unk_count")
for r in results:
    print(f"{r['multiplier']:>4.1f} | {r['thr']:.2f} | "
        f"{r['known_acc']*100:6.2f} | {r['unk_detect']*100:11.2f} | "
        f"{r['overall_acc']*100:8.2f} | {r['coverage']*100:9.2f} | {r['unk_count']}")

results_sorted = sorted(results, 
                        key=lambda r: (r['unk_detect'], r['overall_acc']), 
                        reverse=True)
best = results_sorted[0]

print("\nSelected threshold:")
print(f"multiplier={best['multiplier']}, thr={best['thr']:.2f}, "
    f"known={best['known_acc']*100:.2f}%, unk_detect={best['unk_detect']*100:.2f}%, "
    f"overall={best['overall_acc']*100:.2f}%, coverage={best['coverage']*100:.2f}%")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_dict = {
    "svm": svm,
    "scaler": scaler,
    "centroids": centroids,
    "distance_thr": best["thr"],
    "prob_thr": prob_thr,
    "margin_thr": margin_thr,
    "known_classes": known_classes.tolist(),
    "unknown_class": unknown_class,
    "dist_thr_multiplier": best["multiplier"],
}

model_path = models_dir / "svm_open_set_model.pkl"
joblib.dump(model_dict, model_path)
print(f"\nModel saved to: {model_path}")
