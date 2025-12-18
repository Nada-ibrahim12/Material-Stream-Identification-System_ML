# Material Stream Identification System (ML)

A complete pipeline for material stream identification using deep feature extraction (ResNet-50) and classical machine learning classifiers (KNN, SVM). The system supports data augmentation, feature extraction, model training, batch inference, and real‑time webcam classification with open‑set "Unknown" detection.

## Project Structure

- [data/](data/)
  - [dataset/](data/dataset/): Raw images organized by class folders
  - [augmented/](data/augmented/): Auto-generated train/test splits for training
  - [processed/](data/processed/): Extracted feature matrices (`.npy`)
- [src/](src/)
  - [augmentation/AugFinal.py](src/augmentation/AugFinal.py): Augments images and creates train/test splits
  - [feature_extraction/extraction.py](src/feature_extraction/extraction.py): Extracts ResNet-50 features to `.npy`
  - [models/KNN.py](src/models/KNN.py): Trains KNN with open‑set thresholding
  - [models/SVM.py](src/models/SVM.py): Trains SVM with open‑set rejection
  - [deployment/app.py](src/deployment/app.py): Real‑time webcam inference (KNN or SVM)
- [models/](models/): Saved model pickles (`knn_model.pkl`, `svm_model.pkl`)
- [webcam_results/](webcam_results/): Snapshots saved from webcam UI
- [test.py](test.py): Batch inference for a folder of images using a trained model

## Classes and Labels

- Classes: `Glass (0)`, `Paper (1)`, `Cardboard (2)`, `Plastic (3)`, `Metal (4)`, `Trash (5)`, `Unknown (6)`
- Expected dataset folder names: `glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`

Important: The feature extractor script uses a `class_map` with capitalized keys. Ensure your augmented split uses matching folder names or adjust the mapping in [src/feature_extraction/extraction.py](src/feature_extraction/extraction.py) to your folder casing to avoid skipping classes.

## Setup

Requirements: Python 3.9+ on Windows, a working webcam for the real‑time demo (optional). GPU is optional; PyTorch will default to CPU where needed.

Recommended steps:

```powershell
# From the repository root
python -m venv .venv
.venv\Scripts\Activate

# Core dependencies
pip install --upgrade pip
pip install torch torchvision
pip install tensorflow pillow numpy scikit-learn opencv-python joblib
```

Notes:

- TensorFlow is used only for data augmentation in [src/augmentation/AugFinal.py](src/augmentation/AugFinal.py).
- PyTorch + TorchVision are used for feature extraction.
- If you have a CUDA GPU, install the matching PyTorch build from pytorch.org.

## Data Preparation

1. Place raw images into class folders under [data/dataset/](data/dataset/):

```
data/dataset/
	cardboard/
	glass/
	metal/
	paper/
	plastic/
	trash/
```

2. Run augmentation + split (creates `data/augmented/train` and `data/augmented/test`):

```powershell
python src/augmentation/AugFinal.py
```

3. Extract features for train and test splits into [data/processed/](data/processed/):

```powershell
python src/feature_extraction/extraction.py
```

Outputs:

- `x_features_train.npy`, `y_labels_train.npy`
- `x_features_val.npy` (from the `test` split), `y_labels_val.npy`

Tip: If your augmented split uses `val/` instead of `test/`, either rename the folder to `test/` or update the `split_name` in [src/feature_extraction/extraction.py](src/feature_extraction/extraction.py).

## Train Models

KNN (cosine distance with thresholding for Unknown):

```powershell
python src/models/KNN.py
```

SVM (RBF with distance/probability/margin-based open‑set rejection):

```powershell
python src/models/SVM.py
```

Artifacts:

- KNN model: [models/knn_model.pkl](models/knn_model.pkl)
- SVM model: [models/svm_model.pkl](models/svm_model.pkl)

## Inference

Batch inference on a folder of images (KNN model expected):

```powershell
# Example: classify images under .\test\ using KNN
python test.py

# Or explicitly specify paths by editing the variables in test.py
```

Real‑time webcam inference (saves panels and frames under [webcam_results/](webcam_results/)):

```powershell
# Use KNN
python src/deployment/app.py --model knn

# Use SVM
python src/deployment/app.py --model svm
```

Controls in the webcam UI:

- `Q`: Quit
- `S`: Save camera frame
- `C`: Capture both camera frame and control panel

## Implementation Details

- Feature extractor: ResNet‑50 with final FC replaced by `Identity` to output a 2048‑D feature vector ([src/feature_extraction/extraction.py](src/feature_extraction/extraction.py), [src/deployment/app.py](src/deployment/app.py)).
- KNN: `n_neighbors=3`, `metric='cosine'`, with a distance threshold for open‑set `Unknown` ([src/models/KNN.py](src/models/KNN.py)).
- SVM: RBF kernel with probability estimates and class centroids; rejects as `Unknown` based on distance, probability, and margin thresholds ([src/models/SVM.py](src/models/SVM.py)).

## Model Details

### KNN Classifier

- Input features: 2048‑D vectors from ResNet‑50 (FC layer replaced by `Identity`).
- Preprocessing: `StandardScaler` fit on train set only; applied to validation/test.
- Core params: `n_neighbors=3`, `metric='cosine'`, `weights='uniform'` (see [src/models/KNN.py](src/models/KNN.py)).
- Unknown detection: compute average neighbor distance for each sample; if `avg_distance > threshold` classify as `Unknown (6)`.
  - Default `threshold`: `0.7` (change in [src/models/KNN.py](src/models/KNN.py), variable `thr`).
- Saved artifact: [models/knn_model.pkl](models/knn_model.pkl) containing:
  - `knn`: trained `KNeighborsClassifier`
  - `scaler`: `StandardScaler`
  - `threshold`, `k`, `metric`, `weights`
- Inference path:
  - Batch: [test.py](test.py) loads the pickle, scales features, applies KNN, and uses the same distance‑based unknown logic.
  - Webcam: [src/deployment/app.py](src/deployment/app.py) uses `kneighbors()` distances and confidence = `1/(1+avg_distance)`.

Tuning tips:

- Lower `threshold` to be more conservative (more `Unknown`), raise to be more permissive.
- Adjust `k` and optionally try `weights='distance'` for different behavior.

### SVM Classifier (Open‑Set)

- Input features and preprocessing: same as KNN (`StandardScaler`).
- Core params: `SVC(kernel='rbf', C=3, gamma='scale', probability=True, random_state=42)` (see [src/models/SVM.py](src/models/SVM.py)).
- Known/unknown setup:
  - Known classes: `[0..5]` from training labels; `unknown_class = 6` reserved for rejection.
- Centroid computation:
  - For each known class, compute the mean vector (centroid) in scaled feature space.
- Distance threshold (`distance_thr`):
  - Compute distribution of distances of training samples to their class centroids.
  - Set `thr = mean + m * std` with default `m = 3.0`.
- Secondary rejection using probability and margin:
  - `prob_thr = 0.35`: maximum class probability must exceed this.
  - `margin_thr = 0.05`: difference between top‑1 and top‑2 probabilities must exceed this.
- Unknown decision:
  - Reject (`Unknown (6)`) if `min_dist > thr` OR `max_prob < prob_thr` OR `margin < margin_thr`.
- Saved artifact: [models/svm_model.pkl](models/svm_model.pkl) containing:
  - `svm`, `scaler`, `centroids`, `distance_thr`, `prob_thr`, `margin_thr`, `known_classes`, `unknown_class`, `dist_thr_multiplier`.
- Inference path:
  - Webcam: [src/deployment/app.py](src/deployment/app.py) loads centroids and thresholds; computes `min_dist` and uses `predict_proba` when available to apply the rejection rules.

