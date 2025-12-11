import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = models.efficientnet_b0(pretrained=True)
model.classifier = torch.nn.Identity()  # Remove final layer 
model = model.to(device)
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def extract_features(path):
    """Extract features from a single image using EfficientNet-B0"""
    try:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img)
        return feat.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def extract_features_from_split(dataset_path, split_name, class_map):
    X, y = [], []
    total_processed = 0
    total_failed = 0

    print(f"\n{'='*60}")
    print(f"Extracting features from {split_name.upper()} split")
    print(f"{'='*60}\n")

    split_path = os.path.join(dataset_path, split_name)

    if not os.path.exists(split_path):
        print(f"Warning: {split_path} not found. Skipping {split_name} split.")
        return np.array([]), np.array([])

    for idx, class_name in enumerate(sorted(class_map.keys()), 1):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            print(f"  {idx}. {class_name}: Skipped (directory not found)")
            continue

        # Get all valid images
        images = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'):
            images.extend(Path(class_path).glob(f"*{ext[1:]}"))
            images.extend(Path(class_path).glob(f"*{ext[1:].upper()}"))

        images = list(set(images))  # use SET remove duplicates

        if not images:
            print(f"  {idx}. {class_name}: No images found")
            continue

        print(f"  {idx}. {class_name}: Processing {len(images)} images...")

        class_processed = 0
        class_failed = 0

        for img_path in images:
            feat = extract_features(str(img_path))
            if feat is not None:
                X.append(feat)
                y.append(class_map[class_name])
                class_processed += 1
            else:
                class_failed += 1

        total_processed += class_processed
        total_failed += class_failed

        print(
            f"     ✓ {class_processed} images processed, {class_failed} failed")

    print(f"\n{split_name.upper()} split summary:")
    print(f"  Total processed: {total_processed}")
    print(f"  Total failed: {total_failed}")
    print(f"  Feature shape: ({len(X)}, {len(X[0]) if X else 0})")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


DATASET_PATH = "data/augmented"
class_map = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5
}

print(f"\n{'='*60}")
print("Feature Extraction from Augmented Dataset")
print(f"{'='*60}")
print(f"Classes: {list(class_map.keys())}")
print(f"Model: EfficientNet-B0")
print(f"Device: {device}")
print(f"{'='*60}")

X_train, y_train = extract_features_from_split(
    DATASET_PATH, 'train', class_map)
X_val, y_val = extract_features_from_split(DATASET_PATH, 'val', class_map)

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

if len(X_train) > 0:
    np.save(os.path.join(processed_dir, "x_features_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_labels_train.npy"), y_train)
    print(f"\n✓ Training features saved:")
    print(f"  data/processed/x_features_train.npy: {X_train.shape}")
    print(f"  data/processed/y_labels_train.npy: {y_train.shape}")
else:
    print("\n✗ No training features extracted!")

if len(X_val) > 0:
    np.save(os.path.join(processed_dir, "x_features_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_labels_val.npy"), y_val)
    print(f"\n✓ Validation features saved:")
    print(f"  data/processed/x_features_val.npy: {X_val.shape}")
    print(f"  data/processed/y_labels_val.npy: {y_val.shape}")
else:
    print("\n✗ No validation features extracted!")

# Combined features
X_combined = np.vstack([X_train, X_val]) if (len(X_train) > 0 and len(
    X_val) > 0) else (X_train if len(X_train) > 0 else X_val)
y_combined = np.concatenate([y_train, y_val]) if (len(y_train) > 0 and len(
    y_val) > 0) else (y_train if len(y_train) > 0 else y_val)

if len(X_combined) > 0:
    np.save(os.path.join(processed_dir, "x_features.npy"), X_combined)
    np.save(os.path.join(processed_dir, "y_labels.npy"), y_combined)
    print(f"\n✓ Combined features saved (backward compatible):")
    print(f"  data/processed/x_features.npy: {X_combined.shape}")
    print(f"  data/processed/y_labels.npy: {y_combined.shape}")

print(f"\n{'='*60}")
print("Feature Extraction Summary")
print(f"{'='*60}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Total samples: {len(X_combined)}")
print(
    f"Features per sample: {X_combined.shape[1] if len(X_combined) > 0 else 0}")
print(f"Number of classes: {len(class_map)}")

# Class distribution
if len(y_combined) > 0:
    print(f"\nClass distribution:")
    for class_name, class_id in sorted(class_map.items(), key=lambda x: x[1]):
        count = np.sum(y_combined == class_id)
        train_count = np.sum(y_train == class_id) if len(y_train) > 0 else 0
        val_count = np.sum(y_val == class_id) if len(y_val) > 0 else 0
        percentage = (count / len(y_combined)) * \
            100 if len(y_combined) > 0 else 0
        print(f"  {class_name:12} | Total: {count:5d} | Train: {train_count:5d} | Val: {val_count:5d} | {percentage:6.2f}%")

print(f"{'='*60}")
print("DONE. Features extracted and saved successfully!")
print(f"{'='*60}\n")
