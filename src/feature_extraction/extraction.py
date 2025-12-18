import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path

device = torch.device('cpu')

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

def extract_features(path):
    try:
        img = Image.open(path).convert('RGB')
        img = transform_image(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img)
        return feat.cpu().numpy().flatten()
    except Exception as e:
        return None

def extract_features_from_split(dataset_path, split_name, class_map):
    X, y = [], []
    total_processed = 0
    total_failed = 0

    print(f"\nExtracting features from {split_name.upper()} split")

    split_path = os.path.join(dataset_path, split_name)

    if not os.path.exists(split_path):
        print(f"Warning: {split_path} not found. Skipping {split_name} split.")
        return np.array([]), np.array([])

    for i, class_name in enumerate(sorted(class_map.keys()), 1):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            print(f"{i}. {class_name}: Skipped (directory not found)")
            continue

        extensions = ('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')  
        images = []
        for ext in extensions:
            images.extend(Path(class_path).glob(f"*.{ext}"))
            images.extend(Path(class_path).glob(f"*.{ext.upper()}"))

        images = list(set(images))

        if not images:
            print(f"{i}. {class_name}: No images found")
            continue

        print(f"{i}. {class_name}: Processing {len(images)} images...")

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
            f"{class_processed} images processed, {class_failed} failed")

    print(f"\n{split_name.upper()} split summary:")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print(f"Feature shape: ({len(X)}, {len(X[0]) if X else 0})")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

DATASET_PATH = "data/augmented"
class_map = {
    "Glass": 0,
    "Paper": 1,
    "Cardboard": 2,
    "Plastic": 3,
    "Metal": 4,
    "Trash": 5,
    # "Unknown" : 6
}

X_train, y_train = extract_features_from_split(DATASET_PATH, 'train', class_map)
X_val, y_val = extract_features_from_split(DATASET_PATH, 'test', class_map)

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

if len(X_train) > 0:
    np.save(os.path.join(processed_dir, "x_features_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_labels_train.npy"), y_train)
    print(f"\nTraining features saved:")
    print(f"data/processed/x_features_train.npy: {X_train.shape}")
    print(f"data/processed/y_labels_train.npy: {y_train.shape}")
else:
    print("\nNo training features extracted")

if len(X_val) > 0:
    np.save(os.path.join(processed_dir, "x_features_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_labels_val.npy"), y_val)
    print(f"\nValidation features saved:")
    print(f"data/processed/x_features_val.npy: {X_val.shape}")
    print(f"data/processed/y_labels_val.npy: {y_val.shape}")
else:
    print("\nNo validation features extracted")

if len(X_train) > 0 and len(X_val) > 0:
    y_combined = np.concatenate([y_train, y_val])

print("\nFeature Extraction Summary")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Total samples: {len(X_train)+len(X_val)}")


features_len = X_train.shape[1]
print(f"Features per sample: {features_len}")

print(f"Number of classes: {len(class_map)}")

if len(y_combined) > 0:
    print(f"\nClass distribution:")
    class_items = list(class_map.items())
    class_items.sort(key=lambda item: item[1])

    for class_name, class_id in class_items:
        total_count = np.sum(y_combined == class_id)
        if len(y_train) > 0:
            train_count = np.sum(y_train == class_id)
        else:
            train_count = 0

        if len(y_val) > 0:
            val_count = np.sum(y_val == class_id)
        else:
            val_count = 0

        if len(y_combined) > 0:
            percentage = (total_count / len(y_combined)) * 100
        else:
            percentage = 0.0

        print(f"  {class_name:12} | Total: {total_count:5d} | "f"Train: {train_count:5d} | Val: {val_count:5d} | {percentage:6.2f}%")
print("Features extracted and saved successfully!!!!")
