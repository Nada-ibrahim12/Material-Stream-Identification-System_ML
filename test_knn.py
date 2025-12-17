import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import os
from pathlib import Path


def predict(dataFilePath, bestModelPath):

    model_data = joblib.load(bestModelPath)
    knn = model_data["knn"]
    scaler = model_data["scaler"]
    threshold = model_data.get("threshold", 0.58)

    device = torch.device('cpu')

    resnet_model = models.resnet50(pretrained=True)
    resnet_model.fc = torch.nn.Identity() 
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    def extract_features(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet_model(img)
            return feat.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None


    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(Path(dataFilePath).glob(ext))

    image_paths = sorted(list(set(image_paths)))

    print(f"Found {len(image_paths)} images in {dataFilePath}:")
    for p in image_paths:
        try:
            print(f" - {p.name}")
        except Exception:
            print(f" - {str(p)}")

    if not image_paths:
        print(f"Warning: No images found in {dataFilePath}")
        return []

    # feature extraction 
    features = []
    valid_indices = []

    for idx, img_path in enumerate(image_paths):
        feat = extract_features(str(img_path))
        if feat is not None:
            features.append(feat)
            valid_indices.append(idx)

    if not features:
        print("Error: No valid features extracted")
        return []

    X = np.array(features, dtype=np.float32)

    X_scaled = scaler.transform(X)

    predictions = []

    distances, _ = knn.kneighbors(X_scaled)
    avg_distances = distances.mean(axis=1)

    for i in range(len(X_scaled)):
        # Check if sample is too far from training data
        if avg_distances[i] > threshold:
            predictions.append(6) 
        else:
            pred = knn.predict(X_scaled[i].reshape(1, -1))[0]
            predictions.append(pred)

    return predictions


if __name__ == "__main__":
    data_path = "test/"
    model_path = "models/knn_with_unknown_k3_model.pkl"

    # Make predictions
    predictions = predict(data_path, model_path)

    class_map = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash",
        6: "unknown"
    }

    img_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        img_paths.extend(Path(data_path).glob(ext))
    img_paths = sorted(list(set(img_paths)))

    print("\nPredictions:")
    for i, (p, pred) in enumerate(zip(img_paths, predictions)):
        print(f"{p.name}: {class_map[pred]}")
