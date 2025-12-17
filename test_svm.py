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
    svm = model_data["svm"]
    scaler = model_data["scaler"]

    # Open-set recognition parameters 
    centroids = model_data.get("centroids", None)
    distance_thr = model_data.get("distance_thr", None)
    prob_thr = model_data.get("prob_thr", 0.5)
    margin_thr = model_data.get("margin_thr", 0.1)
    known_classes = model_data.get("known_classes", [0, 1, 2, 3, 4, 5])
    unknown_class = model_data.get("unknown_class", 6)

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

    use_open_set = (centroids is not None and distance_thr is not None)

    if use_open_set:
        for x in X_scaled:
            dists = np.array([np.linalg.norm(x - centroids[c])
                             for c in known_classes])
            min_dist = dists.min()

            if min_dist > distance_thr:
                predictions.append(unknown_class)
                continue

            probs = svm.predict_proba([x])[0]
            sorted_probs = np.sort(probs)[::-1]
            max_prob = sorted_probs[0]
            second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
            margin = max_prob - second_prob
            pred_class = svm.classes_[np.argmax(probs)]

            if max_prob < prob_thr or margin < margin_thr:
                predictions.append(unknown_class)
            else:
                predictions.append(pred_class)
    else:
        predictions = svm.predict(X_scaled).tolist()

    return predictions


if __name__ == "__main__":
    data_path = "data/test"
    model_path = "models/svm_open_set_model.pkl"

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
    for p, pred in zip(img_paths, predictions):
        print(f"{p.name}: {class_map[pred]}")
