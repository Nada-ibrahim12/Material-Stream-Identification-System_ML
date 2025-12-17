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
    threshold = model_data.get("threshold", 0.7)

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

    for i, img_path in enumerate(image_paths):
        feat = extract_features(str(img_path))
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)

    if not features:
        print("Error: No valid features extracted")
        return []

    X = np.array(features, dtype=np.float32)

    X_scaled = scaler.transform(X)

    distances, _ = knn.kneighbors(X_scaled)
    avg_distances = distances.mean(axis=1)

    per_feature_results = []
    for i in range(len(X_scaled)):
        if avg_distances[i] > threshold:
            per_feature_results.append({
                "prediction": 6,
                "avg_distance": float(avg_distances[i])
            })
        else:
            pred = int(knn.predict(X_scaled[i].reshape(1, -1))[0])
            per_feature_results.append({
                "prediction": pred,
                "avg_distance": float(avg_distances[i])
            })

    feat_idx = 0
    results = []
    for i in range(len(image_paths)):
        img_path = image_paths[i]

        if i in valid_indices:
            feature_result = per_feature_results[feat_idx]
            result_item = {
                "image_path": str(img_path),
                "prediction": feature_result["prediction"],
                "avg_distance": feature_result["avg_distance"],
                "status": "ok"
            }
            feat_idx += 1
        else:
            result_item = {
                "image_path": str(img_path),
                "prediction": 6,
                "avg_distance": None,
                "status": "error"
            }
        results.append(result_item)

    return results

if __name__ == "__main__":
    data_path = "test/"
    model_path = "models/knn_k3_model.pkl"

    results = predict(data_path, model_path)
    class_map = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash",
        6: "unknown"
    }

    print("\nPredictions:")
    print(f"{'Index':<6}{'Filename':<40}{'Label':<12}{'ID':<4}{'AvgDist':>10}{'Status':>10}")
    for i in range(len(results)):
        r = results[i]

        path_obj = Path(r["image_path"])
        image_name = path_obj.name

        prediction = r["prediction"]
        if prediction in class_map:
            label = class_map[prediction]
        else:
            label = "unknown"

        if r["avg_distance"] is not None:
            avg_distance = f"{r['avg_distance']:.4f}"
        else:
            avg_distance = "-"

        if "status" in r:
            status = r["status"]
        else:
            status = "ok"

        print(f"{i:<6}{image_name:<40}{label:<12}{prediction:<4}{avg_distance:>10}{status:>10}")

