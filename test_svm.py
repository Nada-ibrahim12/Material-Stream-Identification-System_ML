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

    use_open_set = (centroids is not None and distance_thr is not None)

    per_feature_results = []
    if use_open_set:
        for x in X_scaled:
            dists = np.array([np.linalg.norm(x - centroids[c]) for c in known_classes])
            min_dist = float(dists.min())

            if min_dist > distance_thr:
                per_feature_results.append({"prediction": int(unknown_class), "avg_distance": min_dist})
                continue

            # try to use predict_proba if available
            try:
                probs = svm.predict_proba([x])[0]
                sorted_probs = np.sort(probs)[::-1]
                max_prob = sorted_probs[0]
                second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
                margin = max_prob - second_prob
                pred_class = int(svm.classes_[np.argmax(probs)])

                if max_prob < prob_thr or margin < margin_thr:
                    per_feature_results.append({"prediction": int(unknown_class), "avg_distance": min_dist})
                else:
                    per_feature_results.append({"prediction": pred_class, "avg_distance": min_dist})
            except Exception:
                pred_class = int(svm.predict([x])[0])
                per_feature_results.append({"prediction": pred_class, "avg_distance": min_dist})
    else:
        # closed-set: predict and no distance
        for x in X_scaled:
            pred = int(svm.predict([x])[0])
            per_feature_results.append({"prediction": pred, "avg_distance": None})

    # Map back to original image list, marking errored images
    results = []
    feat_idx = 0
    for idx, img_path in enumerate(image_paths):
        if idx in valid_indices:
            r = per_feature_results[feat_idx]
            results.append({
                "image_path": str(img_path),
                "prediction": r["prediction"],
                "avg_distance": r["avg_distance"],
                "status": "ok"
            })
            feat_idx += 1
        else:
            results.append({
                "image_path": str(img_path),
                "prediction": int(unknown_class),
                "avg_distance": None,
                "status": "error"
            })

    return results


if __name__ == "__main__":
    data_path = "test"
    model_path = "models/svm_open_set_model.pkl"

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
    for i, r in enumerate(results):
        p = Path(r["image_path"])
        label = class_map.get(r["prediction"], "unknown")
        avgd = f"{r['avg_distance']:.4f}" if r["avg_distance"] is not None else "-"
        status = r.get("status", "ok")
        print(f"{i:<6}{p.name:<40}{label:<12}{r['prediction']:<4}{avgd:>10}{status:>10}")
