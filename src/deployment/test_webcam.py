# python src/deployment/test_webcam.py --model svm
# python src/deployment/test_webcam.py --model knn
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import cv2
import os
from pathlib import Path
import datetime
import joblib
import argparse

processed_dir = "data/processed"
KNN_MODEL_PATH = "models/knn_k3_model.pkl"
SVM_MODEL_PATH = "models/svm_open_set_model.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_THRESHOLD = 0.7

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
num_classes = len(class_names)

print(f"Using device: {device}")
print("Loading feature extractor...")

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classifier_type = None  # 'knn' or 'svm'
KNOWN_DISTANCE_THRESHOLD = UNKNOWN_THRESHOLD
svm = None
knn = None
scaler = None
svm_centroids = None
svm_distance_thr = None
svm_prob_thr = 0.5
svm_margin_thr = 0.1
svm_known_classes = [0, 1, 2, 3, 4, 5]
svm_unknown_class = 6


def extract_features_from_image(image_array):
    try:
        # Convert BGR to RGB
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def predict_with_confidence(features):
    try:
        if classifier_type == 'knn':
            distances, _ = knn.kneighbors(features.reshape(1, -1))
            distances = distances[0]
            avg_distance = np.mean(distances)
            confidence = 1 / (1 + avg_distance)
            prediction = knn.predict(features.reshape(1, -1))[0]
            return int(prediction), confidence, avg_distance
        elif classifier_type == 'svm':
            x = features.reshape(1, -1)
            if svm_centroids is not None and svm_distance_thr is not None:
                dists = np.array([np.linalg.norm(x - svm_centroids[c])
                                 for c in svm_known_classes]).flatten()
                min_dist = float(np.min(dists))
                # Probability-based confidence
                if hasattr(svm, 'predict_proba'):
                    probs = svm.predict_proba(x)[0]
                    max_prob = float(np.max(probs))
                    sorted_probs = np.sort(probs)[::-1]
                    second_prob = float(sorted_probs[1]) if len(
                        sorted_probs) > 1 else 0.0
                    margin = max_prob - second_prob
                    pred_class = int(svm.classes_[np.argmax(probs)])
                    # Unknown decision
                    is_unknown = (min_dist > svm_distance_thr) or (
                        max_prob < svm_prob_thr) or (margin < svm_margin_thr)
                    if is_unknown:
                        return svm_unknown_class, max_prob, min_dist
                    return pred_class, max_prob, min_dist
                else:
                    pred_class = int(svm.predict(x)[0])
                    is_unknown = (min_dist > svm_distance_thr)
                    confidence = 1.0 - (min_dist / (1.0 + min_dist))
                    if is_unknown:
                        return svm_unknown_class, confidence, min_dist
                    return pred_class, confidence, min_dist
            else:
                if hasattr(svm, 'predict_proba'):
                    probs = svm.predict_proba(x)[0]
                    max_prob = float(np.max(probs))
                    pred_class = int(svm.classes_[np.argmax(probs)])
                    return pred_class, max_prob, 0.0
                else:
                    pred_class = int(svm.predict(x)[0])
                    return pred_class, 1.0, 0.0
        else:
            raise RuntimeError("Classifier type not initialized")
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1, 0.0, 0.0


def test_webcam():
    print("\n" + "="*70)
    print("WEBCAM TEST")
    print("="*70)
    print("Press 'q' to quit | 's' to save | 'c' to capture full result")
    print("="*70 + "\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not available!")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    output_folder = "webcam_results"
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    last_prediction = "WAITING"
    last_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        features = extract_features_from_image(frame)

        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))[0]
            pred_class, confidence, distance = predict_with_confidence(
                features_scaled)

            # Unknown logic per classifier type
            if classifier_type == 'knn':
                is_unknown = (distance > KNOWN_DISTANCE_THRESHOLD) or (
                    confidence < CONFIDENCE_THRESHOLD)
                score_label = "Confidence"
            else:  # svm
                # For SVM we treat returned class equal to unknown_class as unknown
                is_unknown = (pred_class == svm_unknown_class)
                score_label = "Prob" if hasattr(
                    svm, 'predict_proba') else "Confidence"

            if is_unknown or pred_class == -1:
                last_prediction = "UNKNOWN"
                last_confidence = distance if classifier_type == 'knn' else (
                    1.0 - confidence)
                color_bg = (0, 0, 255)
                score_text = f"Distance: {distance:.4f}" if classifier_type == 'knn' else f"{score_label}: {confidence:.4f}"
                status = "⚠ LOW CONFIDENCE"
            else:
                last_prediction = class_names[pred_class] if pred_class < len(
                    class_names) else "UNKNOWN"
                last_confidence = confidence
                color_bg = (0, 255, 0)
                score_text = f"{score_label}: {confidence:.4f}"
                status = "✓ CONFIDENT"
        else:
            last_prediction = "ERROR"
            last_confidence = 0.0
            color_bg = (0, 0, 255)
            score_text = "Feature extraction failed"
            status = "❌ ERROR"

        camera_window = frame.copy()

        cv2.rectangle(camera_window, (10, 10), (1270, 90), (0, 0, 0), -1)
        cv2.putText(camera_window, f"Frame: {frame_count}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        panel_height = 700
        panel_width = 500
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 240

        # Header
        cv2.rectangle(panel, (0, 0), (panel_width, 100), (50, 50, 50), -1)
        cv2.putText(panel, "CLASSIFIER", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        badge_text = f"Model: {classifier_type.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            badge_text, font, scale, thickness)
        badge_x2 = panel_width - 20
        badge_x1 = max(20, badge_x2 - (text_w + 20))
        badge_y1 = 20
        badge_y2 = badge_y1 + text_h + 10
        badge_color = (0, 180, 0) if classifier_type == 'knn' else (
            0, 120, 255)
        cv2.rectangle(panel, (badge_x1, badge_y1),
                      (badge_x2, badge_y2), badge_color, -1)
        cv2.putText(panel, badge_text, (badge_x1 + 10, badge_y2 - 8),
                    font, scale, (255, 255, 255), thickness)

        # Prediction Result
        cv2.rectangle(panel, (15, 120), (panel_width-15, 250), color_bg, -1)
        cv2.putText(panel, "PREDICTION:", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
        cv2.putText(panel, last_prediction, (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Score
        cv2.rectangle(panel, (15, 270), (panel_width-15, 350),
                      (200, 200, 200), -1)
        cv2.putText(panel, "SCORE:", (30, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1)
        cv2.putText(panel, score_text, (30, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

        # Status
        status_color = (0, 255, 0) if "✓" in status else (0, 0, 255)
        cv2.rectangle(panel, (15, 370),
                      (panel_width-15, 430), status_color, -1)
        cv2.putText(panel, status, (30, 405),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Instructions
        cv2.rectangle(panel, (15, 450), (panel_width-15,
                      panel_height-15), (220, 220, 220), -1)
        instructions = [
            "CONTROLS:",
            "",
            "[Q] - Quit",
            "[S] - Save prediction",
            "[C] - Capture full result",
            "",
            f"Frames: {frame_count}",
            f"Classes: {num_classes}",
            f"Threshold: {CONFIDENCE_THRESHOLD}"
        ]

        y_pos = 475
        for instruction in instructions:
            cv2.putText(panel, instruction, (25, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 1)
            y_pos += 30

        # Display windows
        cv2.imshow('CAMERA FEED', camera_window)
        cv2.imshow('CONTROL PANEL', panel)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n✓ Webcam test finished")
            break

        elif key == ord('s'):
            # Save just the camera frame
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/frame_{last_prediction}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved camera frame: {filename}")

        elif key == ord('c'):
            # Save both windows
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            camera_file = f"{output_folder}/camera_{last_prediction}_{timestamp}.jpg"
            cv2.imwrite(camera_file, camera_window)

            panel_file = f"{output_folder}/panel_{last_prediction}_{timestamp}.jpg"
            cv2.imwrite(panel_file, panel)

            print(f"✓ Saved: {camera_file}")
            print(f"✓ Saved: {panel_file}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Total frames processed: {frame_count}")
    print(f"✓ Results saved to: {output_folder}/")


def init_classifier(model_type: str):
    global classifier_type, knn, svm, scaler
    global KNOWN_DISTANCE_THRESHOLD
    global svm_centroids, svm_distance_thr, svm_prob_thr, svm_margin_thr
    global svm_known_classes, svm_unknown_class

    classifier_type = model_type
    if model_type == 'knn':
        model_path = KNN_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"KNN model pickle not found at {model_path}")
        data = joblib.load(model_path)
        knn = data["knn"]
        scaler = data["scaler"]
        KNOWN_DISTANCE_THRESHOLD = data.get("threshold", UNKNOWN_THRESHOLD)
        print(f"✓ Loaded KNN model: {model_path}")
        print(f"✓ KNN distance threshold: {KNOWN_DISTANCE_THRESHOLD}")
    elif model_type == 'svm':
        model_path = SVM_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SVM model pickle not found at {model_path}")
        data = joblib.load(model_path)
        svm = data["svm"]
        scaler = data["scaler"]
        svm_centroids = data.get("centroids", None)
        svm_distance_thr = data.get("distance_thr", None)
        svm_prob_thr = data.get("prob_thr", svm_prob_thr)
        svm_margin_thr = data.get("margin_thr", svm_margin_thr)
        svm_known_classes = data.get("known_classes", svm_known_classes)
        svm_unknown_class = data.get("unknown_class", svm_unknown_class)
        print(f"✓ Loaded SVM model: {model_path}")
        if svm_centroids is not None and svm_distance_thr is not None:
            print(
                f"✓ SVM open-set enabled (distance_thr={svm_distance_thr}, prob_thr={svm_prob_thr}, margin_thr={svm_margin_thr})")
        else:
            print("ℹ SVM open-set params not found; using standard SVM prediction")
    else:
        raise ValueError("model_type must be 'knn' or 'svm'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam material classifier (KNN or SVM)")
    parser.add_argument(
        "--model", choices=["knn", "svm"], default="knn", help="Classifier to use")
    args = parser.parse_args()
    try:
        init_classifier(args.model)
        test_webcam()
    except KeyboardInterrupt:
        print("\n✓ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
