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

processed_dir = "data/processed"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_THRESHOLD = 0.7

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
num_classes = len(class_names)

print(f"Using device: {device}")
print("Loading trained model and features...")

model = models.efficientnet_b0(weights='DEFAULT')
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

try:
    if os.path.exists(os.path.join(processed_dir, 'x_features_train.npy')):
        X_train = np.load(os.path.join(processed_dir, 'x_features_train.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_labels_train.npy'))
    else:
        X_train = np.load(os.path.join(processed_dir, 'x_features.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_labels.npy'))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(
        n_neighbors=7, weights='uniform', metric='cosine')
    knn.fit(X_train_scaled, y_train)

    print(f"✓ Model loaded successfully")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Classes: {class_names}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)


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
        distances, indices = knn.kneighbors(features.reshape(1, -1))
        distances = distances[0]

        # Average distance to K neighbors
        avg_distance = np.mean(distances)

        confidence = 1 / (1 + avg_distance)

        prediction = knn.predict(features.reshape(1, -1))[0]

        return int(prediction), confidence, avg_distance
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

            # Determine if unknown
            is_unknown = (confidence < CONFIDENCE_THRESHOLD) or (
                distance > UNKNOWN_THRESHOLD)

            if is_unknown or pred_class == -1:
                last_prediction = "UNKNOWN"
                last_confidence = distance
                color_bg = (0, 0, 255)
                score_text = f"Distance: {distance:.4f}"
                status = "⚠ LOW CONFIDENCE"
            else:
                last_prediction = class_names[pred_class]
                last_confidence = confidence
                color_bg = (0, 255, 0)
                score_text = f"Confidence: {confidence:.4f}"
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


if __name__ == "__main__":
    try:
        test_webcam()
    except KeyboardInterrupt:
        print("\n✓ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
