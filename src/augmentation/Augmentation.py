import os
import cv2
import numpy as np
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil # for copying files

dataset = "data/dataset"
augmented = "data/augmented"
os.makedirs(augmented, exist_ok=True)

def valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def augment(img):
    h, w = img.shape[:2]

    scale = random.uniform(0.8, 1.3)   
    scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    if scale > 1:
        nh, nw = scaled.shape[:2]
        start_y = max(0, (nh - h) // 2)
        start_x = max(0, (nw - w) // 2)
        img = scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        nh, nw = scaled.shape[:2]
        pad_y = (h - nh) // 2
        pad_x = (w - nw) // 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[pad_y:pad_y+nh, pad_x:pad_x+nw] = scaled

    rotations = [90, 180, 270]
    rot_type = random.choice(rotations)
    if rot_type == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot_type == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot_type == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flip_type = random.choice([0, 1, -1])
    img = cv2.flip(img, flip_type)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + np.random.uniform(-5, 5)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] + np.random.uniform(-20, 20), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + np.random.uniform(-20, 20), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    std_dev = np.random.uniform(5, 15)
    gaussian = np.random.normal(0, std_dev, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + gaussian, 0, 255).astype(np.uint8)

    return img

for split in ['train', 'val']:
    split_dir = os.path.join(augmented, split)
    os.makedirs(split_dir, exist_ok=True)
    for class_name in os.listdir(dataset):
        class_path = os.path.join(dataset, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

print(f"\n{'='*50}")
print("Starting augmentation with train/val split")
print(f"{'='*50}\n")

for idx, class_name in enumerate(os.listdir(dataset), 1):
    class_path = os.path.join(dataset, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class {idx}: '{class_name}'")

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
        and valid_image(os.path.join(class_path, f))
    ]

    original_count = len(images)
    
    if original_count == 0:
        print(f"Warning: No valid images found in '{class_name}'. Skipping.")
        continue

    print(f"Found {original_count} valid images")

    # Split into train and validation (80/20)
    train_imgs, val_imgs = train_test_split(
        images, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"Split: {len(train_imgs)} train, {len(val_imgs)} validation")

    # Copy validation images (no augmentation)
    val_dir = os.path.join(augmented, 'val', class_name)
    for img_name in val_imgs:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(val_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Copy original training images
    train_dir = os.path.join(augmented, 'train', class_name)
    original_train_count = len(train_imgs)

    for img_name in train_imgs:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(train_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Calculate needed imgs
    target_train_size = 500
    target_aug = target_train_size - original_train_count
    aug_created = 0

    print(f"Original train images: {original_train_count}")
    print(f"Target train size: {target_train_size}")
    print(f"Augmentations needed: {target_aug}")

    if target_aug <= 0:
        print(f"No augmentation needed for class '{class_name}'")
        continue

    # Generate augmented images
    i = 0

    while aug_created < target_aug:
        img_name = train_imgs[i % original_train_count]
        i += 1

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        aug_img = augment(img)
        base = os.path.splitext(img_name)[0]

        save_path = os.path.join(train_dir, f"{base}_aug_{aug_created:03d}.jpg")
        cv2.imwrite(save_path, aug_img)

        aug_created += 1
        if aug_created % 50 == 0:
            print(f"  Generated {aug_created}/{target_aug} augmentations...")

    final_count = len(os.listdir(train_dir))
    print(f"✓ Class '{class_name}' completed: {final_count} training images")

print(f"\n{'='*50}")
print("Augmentation with train/val split completed successfully!")
print(f"{'='*50}")

for split in ['train', 'val']:
    print(f"\n{split.upper()} split:")
    for class_name in os.listdir(dataset):
        class_path = os.path.join(dataset, class_name)
        if os.path.isdir(class_path):
            split_dir = os.path.join(augmented, split, class_name)
            if os.path.exists(split_dir):
                count = len([f for f in os.listdir(split_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {class_name}: {count} images")
