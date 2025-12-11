import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import albumentations as A

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

augmentor = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(var_limit=(10, 40), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    # RandomResizedCrop requires size=(H, W)
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.5),
])


def augment(img):
    aug = augmentor(image=img)["image"]
    return aug

for split in ['train', 'val']:
    split_dir = os.path.join(augmented, split)
    os.makedirs(split_dir, exist_ok=True)
    for class_name in os.listdir(dataset):
        class_path = os.path.join(dataset, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)


print("Starting Augmentation Process:")

TARGET_SIZE = 500

for class_name in os.listdir(dataset):
    class_path = os.path.join(dataset, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {class_name}")

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
        and valid_image(os.path.join(class_path, f))
    ]

    if len(images) == 0:
        print("No valid images — skipping")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    val_dir = os.path.join(augmented, 'val', class_name)
    for img_name in val_imgs:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(val_dir, img_name))

    train_dir = os.path.join(augmented, 'train', class_name)
    for img_name in train_imgs:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(train_dir, img_name))

    original_train_count = len(train_imgs)
    needed = TARGET_SIZE - original_train_count
    print(f"Original train count: {original_train_count}")
    print(f"Augmenting {needed} new images…")

    if needed > 0:
        i = 0
        created = 0

        while created < needed:
            img_name = train_imgs[i % original_train_count]
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            aug_img = augment(img)

            save_name = f"{os.path.splitext(img_name)[0]}_aug_{created:03d}.jpg"
            save_path = os.path.join(train_dir, save_name)

            cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            created += 1
            i += 1

            if created % 50 == 0:
                print(f"Generated {created}/{needed}")

    print(f"Done! Final train count: {len(os.listdir(train_dir))}")

print("Augmentation Completed Successfully!")

