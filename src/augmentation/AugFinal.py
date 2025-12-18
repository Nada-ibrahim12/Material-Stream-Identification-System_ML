import os
from pathlib import Path
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import numpy as np
from sklearn.model_selection import train_test_split

# Get dataset path and create augmented path
dataset = Path(__file__).resolve().parent.parent.parent / "data" / "dataset"
augmented = Path(__file__).resolve().parent.parent.parent / "data" / "augmented"

os.makedirs(augmented, exist_ok=True)

IMAGE_SIZE = (224, 224)
MIN_TRAIN_SIZE = 500   

# Random augmentation generator
image_generator = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20,
    fill_mode='nearest',
)

# Split data into train and test folders
for split in ['train', 'test']:
    split_dir = augmented / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for class_name in os.listdir(dataset):
        class_path = dataset / class_name
        if class_path.is_dir():
            (split_dir / class_name).mkdir(exist_ok=True)


for class_name in os.listdir(dataset):
    class_path = dataset / class_name
    if not class_path.is_dir():
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(images)

    # Split images into train and test and save them
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)

    train_dir = augmented / 'train' / class_name
    test_dir = augmented / 'test' / class_name

    for img_name in test_imgs:
        src = class_path / img_name
        dst = test_dir / img_name
        if not dst.exists():
            shutil.copy(src, dst)

    for img_name in train_imgs:
        src = class_path / img_name
        dst = train_dir / img_name
        if not dst.exists():
            shutil.copy(src, dst)


    current_train_count = len(os.listdir(train_dir))
    needed_aug = max(0, MIN_TRAIN_SIZE - current_train_count)

    print(f"Augmenting {needed_aug} images for class {class_name}")

    i = 0
    # Augment needed no of augmented images
    while needed_aug > 0:
        img_name = train_imgs[i % len(train_imgs)]
        i += 1

        img_path = class_path / img_name
        try:
            img = load_img(img_path, target_size=IMAGE_SIZE)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)

            aug_iter = image_generator.flow(arr, batch_size=1)
            aug_img = next(aug_iter)[0].astype('uint8')

            save_name = f"{Path(img_name).stem}_aug_{i:04d}.jpg"
            save_path = train_dir / save_name

            array_to_img(aug_img).save(save_path)
            needed_aug -= 1

        except Exception as e:
            continue            

print("Augmentation and splitting completed successfully!")
