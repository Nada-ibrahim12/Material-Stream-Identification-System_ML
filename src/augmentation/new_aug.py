import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

dataset = "data/dataset"
augmented = "data/augmented"
os.makedirs(augmented, exist_ok=True)

target_train_size = 500  
image_size = (224, 224)  

for split in ['train', 'val']:
    split_dir = os.path.join(augmented, split)
    os.makedirs(split_dir, exist_ok=True)
    for class_name in os.listdir(dataset):
        class_path = os.path.join(dataset, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

image_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    horizontal_flip=True,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode='nearest',
    shear_range=0.1,
)

for class_name in os.listdir(dataset):
    class_path = os.path.join(dataset, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) == 0:
        print(f"Warning: No images found in {class_name}")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)

    train_dir = os.path.join(augmented, 'train', class_name)
    val_dir = os.path.join(augmented, 'val', class_name)

    for img_name in val_imgs:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(val_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    for img_name in train_imgs:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(train_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    current_train_count = len(os.listdir(train_dir))
    needed_aug = target_train_size - current_train_count

    if needed_aug > 0:
        print(f"Augmenting {needed_aug} images for class {class_name}")
        i = 0
        while needed_aug > 0:
            img_name = train_imgs[i % len(train_imgs)]
            i += 1
            img_path = os.path.join(class_path, img_name)
            try:
                img = load_img(img_path, target_size=image_size)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                aug_iter = image_generator.flow(x, batch_size=1)
                aug_img = next(aug_iter)[0].astype('uint8')

                save_name = f"{os.path.splitext(img_name)[0]}_aug_{i:03d}.jpg"
                save_path = os.path.join(train_dir, save_name)
                array_to_img(aug_img).save(save_path)

                needed_aug -= 1
            except Exception as e:
                print(f"Error augmenting {img_name}: {e}")

print("Augmentation and splitting completed successfully!!!!")

