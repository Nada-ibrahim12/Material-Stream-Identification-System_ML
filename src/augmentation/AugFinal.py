import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split

#Get dataset path and create augmented path
dataset = Path(__file__).resolve().parent.parent / "data" / "dataset"
augmented = Path(__file__).resolve().parent.parent / "data" / "augmented"
#dataset = "data/dataset"
#augmented = "data/augmented"
# if os.path.exists(augmented):
#     shutil.rmtree(augmented)
os.makedirs(augmented, exist_ok=True)

TRAIN_SIZE = 500  
IMAGE_SIZE = (224, 224)  

#Random augmentation generator
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

#split data into train and test folders
for split in ['train', 'val']:
    split_dir = os.path.join(augmented, split)
    os.makedirs(split_dir, exist_ok=True)
    for class_name in os.listdir(dataset):
        class_path = os.path.join(dataset, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)




for class_name in os.listdir(dataset):
    # print(f"\nProcessing class: '{class_name}'")
    class_path = os.path.join(dataset, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    

    #Split images into train and test and save them
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


    train_count = len(os.listdir(train_dir))
    needed_aug = TRAIN_SIZE - train_count


    print(f"Augmenting {needed_aug} images for class {class_name}")
    i = 0
    #Augment needed no of augmented images
    while needed_aug > 0:
        img_name = train_imgs[i % len(train_imgs)]
        i += 1
        img_path = os.path.join(class_path, img_name)
        try:
            img = load_img(img_path, target_size=IMAGE_SIZE)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            aug_iter = image_generator.flow(arr, batch_size=1)
            aug_img = next(aug_iter)[0].astype('uint8')
            save_name = f"{os.path.splitext(img_name)[0]}_aug_{i:03d}.jpg"
            save_path = os.path.join(train_dir, save_name)
            array_to_img(aug_img).save(save_path)
            needed_aug -= 1
        except Exception as e:
            print(f"Error augmenting {img_name}")

print("Augmentation and splitting completed successfully!!!!")
