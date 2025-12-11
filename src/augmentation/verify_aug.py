import os
import cv2
import matplotlib.pyplot as plt
import random


def verify_augmentation(augmented_dataset="data/augmented"):
    classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

    fig, axes = plt.subplots(len(classes), 2, figsize=(10, 3*len(classes)))

    if len(classes) == 1:
        axes = axes.reshape(1, -1)

    for idx, class_name in enumerate(classes):
        train_dir = os.path.join(augmented_dataset, "train", class_name)

        if not os.path.exists(train_dir):
            print(f"Warning: {train_dir} not found")
            continue

        images = [f for f in os.listdir(train_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not images:
            print(f"Warning: No images in {class_name}")
            continue

        orig_images = [img for img in images if "_aug" not in img]
        aug_images = [img for img in images if "_aug" in img]

        if orig_images:
            orig_path = os.path.join(train_dir, random.choice(
                orig_images[:5]))  
            img = cv2.imread(orig_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx, 0].imshow(img)
                axes[idx, 0].set_title(f"{class_name} - Original")
            else:
                axes[idx, 0].text(0.5, 0.5, "Failed to load",
                                  ha='center', va='center')
                axes[idx, 0].set_title(f"{class_name} - Error")
        else:
            axes[idx, 0].text(0.5, 0.5, "No original images",
                              ha='center', va='center')
            axes[idx, 0].set_title(f"{class_name} - No originals")

        axes[idx, 0].axis('off')

        if aug_images:
            aug_path = os.path.join(train_dir, random.choice(aug_images))
            img = cv2.imread(aug_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx, 1].imshow(img)
                axes[idx, 1].set_title(f"{class_name} - Augmented")
            else:
                axes[idx, 1].text(0.5, 0.5, "Failed to load",
                                  ha='center', va='center')
                axes[idx, 1].set_title(f"{class_name} - Error")
        else:
            axes[idx, 1].text(0.5, 0.5, "No augmented images",
                              ha='center', va='center')
            axes[idx, 1].set_title(f"{class_name} - No augmentations")

        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("AUGMENTED DATASET VERIFICATION")
    print("=" * 60)

    for split in ["train", "val"]:
        print(f"\n{split.upper()} SPLIT:")
        total = 0
        for class_name in classes:
            class_dir = os.path.join(augmented_dataset, split, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                orig_count = len([img for img in images if "_aug" not in img])
                aug_count = len([img for img in images if "_aug" in img])
                print(f"  {class_name:12} | Total: {len(images):4d} | "
                      f"Original: {orig_count:4d} | Augmented: {aug_count:4d}")
                total += len(images)
            else:
                print(f"  {class_name:12} | Directory not found")
        print(f"  {'TOTAL':12} | {total:4d} images")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    verify_augmentation()
