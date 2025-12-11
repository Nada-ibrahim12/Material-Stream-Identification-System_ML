## Directory Organization

```
Machine-Learning-Project/
│
├── data/
│   ├── dataset/                    # Original raw images
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   │
│   ├── augmented/                  # Augmented images with train/val split
│   │   ├── train/
│   │   │   ├── cardboard/
│   │   │   ├── glass/
│   │   │   ├── metal/
│   │   │   ├── paper/
│   │   │   ├── plastic/
│   │   │   └── trash/
│   │   └── val/
│   │       ├── cardboard/
│   │       ├── glass/
│   │       ├── metal/
│   │       ├── paper/
│   │       ├── plastic/
│   │       └── trash/
│   │
│   └── processed/                  # Extracted feature files (.npy)
│       ├── x_features_train.npy
│       ├── y_labels_train.npy
│       ├── x_features_val.npy
│       ├── y_labels_val.npy
│       ├── x_features.npy          # Combined (backward compatible)
│       └── y_labels.npy            # Combined (backward compatible)
│
├── src/
│   ├── augmentation/
│   │   └── Augmentation.py         # Reads from data/dataset, writes to data/augmented
│   └── feature_extraction/
│       └── new_extr.py             # Reads from data/augmented, writes to data/processed
│
|                        
└── src/
    └── KNN.py                  
```

## Data Pipeline Flow

```
data/dataset/               (Raw images)
     ↓
[Augmentation.py]
     ↓
data/augmented/train/       (Augmented training images)
data/augmented/val/         (Validation images)
     ↓
[new_extr.py]
     ↓
data/processed/
├── x_features_train.npy
├── y_labels_train.npy
├── x_features_val.npy
├── y_labels_val.npy
├── x_features.npy (optional)
└── y_labels.npy (optional)
     ↓
[KNN.py]
     ↓
Classification Results
```
