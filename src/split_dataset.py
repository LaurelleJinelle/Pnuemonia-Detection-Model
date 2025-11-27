import os
import shutil
import random

# Path to your downloaded Kaggle dataset
SOURCE_DIR = "3_kinds_of_pneumonia"   # e.g. "3_kinds_of_pneumonia" or wherever you extracted it
DEST_DIR = "data"
TRAIN_SPLIT = 0.8        # 80% train, 20% test

# Create destination folders
train_dir = os.path.join(DEST_DIR, "train")
test_dir = os.path.join(DEST_DIR, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Create class subfolders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move training images
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    # Move test images
    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

print("Dataset split complete!")
