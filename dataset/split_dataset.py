import os
import shutil
import random

dataset_path = "C:/Users/Manoj/dsa38/project/dataset/"
train_path = "C:/Users/Manoj/dsa38/project/dataset/train/"
test_path = "C:/Users/Manoj/dsa38/project/dataset/test/"

#train & test directories
for category in ["covid", "non-covid"]:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

    images = os.listdir(os.path.join(dataset_path, category))
    random.shuffle(images)

    # 80% Train, 20% Test split
    train_size = int(0.8 * len(images))

    # Move images
    for img in images[:train_size]:
        shutil.move(os.path.join(dataset_path, category, img), os.path.join(train_path, category, img))
    for img in images[train_size:]:
        shutil.move(os.path.join(dataset_path, category, img), os.path.join(test_path, category, img))

print("Dataset split into train & test successfully!")
