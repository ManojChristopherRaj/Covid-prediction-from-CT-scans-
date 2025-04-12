import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from PIL import Image  


train_dir = "C:/Users/Manoj/dsa38/project/dataset/train"
test_dir = "C:/Users/Manoj/dsa38/project/dataset/test"
save_dir = "C:/Users/Manoj/dsa38/project/features/"


os.makedirs(save_dir, exist_ok=True)

# Load Pretrained ResNet50 Model 
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(directory):
    features = []
    labels = []

    class_map = {"covid": 1, "non-covid": 0}  


    for label in class_map:
        folder_path = os.path.join(directory, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Open image in RGB format
            img = Image.open(img_path).convert("RGB")  
            img = img.resize((224, 224))  

            # Convert to array
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  
            img_array = preprocess_input(img_array)  # Preprocess for ResNet50

            # Extract deep features
            feature = base_model.predict(img_array)
            features.append(feature.flatten())  
            labels.append(class_map[label])

    return np.array(features), np.array(labels)

# Extract Features 
train_features, train_labels = extract_features(train_dir)
test_features, test_labels = extract_features(test_dir)

# Save Features & Labels 
np.save(os.path.join(save_dir, "train_features.npy"), train_features)
np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
np.save(os.path.join(save_dir, "test_features.npy"), test_features)
np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)

print(" Feature extraction using ResNet50 completed! Files saved in:", save_dir)
