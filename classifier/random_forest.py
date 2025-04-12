import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


features_dir = "../features"

# Load training data
train_features = np.load(os.path.join(features_dir, "train_features.npy"))
train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))

# Load test data
test_features = np.load(os.path.join(features_dir, "test_features.npy"))
test_labels = np.load(os.path.join(features_dir, "test_labels.npy"))

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(train_features, train_labels)

predictions = rf_model.predict(test_features)

accuracy = accuracy_score(test_labels, predictions)
print(f"Random Forest Accuracy: {accuracy:.4f}")

model_path = "./random_forest.pkl"
joblib.dump(rf_model, model_path)
print(f"Model saved successfully at: {model_path}")

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Predict on the test set
y_pred = rf_model.predict(test_features)

cm = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(test_labels, y_pred))

print("Train Labels:", np.unique(train_labels, return_counts=True))
print("Test Labels:", np.unique(test_labels, return_counts=True))
