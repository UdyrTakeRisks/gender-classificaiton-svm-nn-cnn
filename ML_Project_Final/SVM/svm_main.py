# svm_main.py

import os

from matplotlib import pyplot as plt

from data_loader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm

# Path to the Gender Classification Dataset folder
dataset_path = "../Gender Classification Dataset"

# Load the full training data - Data Preparation
full_train_images, full_train_labels = load_dataset(os.path.join(dataset_path, "Training"))

# Split the full training data into new training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    full_train_images, full_train_labels, test_size=0.2, random_state=165)

# Load validation data
val_images, val_labels = load_dataset(os.path.join(dataset_path, "Validation"))

# Display some information about the loaded data - Data Exploration
print("Number of training images:", len(train_images))
print("Number of testing images:", len(test_images))
print("Number of validation images:", len(val_images))

print("images shape:", train_images[0].shape)

print("images labels:", train_labels[0])

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Reshape images for SVM (flatten each image to a 1D array (in one column))
train_images = np.array([image.reshape(-1) for image in tqdm(train_images, desc="Reshaping training images")])  #?
test_images = np.array([image.reshape(-1) for image in tqdm(test_images, desc="Reshaping testing images")])  #?
val_images = np.array([image.reshape(-1) for image in tqdm(val_images, desc="Reshaping validation images")])  #?

# train_images = np.array(train_images)
# test_images = np.array(test_images)
# val_images = np.array(val_images)

# Train an SVM model on the grayscale images
# svm_model = SVC(kernel='linear', random_state=42)  #
svm_model = SVC()
svm_model.fit(train_images, train_labels)
svm_model.fit(val_images, val_labels)  #

# Test the model on the new testing set and provide the confusion matrix and the average F1 scores
test_predictions = svm_model.predict(test_images)
test_conf_matrix = confusion_matrix(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

print("Testing Confusion Matrix:\n", test_conf_matrix)
print("Testing Average F1 Score:", test_f1)

# Validate the model on the validation set
# val_predictions = svm_model.predict(val_images)  # ?
# val_conf_matrix = confusion_matrix(val_labels, val_predictions)  # ?
# val_f1 = f1_score(val_labels, val_predictions, average='weighted')  # ?
#
# print("Validation Confusion Matrix:\n", val_conf_matrix)
# print("Validation Average F1 Score:", val_f1)
