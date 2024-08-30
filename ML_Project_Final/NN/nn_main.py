# NN/main_nn.py

import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
import keras

from tensorflow.keras.layers import Dense, Flatten, Dropout

# Path to the Gender Classification Dataset folder
dataset_path = "../Gender Classification Dataset"

# Load the full training data
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


# Define the first neural network model
def build_NN_1():
    model = keras.Sequential([
        Flatten(input_shape=(64, 64, 1)),  # 1 - prefer Input(shape)
        Dense(128, activation='relu'),
        Dropout(0.5),  # to avoid overfitting and provide more generalization
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define the second neural network model
def build_NN_2():
    model = keras.Sequential([
        Flatten(input_shape=(64, 64, 1)),  # 1 - prefer Input(shape)
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # sigmoid
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create models
model_1 = build_NN_1()
model_2 = build_NN_2()


# Train the first model
history1 = model_1.fit(train_images, train_labels, epochs=50, batch_size=32,
                       validation_data=(val_images, val_labels), verbose=1)

# Show the model architecture
model_1.summary()

# Train the second model
history2 = model_2.fit(train_images, train_labels, epochs=100, batch_size=64,
                       validation_data=(val_images, val_labels), verbose=1)

# Show the model architecture
model_2.summary()


# Plot the error and accuracy curves for the training and validation data
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return history.history['loss'], history.history['accuracy']


# Call plot
model1_error, model1_acc = plot_history(history1, 'Model 1')
model2_error, model2_acc = plot_history(history2, 'Model 2')

# Compare the models based on the validation accuracy
if max(model1_error) < max(model2_error):  # model 1 is better
    if max(model1_acc) > max(model2_acc):
        model_1.save('NN_Model_1.keras')
        bestModel = 'NN_Model_1.keras'
    else:
        model_2.save('NN_Model_2.keras')
        bestModel = 'NN_Model_2.keras'
else: # model2_error < model1error
    if max(model2_acc) > max(model1_acc):
        model_2.save('NN_Model_2.keras')
        bestModel = 'NN_Model_2.keras'
    else:
        model_1.save('NN_Model_1.keras')
        bestModel = 'NN_Model_1.keras'

# Load the best model
best_model = tf.keras.models.load_model(bestModel)
print(bestModel)  # to check if it is the best or not

# # Test the best model on the testing set and provide the confusion matrix and the average F1 scores
# test_predictions = best_model.predict(test_images)
# test_conf_matrix = confusion_matrix(test_labels, test_predictions)
# test_f1 = f1_score(test_labels, test_predictions, average='weighted')
#
# print("Testing Confusion Matrix:\n", test_conf_matrix)
# print("Testing Average F1 Score:", test_f1)
# Test the best model on the testing set and provide the confusion matrix and the average F1 scores
test_predictions = best_model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)  # Convert probabilities to class labels
test_conf_matrix = confusion_matrix(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

print("Testing Confusion Matrix:\n", test_conf_matrix)
print("Testing Average F1 Score:", test_f1)
