# NN/main_nn.py

import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dataset, load_rgb_images
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import keras

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

def build_CNN(input_shape):  # Gray or RGB
    model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # to avoid overfitting and provide more generalization
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])  # loss='categorical_crossentropy' ??
    return model


# Create models
# gray_images
# cnn_gray = build_CNN(train_images.shape[1:])
cnn_gray = build_CNN((64, 64, 1))
print(train_images.shape[1:])  # debug

full_train_rgb_images, full_train_rgb_labels = load_rgb_images(os.path.join(dataset_path, "Training"))

# rgb_images
train_rgb_images, test_rgb_images, train_rgb_labels, test_rgb_labels = train_test_split(
    full_train_rgb_images, full_train_rgb_labels, test_size=0.2, random_state=165)

val_rgb_images, val_rgb_labels = load_rgb_images(os.path.join(dataset_path, "Validation"))

# cnn_rgb = build_CNN(train_rgb_images.shape[1:])
cnn_rgb = build_CNN((64, 64, 3))
print(train_rgb_images.shape[1:])  # debug

# Train the first model
history1 = cnn_gray.fit(train_images, train_labels, epochs=50, batch_size=32,
                        validation_data=(val_images, val_labels), verbose=1)

# Show the model architecture
cnn_gray.summary()

# Train the second model
history2 = cnn_rgb.fit(train_rgb_images, train_rgb_labels, epochs=50, batch_size=32,
                       validation_data=(val_rgb_images, val_rgb_labels), verbose=1)

# Show the model architecture
cnn_rgb.summary()


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
model1_error, model1_acc = plot_history(history1, 'CNN Gray')  # model 1 === cnn gray
model2_error, model2_acc = plot_history(history2, 'CNN RGB')  # model 2 === cnn rgb

# Compare the models based on the validation accuracy
if max(model1_error) < max(model2_error):  # model 1 is better
    if max(model1_acc) > max(model2_acc):
        cnn_gray.save('CNN_Gray.keras')
        bestModel = 'CNN_Gray.keras'
    else:
        cnn_rgb.save('CNN_RGB.keras')
        bestModel = 'CNN_RGB.keras'
else:
    if max(model2_acc) > max(model1_acc):
        cnn_rgb.save('CNN_RGB.keras')
        bestModel = 'CNN_RGB.keras'
    else:
        cnn_gray.save('CNN_Gray.keras')
        bestModel = 'CNN_Gray.keras'

# Load the best model
best_model = tf.keras.models.load_model(bestModel)
print(bestModel)  # debug to check if it is the best or not


if best_model == 'CNN_Gray.keras':
    test_predictions = best_model.predict(test_images)
else:  # 'CNN_RGB.keras'
    test_predictions = best_model.predict(test_rgb_images)

test_predictions = np.argmax(test_predictions, axis=1)  # Convert probabilities to class labels
test_conf_matrix = confusion_matrix(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

print("Testing Confusion Matrix:\n", test_conf_matrix)
print("Testing Average F1 Score:", test_f1)
