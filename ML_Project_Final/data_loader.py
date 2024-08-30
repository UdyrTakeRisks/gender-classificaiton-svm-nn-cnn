# data_loader.py

import os
import cv2
import numpy as np
from tqdm import tqdm


def load_images_from_folder(folder_path, label):
    """Load images from a specific folder and assign a label"""
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading {label} images from {folder_path}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image file extensions
            image_path = os.path.join(folder_path, filename)
            # Data Preparation
            # Read the image and convert to grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Resize image to (64, 64)
            image = cv2.resize(image, (64, 64))
            # Normalize image
            image = image / 255.0
            images.append(image)
            labels.append(label)  # Use numerical label (0 for male, 1 for female)
    return images, labels


def load_dataset(dataset_path):
    """Load images from the dataset folders"""
    images = []
    labels = []

    for folder in tqdm(os.listdir(dataset_path), desc="Loading dataset"):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Assign numerical label based on folder name
            label = 0 if folder == 'male' else 1
            folder_images, folder_labels = load_images_from_folder(folder_path, label)
            images.extend(folder_images)
            labels.extend(folder_labels)

    return np.array(images), np.array(labels)


def load_rgb_images_from_folder(folder_path, label):
    """Load images from a specific folder and assign a label"""
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading {label} images from {folder_path}"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image file extensions
            image_path = os.path.join(folder_path, filename)
            # Data Preparation
            # Read the image
            image = cv2.imread(image_path)
            # Resize image to (64, 64)
            image = cv2.resize(image, (64, 64))
            # Normalize image
            image = image / 255.0
            images.append(image)
            labels.append(label)  # Use numerical label (0 for male, 1 for female)
    return images, labels


def load_rgb_images(dataset_path):
    """Load images from the dataset folders"""
    images = []
    labels = []

    for folder in tqdm(os.listdir(dataset_path), desc="Loading dataset"):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Assign numerical label based on folder name
            label = 0 if folder == 'male' else 1
            folder_images, folder_labels = load_rgb_images_from_folder(folder_path, label)
            images.extend(folder_images)
            labels.extend(folder_labels)

    return np.array(images), np.array(labels)
