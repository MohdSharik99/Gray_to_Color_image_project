import tensorflow as tf
import numpy as np
import cv2
import os
import glob

# Load and preprocess image
def load_and_preprocess_image(image_path, target_size, is_grayscale=False):
    img = tf.io.read_file(image_path)
    channels = 1 if is_grayscale else 3
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img

# Convert RGB image to LAB using OpenCV (requires numpy conversion)
def convert_to_lab(image):
    image_np = image.numpy() * 255.0  # Convert to 0-255 range for OpenCV
    image_np = image_np.astype(np.uint8)
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    lab_image = lab_image.astype(np.float32) / 255.0  # Normalize to 0-1
    return lab_image

# Wrapper to use convert_to_lab in TensorFlow pipeline
def preprocess_lab_image(image):
    lab_image = tf.py_function(func=convert_to_lab, inp=[image], Tout=tf.float32)
    lab_image.set_shape([None, None, 3])  # Set shape to avoid shape issues in TensorFlow
    return lab_image

# Load and preprocess images
def load_and_preprocess(gray_path, lab_path, target_size):
    gray_img = load_and_preprocess_image(gray_path, target_size, is_grayscale=True)
    rgb_img = load_and_preprocess_image(lab_path, target_size, is_grayscale=False)
    lab_img = preprocess_lab_image(rgb_img)
    return gray_img, lab_img

# Create separate datasets
def create_datasets(gray_image_paths, lab_image_paths, target_size=hyperparams['target_size'], batch_size=32):
    gray_image_paths = tf.constant(gray_image_paths, dtype=tf.string)
    lab_image_paths = tf.constant(lab_image_paths, dtype=tf.string)
    
    # Check if paths are correctly loaded
    if len(gray_image_paths) == 0 or len(lab_image_paths) == 0:
        raise ValueError("No image paths provided.")
    
    # Define grayscale dataset
    gray_dataset = tf.data.Dataset.from_tensor_slices(gray_image_paths)
    gray_dataset = gray_dataset.map(lambda path: load_and_preprocess_image(path, target_size, is_grayscale=True), 
                                    num_parallel_calls=tf.data.AUTOTUNE)
    
    # Define LAB dataset
    lab_dataset = tf.data.Dataset.from_tensor_slices(lab_image_paths)
    lab_dataset = lab_dataset.map(lambda path: preprocess_lab_image(load_and_preprocess_image(path, target_size, is_grayscale=False)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply dataset operations
    gray_dataset = gray_dataset.cache().shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    lab_dataset = lab_dataset.cache().shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return gray_dataset, lab_dataset
