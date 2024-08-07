import tensorflow as tf
import numpy as np
import cv2
import os

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

# Create dataset
def create_dataset(gray_image_paths, lab_image_paths, target_size=(256, 256), batch_size=32):
    gray_image_paths = tf.constant(gray_image_paths, dtype=tf.string)
    lab_image_paths = tf.constant(lab_image_paths, dtype=tf.string)
    
    # Check if paths are correctly loaded
    if len(gray_image_paths) == 0 or len(lab_image_paths) == 0:
        raise ValueError("No image paths provided.")
    
    # Define the dataset from image paths
    dataset = tf.data.Dataset.from_tensor_slices((gray_image_paths, lab_image_paths))
    
    # Map the dataset to preprocess images
    def map_fn(gray_path, lab_path):
        gray_img, lab_img = tf.py_function(
            func=lambda x, y: load_and_preprocess(x, y, target_size),
            inp=[gray_path, lab_path],
            Tout=[tf.float32, tf.float32]
        )
        gray_img.set_shape([target_size[0], target_size[1], 1])
        lab_img.set_shape([target_size[0], target_size[1], 3])
        return gray_img, lab_img
    
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()  # Add caching
    dataset = dataset.shuffle(buffer_size=100)  # Adjust buffer size if needed
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset