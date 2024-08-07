import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import random  # Import random module

from Data_processing_functions import load_and_preprocess_image, preprocess_lab_image, load_and_preprocess

# Create dataset for test images
def create_test_dataset(gray_image_paths, target_size=(256, 256), batch_size=32):
    gray_image_paths = tf.constant(gray_image_paths, dtype=tf.string)
    
    # Define the dataset from image paths
    dataset = tf.data.Dataset.from_tensor_slices(gray_image_paths)
    
    # Map the dataset to preprocess images
    dataset = dataset.map(lambda gray_path: load_and_preprocess_image(gray_path, target_size, is_grayscale=True), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Function to generate colorful images
def generate_images(model, test_input):
    prediction = model(test_input, training=False)  # Set training=False for inference
    return prediction

# Function to convert LAB to RGB
def lab_to_rgb(lab_image):
    lab_image_np = lab_image.numpy() * 255.0  # Convert to 0-255 range for OpenCV
    lab_image_np = lab_image_np.astype(np.uint8)
    rgb_image = cv2.cvtColor(lab_image_np, cv2.COLOR_LAB2RGB)
    rgb_image = rgb_image / 255.0  # Normalize to [0, 1]
    return rgb_image

# Function to show images
def show_images(gray_images, color_images, predicted_images, num_images):
    plt.figure(figsize=(18, 12))  # Adjusted height for 3 rows

    for i in range(num_images):
        # Display grayscale images (first row)
        plt.subplot(3, num_images, i + 1)
        plt.imshow(tf.squeeze(gray_images[i]), cmap='gray')
        plt.axis('off')
        plt.title('Grayscale Image')

        # Display original color images (second row)
        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(tf.squeeze(color_images[i]))
        plt.axis('off')
        plt.title('Original Color Image')

        # Display predicted images (third row)
        predicted_rgb = lab_to_rgb(predicted_images[i])  # Convert LAB to RGB
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(predicted_rgb)
        plt.axis('off')
        plt.title('Predicted Color Image')

    plt.show()

# Define directories
black_test_dir = 'data/test_black'  # Update with your actual path
color_test_dir = 'data/test_color'   # Update with your actual path

# Load image paths
black_image_paths = glob.glob(os.path.join(black_test_dir, '*.jpg'))
color_image_paths = glob.glob(os.path.join(color_test_dir, '*.jpg'))

# Randomly select five image paths
black_images_test_paths = random.sample(black_image_paths, 5)
color_images_test_paths = random.sample(color_image_paths, 5)

# Ensure these paths are not empty
assert len(black_images_test_paths) > 0, "No black and white test images found."
assert len(color_images_test_paths) > 0, "No color test images found."



# Create the test dataset
test_dataset = create_test_dataset(black_images_test_paths, target_size=(256, 256), batch_size=32)

# Load the trained generator model
generator = tf.keras.models.load_model('./Trained_models/pix2pix_model_generator.keras')



# Collect images for display
predicted_images_list = []
original_images_list = []
gray_images_list = []

for gray_img_batch in test_dataset:
    generated_images = generate_images(generator, gray_img_batch)
    
    for i in range(generated_images.shape[0]):
        predicted_images_list.append(generated_images[i])
        
        # Get the grayscale image tensor from the dataset (assuming batches are consistent)
        gray_img = gray_img_batch[i]
        gray_images_list.append(gray_img)
        
        # Match the color image paths to generated images
        original_color_img_path = color_images_test_paths[i % len(color_images_test_paths)]
        original_color_img = load_and_preprocess_image(original_color_img_path, target_size=(256, 256), is_grayscale=False)
        original_images_list.append(original_color_img)

# Ensure the number of images to show is consistent
num_images = min(len(predicted_images_list), len(original_images_list), len(gray_images_list))
show_images(gray_images=gray_images_list, color_images=original_images_list, predicted_images=predicted_images_list, num_images=num_images)