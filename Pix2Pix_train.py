import tensorflow as tf
import numpy as np
import cv2
import os
import time
import glob
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from Data_processing_functions import *
from Pix2Pix_model import *

# Hyperparameters
hyperparams = {
    'initial_filters': 48,
    'kernel_size': 5,
    'num_layers': 5,
    'dropout_rate': 0.5,
    'batch_norm': True,
    'lambda_l1': 100,
    'learning_rate': 1e-3,
    'beta_1': 0.5,
    'batch_size': 32,
    'epochs': 50,
    'dropout': True,
    'input_shape': (256, 256, 1)
}

# Define Data Directory
dir_path = 'data'
color_dir = os.path.join(dir_path, 'train_color')
black_dir = os.path.join(dir_path, 'train_black')

# List all images
color_images_paths = glob.glob(os.path.join(color_dir, '*.jpg'))
black_images_paths = glob.glob(os.path.join(black_dir, '*.jpg'))

# Ensure there are images in both directories
if not color_images_paths or not black_images_paths:
    raise ValueError("No images found in one or both directories.")

# Create dataset
dataset = create_dataset(black_images_paths, color_images_paths, target_size=(256, 256), batch_size=hyperparams['batch_size'])

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=len(color_images_paths), reshuffle_each_iteration=True)

# Initialize models
generator = Generator(hyperparams)
discriminator = Discriminator(hyperparams)

# Print model summaries
print("Generator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

# Define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])

# Define the checkpoint directory
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Create a checkpoint object
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Restore the latest checkpoint if it exists
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print(f"Restored from checkpoint: {latest_checkpoint}")
else:
    print("Starting fresh training")

# Train the model
gen_losses, disc_losses = model_fit(dataset, hyperparams, checkpoint_prefix)

# Save the models
generator.save('./Trained_models/pix2pix_model_generator.keras')
discriminator.save('./Trained_models/pix2pix_model_discriminator.keras')

# Visualize losses
def visualize_losses(gen_losses, disc_losses):
    plt.figure(figsize=(12, 6))
    
    # Plot Generator and Discriminator Losses
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Losses')

    plt.tight_layout()
    plt.show()

visualize_losses(gen_losses, disc_losses)
