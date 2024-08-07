import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.models import Model

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
    'batch_size': 1,
    'epochs': 50,
    'dropout': True,
    'input_shape': (256, 256, 1)
}

# Define the downsampling and upsampling blocks
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(hyperparams['dropout_rate']))
    result.add(tf.keras.layers.ReLU())
    return result

# Define the Generator model
def Generator(hyperparams):
    inputs = tf.keras.layers.Input(shape=hyperparams['input_shape'])

    down_stack = [downsample(hyperparams['initial_filters'] * (2 ** i), hyperparams['kernel_size'], apply_batchnorm=hyperparams['batch_norm']) for i in range(hyperparams['num_layers'])]
    up_stack = [upsample(hyperparams['initial_filters'] * (2 ** i), hyperparams['kernel_size'], apply_dropout=hyperparams['dropout']) for i in range(hyperparams['num_layers']-1, 0, -1)]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, hyperparams['kernel_size'], strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.concatenate([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# Define the Discriminator model
def Discriminator(hyperparams):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=hyperparams['input_shape'], name='input_image')
    tar = tf.keras.layers.Input(shape=[*hyperparams['input_shape'][:2], 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# Loss functions
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (hyperparams['lambda_l1'] * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# PSNR metric
def psnr_metric(y_true, y_pred):
    max_pixel = 1.0
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=max_pixel)
    return tf.reduce_mean(psnr_value)

# Training step function
def train_step(generator, discriminator, input_image, target, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_total_loss, disc_loss

# Training function without validation
def model_fit(train_ds, hyperparams, checkpoint_prefix):
    gen_losses = []
    disc_losses = []

    generator = Generator(hyperparams)
    discriminator = Discriminator(hyperparams)

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])

    # Define checkpoints
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    for epoch in range(hyperparams['epochs']):
        start = time.time()
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        # Progress bar
        progbar = tf.keras.utils.Progbar(len(train_ds), stateful_metrics=['loss'])

        for step, (input_image, target) in enumerate(train_ds):
            gen_total_loss, disc_loss = train_step(generator, discriminator, input_image, target, generator_optimizer, discriminator_optimizer)
            epoch_gen_loss += gen_total_loss
            epoch_disc_loss += disc_loss

            # Update progress bar
            progbar.update(step + 1, [('gen_loss', gen_total_loss), ('disc_loss', disc_loss)])

        gen_losses.append(epoch_gen_loss / len(train_ds))
        disc_losses.append(epoch_disc_loss / len(train_ds))

        # Save checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch+1}, Gen Loss: {gen_losses[-1]}, Disc Loss: {disc_losses[-1]}, Time: {time.time() - start}')

        # Early stopping logic
        if len(gen_losses) > 1 and gen_losses[-1] > gen_losses[-2]:
            print("Early stopping triggered")
            break

    return gen_losses, disc_losses

# Visualize losses
def visualize_losses(gen_losses, disc_losses):
    plt.figure(figsize=(12, 6))
    
    # Plot Generator and Discriminator Losses
    plt.subplot(1, 1, 1)
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Losses')

    plt.tight_layout()
    plt.show()