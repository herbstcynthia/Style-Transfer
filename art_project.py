import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import tensorflow_hub as hub
from PIL import Image

#------------------------------------Functions used in the preprocessing------------------------------------

def image_scaler(image, max_dim = 512):

  # Casts a tensor to a new type.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

  # Creates a scale constant for the image
  scale_ratio = max_dim / max(original_shape)

  # Casts a tensor to a new type.
  new_shape = tf.cast(original_shape * scale_ratio, tf.int32)

  # Resizes the image based on the scaling constant generated above
  return tf.image.resize(image, new_shape)

def load_image(img_path):

  img = tf.io.read_file(img_path)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]
  return img
#----------------------- Converting image to numpy array------------------------

image = load_image('luci.jpeg') # arbitrary image that I chose
style_image = load_image('girassois.jpg') #arbitrary style that I chosen

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(image[0])
plt.title('Image')
plt.subplot(1, 2, 2)
plt.imshow(style_image[0])
plt.title('Style Image')

# Load Magenta's Arbitrary Image Stylization network from TensorFlow Hub  
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

# Pass content and style images as arguments in TensorFlow Constant object format
stylized_final_image = hub_module(tf.constant(image), tf.constant(style_image))[0]

# Set the size of the plot figure
plt.figure(figsize=(12, 12))

# Plot the stylized image
plt.imshow(stylized_final_image[0])

# Add title to the plot
plt.title('Stylized Final Image')

# Hide axes
plt.axis('off')

# Show the plot
plt.show()