import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import tensorflow_hub as hub

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

def load_image(path_to_img):

  # Reads and outputs the entire contents of the input filename.
  img = tf.io.read_file(path_to_img)

  # Detect whether an image is a BMP, GIF, JPEG, or PNG, and 
  # performs the appropriate operation to convert the input 
  # bytes string into a Tensor of type dtype
  img = tf.image.decode_image(img, channels=3)

  # Convert image to dtype, scaling (MinMax Normalization) its values if needed.
  img = tf.image.convert_image_dtype(img, tf.float32)

  # Scale the image using the custom function we created
  img = image_scaler(img)

  # Adds a fourth dimension to the Tensor because
  # the model requires a 4-dimensional Tensor
  return img[tf.newaxis, :]

#---------------------------------------

image_path = tf.keras.utils.get_file('photo-1501820488136-72669149e0d4', 
                                       'https://images.unsplash.com/photo-1501820488136-72669149e0d4')
style_path = tf.keras.utils.get_file('Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg',
                                     'https://upload.wikimedia.org/wikipedia/commons/8/8c/Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg')

image = load_image(image_path)
style_image = load_image(style_path)

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