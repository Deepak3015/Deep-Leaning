import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print dataset shape
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Display a sample image
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()

# Normalize pixel values (from 0-255 to 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

