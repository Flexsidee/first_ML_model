import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile the neural network
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the neural network
model.fit(xs, ys, epochs=100)

# Use the model
y = model.predict([10.0])
print(f"When x = 10.0, the model predicts Y to be = {y}")  # the value x predicts the value y
