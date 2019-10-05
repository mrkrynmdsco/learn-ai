
import os
# Suppress TensorFlow debugging info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt


logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Feature
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# Label
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

print('\nLabeled Data:')
for i, c in enumerate(celsius_q):
    print("{} °C = {} °F".format(c, fahrenheit_a[i]))

# Build a layer
#   units       -- number of neurons in the layer
#   input_shape -- number of value input to the layer
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Assemble the layer into the model
#   order of layers in the list parameter -- input to output
model = tf.keras.Sequential([l0])

# Compile the model, with loss and optimizer functions
#   loss function       -- a way of measuring how far off predictions are from the desired outcome
#   optimizer function  -- a way of adjusting internal values in order to reduce the loss
#   learning rate       -- the step size taken when adjusting values in the model (0.001 ~ 0.1)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.9))

# Train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("\nFinished training the model")

# Display training statistics
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

# Used the model to predict values
print('\nTry: 100 °C:')
print('Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit'.format(model.predict([100.0])))
print('\nThe correct answer is 100 * 1.8 + 32 = 212')

# Looking at the layers weights
print('\nThese are the layer variables: \n{}'.format(l0.get_weights()))

# Show the plot
plt.show()
