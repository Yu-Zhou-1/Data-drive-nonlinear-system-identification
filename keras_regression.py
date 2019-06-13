import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prepare data
x_data = np.linspace(-0.5, 0.5, 2000)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation=tf.nn.tanh, input_shape=(1,)))
model.add(keras.layers.Dense(20, activation=tf.nn.tanh))
model.add(keras.layers.Dense(20, activation=tf.nn.tanh))
model.add(keras.layers.Dense(1))

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=tf.keras.losses.mae, metrics=['mae'])

history = model.fit(x_data, y_data, batch_size=100, epochs=1000)
# print(model.predict(x_data))
plt.figure(1)
plt.scatter(x_data, y_data)
plt.plot(x_data, model.predict(x_data),'r-',lw = 5)
plt.figure(2)
plt.plot(history.history['loss'])
plt.show()