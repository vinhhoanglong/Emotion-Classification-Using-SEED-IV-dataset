# define some model to test with the data
import tensorflow as tf

def dense():
    return tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 1440)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(4)
])