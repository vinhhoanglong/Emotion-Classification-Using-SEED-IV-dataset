import tensorflow as tf
from tensorflow.keras import activations

def dense():
    return tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 1440)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(4)
])

def convlstm2d():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ConvLSTM2D(16, kernel_size=5, padding="same", return_sequences = False, input_shape=(64, 8, 9, 5)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(4))
    return model

def threedConv():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3D(32, kernel_size=5, padding="same", input_shape=(64, 8, 9, 20)))
    model.add(tf.keras.layers.BatchNormalization(trainable=False))
    model.add(tf.keras.layers.Activation(activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization(trainable=False))
    model.add(tf.keras.layers.Activation(activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4))
    return model
