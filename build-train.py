import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np

print(tf. __version__) 

tf.config.set_visible_devices([], 'GPU')


image_size = (200, 290)
batch_size = 2
epochs = 20

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "augmented-data",
    validation_split=0.2,
    subset="training",
    label_mode="binary",
    seed=2,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "augmented-data",
    validation_split=0.2,
    subset="validation",
    label_mode="binary",
    seed=2,
    image_size=image_size,
    batch_size=batch_size,
)

model = Sequential()
model.add(Conv2D(32,5,padding="same", activation="relu", input_shape=(200,290,3)))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(32,3,padding="same", activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(256, 3, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds, epochs=epochs, validation_data=val_ds,
)

model.save('model')

