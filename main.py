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

epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print(val_ds)
print(train_ds)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

average_bad = 0
highest_bad = -1
lowest_bad = 1
for file in os.listdir("true-validation/bad"):
    img = keras.preprocessing.image.load_img(
        "true-validation/bad/" + file, target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    average_bad = average_bad + score
    if score < lowest_bad:
        lowest_bad = score
    if score > highest_bad:
        highest_bad = score


average_good = 0
highest_good = -1
lowest_good = 1
for file in os.listdir("true-validation/good"):
    img = keras.preprocessing.image.load_img(
        "true-validation/good/" + file, target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    average_good = average_good + score
    if score < lowest_good:
        lowest_good = score
    if score > highest_good:
        highest_good = score
        
        
print("Lowest bad: " + str(lowest_bad))
print("Average bad: " + str(average_bad/len(os.listdir("true-validation/bad"))))
print("Highest bad: " + str(highest_bad))

print("Lowest good: " + str(lowest_good))
print("Average good: " + str(average_good/len(os.listdir("true-validation/good"))))
print("Highest good: " + str(highest_good))




def get_labels_and_predictions():
    labels = np.array([])
    preds = np.array([])
    for file in os.listdir("true-validation/good"):
        img = keras.preprocessing.image.load_img(
            "true-validation/good/" + file, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        labels = np.append(labels, np.array([1]))
        preds = np.append(preds, (predictions > 0.5).astype(np.int_))
    
    for file in os.listdir("true-validation/bad"):
        img = keras.preprocessing.image.load_img(
            "true-validation/bad/" + file, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        labels = np.append(labels, np.array([0]))
        preds = np.append(preds, (predictions > 0.5).astype(np.int_))
        
    for file in os.listdir("augmented-data/good"):
        img = keras.preprocessing.image.load_img(
            "augmented-data/good/" + file, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        labels = np.append(labels, np.array([1]))
        preds = np.append(preds, (predictions > 0.5).astype(np.int_))
    
    for file in os.listdir("augmented-data/bad"):
        img = keras.preprocessing.image.load_img(
            "augmented-data/bad/" + file, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        labels = np.append(labels, np.array([0]))
        preds = np.append(preds, (predictions > 0.5).astype(np.int_))
        
    print(labels)
    print(preds)
    
    return (labels, preds)
        
(label, pred) = get_labels_and_predictions()

print(label.shape)
print(pred.shape)

cm = confusion_matrix(label, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()