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

model = keras.models.load_model("model")

image_size = (200, 290)

def prediction_stats(path):
    average = 0
    highest = -1
    lowest = 1
    for file in os.listdir(path):
        img = keras.preprocessing.image.load_img(
            path + file, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        average = average + score
        if score < lowest:
            lowest = score
        if score > highest:
            highest = score
            
    print("Lowest: " + str(lowest))
    print("Average: " + str(average/len(os.listdir(path))))
    print("Highest: " + str(highest))

prediction_stats("true-validation/bad/")
prediction_stats("true-validation/good/")

# Sorry for the ugly repetitive code.
# Just combining true-validation data
# with the training and testing data
# to generate the confusion matrix.
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
    
    return (labels, preds)
        
(label, pred) = get_labels_and_predictions()

cm = confusion_matrix(label, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()