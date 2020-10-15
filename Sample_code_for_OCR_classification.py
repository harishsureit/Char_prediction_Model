import pandas as pd
import numpy as np
import cv2
import os

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Activation, MaxPool2D

#from matplotlib import pyplot as plt
#from segmentchars import plate_segmentation

# Load model
model = load_model('Generic_character_classifier.h5')
# Inverse one hot encoding
alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
             'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

classes = []
for a in alphabets:
    classes.append([a])
"""
Below is the sample code for testing the characters in the bounding box...
instead of reading we will be directly resizing the bounding box and reshaping it as in the below code
then predicting the class of the character in the bounding box,

The one thing that should be important is the bounding boxes should be sorted from left to right.. 
and should be sent to the model in the same order . 

"""
imgg = cv2.imread("generic_dataset\\R\\R_2.jpg", 0)
imgg = cv2.resize(imgg, (28, 28))

imgg = np.reshape(imgg, (1, 28, 28, 1))

predict = model.predict(imgg)
prediction = np.argmax(predict)
Char_predicted = classes[prediction]
# print(type(Char_predicted))  # list
print(Char_predicted[0])
