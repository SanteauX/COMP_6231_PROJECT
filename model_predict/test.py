from keras.models import Sequential
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import boto3
import pickle
import cv2
import base64
import keras
import os, shutil
import tensorflow as tf


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.utils.np_utils import to_categorical
from io import StringIO
from cv2 import resize
from sklearn.model_selection import train_test_split
from keras import layers
from keras import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


__S3_BUCKET_NAME = 'comp6231-demo-s3-bucket'
__DATASET_METADATA_CSV_FILENAME = 'HAM10000_metadata.csv'
__PICKLED_METADATA_FILENAME = 'metadata.pkl'
__MODEL_FILENAME = 'model.h5'

###################################### STEP 1: GET MODEL FROM S3 ##################################
try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(__S3_BUCKET_NAME)
        s3.Bucket(__S3_BUCKET_NAME).download_file(__MODEL_FILENAME, '/tmp/model.h5')
        model = keras.models.load_model("/tmp/model.h5") # To run on AWS
        #model = keras.models.load_model("model.h5")      # To run locally
        #print(f"successfully got model from S3")

except Exception as e:
    raise e

###################################### STEP 2: GET IMAGE READY FOR MODEL ##########################

image = tf.keras.preprocessing.image.load_img("test.jpg", target_size=(224, 224))       # Resize image
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])                                                               # Convert single image to a batch

###################################### STEP 3: PREDICT ############################################

predictions = model.predict(input_arr)
print("prediction: "+predictions)

###################################### STEP 4: RETURN RESULT ######################################
