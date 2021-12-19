import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import boto3
import pickle
import cv2
import base64
import keras
import os

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


__S3_BUCKET_NAME = 'comp6231-demo-s3-bucket'
__DATASET_METADATA_CSV_FILENAME = 'HAM10000_metadata.csv'
__PICKLED_METADATA_FILENAME = 'metadata.pkl'
__MODEL_FILENAME = 'model.h5'
s3 = boto3.client('s3')


def lambda_handler(event, context):

    if "numFilesProcess" not in event.keys() or event['numFilesProcess'] < 20:
        return {
            "statusCode" : 400,
            "headers" : {
                "Content-Type" : "application/json"
            },
            "body" : "Bad Request",
            "isBase64Encoded" : False
        }

    model = train(generate_metadata(), event['numFilesProcess'])

    return {
        "statusCode" : 200,
        "headers" : {
            "Content-Type" : "application/json"
        },
        "body" : {
            "numFilesProcessed" : event['numFilesProcess']
        },
        "isBase64Encoded" : False
    }

def pickle_metadata_to_s3(metadata, filename):
    pickle_byte_obj = pickle.dumps(metadata)
    s3.put_object(
        Bucket=__S3_BUCKET_NAME,
        Body=pickle_byte_obj,
        Key=filename
    )

def generate_metadata():
    csv_object = s3.get_object(Bucket=__S3_BUCKET_NAME, Key=__DATASET_METADATA_CSV_FILENAME)
    body = csv_object['Body']
    csv_string = body.read().decode('utf-8')
    df_skin = pd.read_csv(StringIO(csv_string))

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    lesion_ID_dict = {
        'nv': 0,
        'mel': 1,
        'bkl': 2,
        'bcc': 3,
        'akiec': 4,
        'vasc': 5,
        'df': 6
    }

    df_skin['lesion_type']=df_skin['dx'].map(lesion_type_dict)
    df_skin['lesion_ID'] = df_skin['dx'].map(lesion_ID_dict)

    pickle_metadata_to_s3(df_skin, __PICKLED_METADATA_FILENAME)
    
    return df_skin

def read_image(b64encodedImage):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64encodedImage), np.uint8), -1)

def produce_new_img(img):
    # produce new images by rotating of flipping the original one
    # this helps to increase the dimension of the dataset, avoiding overfitting of a single class
    imga = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    imgb = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgc = cv2.rotate(img,cv2.ROTATE_180)
    imgd = cv2.flip(img,0)
    imge = cv2.flip(img,1)

    return imga,imgb,imgc,imgd,imge

def train(df_skin, numFilesProcess, path=None):

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(__S3_BUCKET_NAME)

    allFiles = []

    X = []
    Y = []

    fileCounter = 0

    # examine dataset in s3
    for i, o in enumerate(bucket.objects.all()):
        if fileCounter == numFilesProcess:
            break
        if o.key.endswith('.jpg') or o.key.endswith('.png'):
            allFiles.append(o.key)
            fileCounter += 1
            if fileCounter % 10 == 0:
                print(f"{fileCounter} images loaded from S3")

    processedImageCounter = 0

    for i, file in enumerate(allFiles):
        fname_ID = os.path.splitext(os.path.basename(file))[0]
        img = s3.Object(__S3_BUCKET_NAME, file)
        img = img.get()['Body'].read()

        decoded = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
        img2 = resize(decoded,(100,100))
        X.append(img2)

        output = np.array(df_skin[df_skin['image_id'] == fname_ID].lesion_ID)
        Y.append(output[0])

        processedImageCounter += 1

        if output != 0:
            new_img = produce_new_img(img2) 
            for i in range(5):
                X.append(new_img[i])
                Y.append(output[0])

        if processedImageCounter % 10 == 0:
            print(f"{processedImageCounter} image(s) processed")

    X = np.array(X)
    Y = np.array(Y)

    X = X.astype("float32")
    X /= 255

    Y = to_categorical(Y)

    # start model stuff here
    data_train, data_test, labels_train, labels_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    pre_trained_model = VGG19(input_shape=(100, 100, 3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output
    
    # Flatten the output layer to 1 dimension
    a = layers.GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    a = layers.Dense(512, activation='relu')(a)
    # Add a dropout rate of 0.5
    a = layers.Dropout(0.5)(a)
    # Add a final sigmoid layer for classification
    a = layers.Dense(7, activation='softmax')(a)

    # Configure and compile the model

    # see if model exists in S3 already
    modelExists = False

    try:
        s3.Object(__S3_BUCKET_NAME, __MODEL_FILENAME).load()
        modelExists = True
    except Exception as e:
        pass

    model = None

    if modelExists:
        try:
            print(f"downloading model from S3 bucket=[{__S3_BUCKET_NAME}], key={__MODEL_FILENAME}")
            s3.Bucket(__S3_BUCKET_NAME).download_file(__MODEL_FILENAME, '/tmp/model.h5')
            model = keras.models.load_model("/tmp/model.h5")
        except Exception as e:
            raise
    else:
        model = Model(pre_trained_model.input, a)
    
    if model is not None:
        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

        train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

        train_datagen.fit(data_train)
        test_datagen = ImageDataGenerator()
        test_datagen.fit(data_test)

        batch_size = 10
        epochs = 1

        history = model.fit(train_datagen.flow(data_train,labels_train, batch_size=batch_size),
                                epochs = epochs, validation_data = test_datagen.flow(data_test, labels_test),
                                verbose = 1, steps_per_epoch=(data_train.shape[0] // batch_size), 
                                validation_steps=(data_test.shape[0] // batch_size))

        if numFilesProcess >= 500:
            for layer in pre_trained_model.layers:
                layer.trainable = True

            optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['acc'])

            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, 
                                                        min_lr=0.000001, cooldown=2)

            batch_size = 64
            epochs = 3
            history = model.fit(train_datagen.flow(data_train,labels_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = test_datagen.flow(data_test, labels_test),
                                        verbose = 1, steps_per_epoch=(data_train.shape[0] // batch_size), 
                                        validation_steps=(data_test.shape[0] // batch_size),
                                        callbacks=[learning_rate_reduction])
        
        return model